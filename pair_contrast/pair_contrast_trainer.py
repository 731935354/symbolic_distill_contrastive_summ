import os
import json
import math
import time

from torch.nn import CrossEntropyLoss
from transformers.trainer_pt_utils import nested_numpify, nested_concat, IterableDatasetShard, nested_truncate, \
    find_batch_size
from transformers.trainer_utils import has_length, EvalPrediction, denumpify_detensorize, EvalLoopOutput, speed_metrics
from transformers.training_args import TrainingArguments
from dataclasses import dataclass, field


from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from torch import nn
from torch.utils.data import DistributedSampler, RandomSampler, DataLoader, Dataset

from transformers import logging, Seq2SeqTrainer

logger = logging.get_logger(__name__)


class ContrastiveTrainer(Seq2SeqTrainer):
    def __init__(
            self,
            theta,
            model = None,
            args = None,
            data_collator = None,
            train_dataset = None,
            eval_dataset = None,
            tokenizer = None,
            model_init = None,
            compute_metrics = None,
            callbacks = None,
            optimizers = (None, None),
            ori_validation_dataset=None,
            data_collator_for_val_test=None,
            test_dataset=None,
            ori_dev_test_batch_size=None
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers
        )
        self.theta = theta

        self._max_length = 128
        self._num_beams = 8

        self.ori_validation_dataset = ori_validation_dataset
        self.data_collator_for_val_test = data_collator_for_val_test
        self.test_dataset = test_dataset
        self.ori_dev_test_batch_size = ori_dev_test_batch_size

        self.config = self.model.config

    def _compute_loss(self, model, inputs):
        labels = inputs.pop("labels")  # size=(14, 91) when batch_size=2. What is 14 here?
        ce_pos = inputs.pop("ce_pos")  # size=2 when batch size=2 [the index of reference summaries in this batch]
        positive_contrast = inputs.pop("positive_contrast")
        valid_contrast = inputs.pop("valid_contrast")

        model_output = model(**inputs, use_cache=False)
        logits = model_output.logits

        ce_logits = logits[ce_pos]  # ce_pos is the index tensor, for choosing reference summaries
        ce_targets = labels[ce_pos]

        if self.label_smoother:
            loss = self.label_smoother((ce_logits,), ce_targets)
        else:
            lm_loss_fct = CrossEntropyLoss(reduction='mean')
            loss = lm_loss_fct(ce_logits.view(-1, logits.size(-1)), ce_targets.view(-1))

        # record lm_loss, logits and labels in trainer state
        self.state.lm_loss = loss.detach().clone()
        if torch.isnan(loss):
            logger.warning(f"lm loss is NaN at step {self.state.global_step}")
        self.state.lm_loss_logits = ce_logits.detach().clone()
        self.state.lm_loss_labels = ce_targets.detach().clone()

        # let N represent the total number of positive and negative summaries in the batch
        # representation.size = N x seqlen x dim (contextualized embedding dimension)
        representation = model_output.contrast_states
        ne_representation = representation.masked_fill((labels == -100).unsqueeze(-1), 0)  # N x seq_len x dim
        representation = ne_representation.sum(dim=1)  # size=(N, dim) sequence level representation by sum (?)
        representation_ne_denom = (labels != -100).sum(dim=1, keepdim=True)  # size=(N,1) length of each label sequence
        representation = representation / torch.max(representation_ne_denom,  # What is happening here?
                                                    1e-8 * torch.ones_like(representation_ne_denom))

        representation_n = representation.norm(dim=-1, keepdim=True)  # size=(N, 1)  L2 norm by default
        representation_norm = representation / torch.max(representation_n, 1e-8 * torch.ones_like(representation_n))
        similarity = torch.matmul(representation_norm, representation_norm.transpose(0, 1))  # pos+neg x pos+neg, size=(N,N)
        similarity = similarity.exp()
        similarity = similarity.masked_fill(~valid_contrast, 0.)
        denominator = similarity.sum(dim=-1, keepdim=True)  # pos+neg  size=(N,1)

        denom_similarity = similarity / torch.max(denominator, 1e-8 * torch.ones_like(denominator))  # pos+neg x pos+neg, size=(N,N)
        contrast_loss = denom_similarity[positive_contrast]
        contrast_loss = - contrast_loss.log()
        log_contrastive_loss = contrast_loss.detach().clone()
        contrast_loss_denom = positive_contrast.sum()

        contrast_loss = contrast_loss.sum() / torch.max(contrast_loss_denom,
                                                        1e-8 * torch.ones_like(contrast_loss_denom))

        # log the intermediate steps of the contrastive loss, for debugging loss=NaN
        self.state.denon_similarity = denom_similarity.detach().clone()
        self.state.representation = representation.detach().clone()
        self.state.representation_norm = representation_norm.detach().clone()
        self.state.similarity = similarity.detach().clone()
        self.state.denominator = denominator.detach().clone()
        self.state.log_contrastive_loss = log_contrastive_loss

        total_loss = loss + self.theta * contrast_loss
        self.state.contrast_loss = contrast_loss.detach().clone()
        self.state.total_loss = total_loss.detach().clone()

        return total_loss, logits

    def compute_loss(self, model, inputs):
        loss, _ = self._compute_loss(model, inputs)
        return loss

    def _pad_tensors_to_max_len(self, tensor, max_length):
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.eos_token_id

        if pad_token_id is None:
            raise ValueError(
                f"Make sure that either `config.pad_token_id` or `config.eos_token_id` is defined if tensor has to be padded to `max_length`={max_length}"
            )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()
        gen_kwargs = self._gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = 128
        gen_kwargs["num_beams"] = 4
        default_synced_gpus = False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
                "labels" in inputs
                and "decoder_input_ids" in inputs
                and inputs["labels"].shape == inputs["decoder_input_ids"].shape
        ):
            inputs_for_gen = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}
        generated_tokens = self.model.generate(**inputs_for_gen, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        if "src_select_indices" in inputs:  # for training / contrastive dev set
            with torch.no_grad():
                # compute loss on predict data
                loss, _ = self._compute_loss(model, inputs)
            loss = loss.mean().detach()
        else:  # for original validation set and test set
            loss = None

        labels = None
        return (loss, generated_tokens, labels)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        # if self.is_deepspeed_enabled and self.model_wrapped is self.model:
        #     _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        # if is_torch_tpu_available():
        #     dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            # show the input documents and generated summaries, to make sure the training process is correct
            if step == 0:
                # XXX: adapt synced_gpus for fairscale as well
                gen_kwargs = {
                    "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
                    "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
                    "synced_gpus": False,
                }

                if self.tokenizer is not None:
                    # some fileds got filtered out here
                    generation_inputs = {k: v for k, v in inputs.items() if k in self.tokenizer.model_input_names}
                    # very ugly hack to make it work
                    generation_inputs["input_ids"] = generation_inputs.pop(self.tokenizer.model_input_names[0])
                else:
                    generation_inputs = {"input_ids": inputs["input_ids"]}
                print("source article:")
                print(self.tokenizer.batch_decode(generation_inputs["input_ids"]))

                if torch.cuda.is_available():
                    generation_inputs["input_ids"] = generation_inputs["input_ids"].to('cuda')
                    generation_inputs["attention_mask"] = generation_inputs["attention_mask"].to('cuda')

                generated_tokens = self.model.generate(
                    **generation_inputs,
                    **gen_kwargs,
                )
                decoded_generations = self.tokenizer.batch_decode(generated_tokens)
                print("generation")
                print(decoded_generations)

            # if is_torch_tpu_available():
            #     xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        print("=> Begin Evaluation Loop")
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        # if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
        #     # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
        #     xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        # save generated summaries to disk
        summary_file = os.path.join(self.args.output_dir, f"STEP-{self.state.global_step}_generated_summaries.json")
        # Replace -100s used for padding as we can't decode them
        preds = output.predictions
        tokenizer = self.tokenizer
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # post process
        decoded_preds = [pred.strip().replace("\n", " ") for pred in decoded_preds]  # ensure no newline char in output
        # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
        with open(summary_file, "w") as file:
            json.dump(decoded_preds, file, indent=4)

        logger.info("=== Generate Summaries on Test Set ===")
        # predict summaries on test set
        test_dataloader = self.get_dataloader_for_val_test(self.test_dataset)
        test_output = eval_loop(
            test_dataloader,
            description="Test",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
            metric_key_prefix="test",
        )
        # save generated summaries to disk
        test_summary_file = os.path.join(self.args.output_dir, f"STEP-{self.state.global_step}_test_generated_summaries.json")
        # Replace -100s used for padding as we can't decode them
        test_preds = test_output.predictions
        tokenizer = self.tokenizer
        test_preds = np.where(test_preds != -100, test_preds, tokenizer.pad_token_id)
        test_decoded_preds = tokenizer.batch_decode(test_preds, skip_special_tokens=True)

        # post process
        test_decoded_preds = [pred.strip().replace("\n", " ") for pred in test_decoded_preds]  # ensure no newline char in output
        # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
        with open(test_summary_file, "w") as file:
            json.dump(test_decoded_preds, file, indent=4)

        # =======================================
        #  predict summaries on original dev set
        # =======================================
        logger.info("=== Generate Summaries on Ori Dev Set")
        ori_val_dataloader = self.get_dataloader_for_val_test(self.ori_validation_dataset)
        output = eval_loop(
            ori_val_dataloader,
            description="Original Validation Set",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
            metric_key_prefix="ori_val",
        )
        # save generated summaries to disk
        summary_file = os.path.join(
            self.args.output_dir,
            f"STEP-{self.state.global_step}_oridev_generated_summaries.json"
        )
        # Replace -100s used for padding as we can't decode them
        preds = output.predictions
        tokenizer = self.tokenizer
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # post process
        decoded_preds = [pred.strip().replace("\n", " ") for pred in decoded_preds]  # ensure no newline char in output
        # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
        with open(summary_file, "w") as file:
            json.dump(decoded_preds, file, indent=4)

        return output.metrics

    def get_dataloader_for_val_test(self, dataset):
        eval_dataset = dataset
        eval_dataset = eval_dataset.remove_columns(["id", "text", "summary"])

        data_collator = self.data_collator_for_val_test

        # if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
        #     eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        # else:
        #     data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.ori_dev_test_batch_size,  # a different batch size for faster inference on original dev/test set
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )