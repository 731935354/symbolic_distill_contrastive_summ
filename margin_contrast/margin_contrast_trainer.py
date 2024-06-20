"""
torch 2.0.1
transformers 4.31.0
peft 0.5.0.dev0
"""
import json
import math
import os
import time
import datasets

import torch
import numpy as np

from typing import *

from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, is_torch_tpu_available, PretrainedConfig, TrainerState, Seq2SeqTrainer, \
    is_datasets_available
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME, is_apex_available, is_sagemaker_dp_enabled, is_sagemaker_mp_enabled
)
from transformers.trainer_pt_utils import nested_numpify, nested_concat, find_batch_size, IterableDatasetShard, \
    nested_truncate
from transformers.trainer_utils import has_length, EvalLoopOutput, EvalPrediction, denumpify_detensorize, speed_metrics
from transformers.utils import logging

if is_sagemaker_dp_enabled():
    import smdistributed.dataparallel.torch.distributed as dist
else:
    import torch.distributed as dist

if is_apex_available():
    from apex import amp

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


logger = logging.get_logger(__name__)


class SeqUnlikelihoodTrainer(Seq2SeqTrainer):
    def __init__(
            self,
            model = None,
            args = None,
            data_collator = None,
            data_collator_for_val_test=None,
            train_dataset = None,
            eval_dataset = None,
            ori_validation_dataset = None,
            test_dataset = None,
            tokenizer = None,
            model_init = None,
            compute_metrics = None,
            callbacks = None,
            optimizers = (None, None),
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
        self._max_length = 128
        self._num_beams = 8

        self.ori_validation_dataset = ori_validation_dataset
        self.data_collator_for_val_test = data_collator_for_val_test
        self.test_dataset = test_dataset
        self.ori_dev_test_batch_size = ori_dev_test_batch_size

    def _remove_unused_columns(self, dataset, description: Optional[str] = None):
        pass

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, _, _, model_output = self._compute_loss(model, inputs)
        return loss

    def _compute_loss(self, model, inputs):
        # TODO: What is "inputs"? Is it a batch? -> yes

        # uncomment this block to monitor gpu usage/cpu memory/cpu time/gpu time, .etc. of all function calls
        # can be very slow. unblock when finished debugging
        # with torch.profiler.profile(profile_memory=True, record_shapes=True) as prof:
        #     model_output = model(**inputs, use_cache=False)  # some heavy computations potentially causing OOM
        # print(">>> profiler details:")
        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

        model_output = model(**inputs, use_cache=False)

        # shape: N x seq_len x vocab_size
        # N is the total number of positive and negative target summaries in a batch
        # usually N is larger than batch_size since there can be multiple positive
        # and negative target summaries for each sample (dialogue) in the batch
        lm_logits = model_output.logits

        # calculate the log likelihood for each target sequence (which is the negative cross entropy loss)
        loss = None
        lm_loss = None
        batch_size = inputs['input_ids'].size(0)
        margin_loss = torch.zeros(batch_size, device=inputs['input_ids'].device)
        theta = self.args.theta  # the margin in margin loss
        batch_pos_tgt_scores = []
        batch_neg_tgt_scores = []
        batch_min_pos_score = []
        batch_max_neg_score = []
        if 'labels' in inputs:
            logits_permuted = lm_logits.permute(0, 2, 1)
            # print(f"logits_permuted size: {logits_permuted.size()}")
            loss_fct = CrossEntropyLoss(reduction='none')
            log_probs = -loss_fct(logits_permuted, inputs['labels']).mean(dim=1)
            # ==============
            #   margin loss
            # ==============
            for i in range(batch_size):
                target_scores_for_sample = log_probs[inputs['src_select_indices'] == i]
                pos_neg_for_sample = inputs['pos_neg'][inputs['src_select_indices'] == i]
                pos_tgt_scores = target_scores_for_sample[pos_neg_for_sample == 1]
                neg_tgt_scores = target_scores_for_sample[pos_neg_for_sample == 0]
                min_pos_score = pos_tgt_scores.min()
                max_neg_score = neg_tgt_scores.max()
                # margin loss
                sample_margin_loss = max(0, theta + max_neg_score - min_pos_score)
                margin_loss[i] = sample_margin_loss

                batch_min_pos_score.append(min_pos_score.detach())
                batch_max_neg_score.append(max_neg_score.detach())
                batch_pos_tgt_scores.append(pos_tgt_scores.detach())
                batch_neg_tgt_scores.append(neg_tgt_scores.detach())

            # add likelihood information in trainer state
            self.state.batch_pos_tgt_scores = batch_pos_tgt_scores
            self.state.batch_neg_tgt_scores = batch_neg_tgt_scores
            self.state.batch_min_pos_score = batch_min_pos_score
            self.state.batch_max_neg_score = batch_max_neg_score
            self.state.margin_loss = margin_loss.detach().clone()

            # =================
            #   cross entropy
            # =================
            ref_summs_pred = lm_logits[inputs["ref_summ_idx"] == 1]
            ref_labels = inputs["labels"][inputs["ref_summ_idx"] == 1]
            lm_loss_fct = CrossEntropyLoss(reduction='mean')
            lm_loss = lm_loss_fct(ref_summs_pred.view(-1, lm_logits.size(-1)), ref_labels.view(-1))

            loss = lm_loss + self.args.alpha * margin_loss.mean()
            self.state.model_loss = loss.detach().clone()

        return loss, lm_loss, margin_loss, model_output

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
            inputs_for_gen = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
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
                loss, lm_loss, margin_loss, model_output = self._compute_loss(model, inputs)
                # print(f"lm loss: {lm_loss}, margin loss: {margin_loss}")
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