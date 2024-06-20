import os
import re
import nltk

import numpy as np

from datasets import load_metric, load_dataset
from huggingface_hub.utils import logging
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, BartTokenizerFast, PegasusTokenizerFast, \
    T5TokenizerFast, DataCollatorForSeq2Seq

from margin_contrast.callbacks import CustomWandbCallback
from margin_contrast.margin_contrast_data import ContrastiveMultiDataset, DataCollatorForContrastive
from margin_contrast.margin_contrast_models import BartForContrastive, T5ForContrastive, PegasusForContrastive
from margin_contrast.margin_contrast_trainer import MarginContrastTrainer

logger = logging.get_logger(__name__)


def main():
    parser = HfArgumentParser((Seq2SeqTrainingArguments,))
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--model_type', type=str, choices=["bart", "pegasus", "t5"])
    parser.add_argument('--theta', type=float, default=0.5, help="the margin in margin loss")
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help="the weight of margin loss in the final loss "
             "(i.e., final loss = standard_crossentropy_loss + alpha * margin_loss)"
    )
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_target_length', type=int, default=128)
    parser.add_argument('--ignore_pad_token_for_loss', type=bool, default=True)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--pad_to_max_length', type=bool, default=False)
    parser.add_argument('--contrast_policy', type=str, default="setwise",
                        choices=["setwise", "pairwise"])
    parser.add_argument('--contrastive_train_file', type=str,
                        help="file for contrastive training set")
    parser.add_argument('--contrastive_dev_file', type=str,
                        help="file for contrastive development set")
    parser.add_argument('--validation_file', type=str, help="file for original development set (instead of contrastive)")
    parser.add_argument('--test_file', type=str, help="file for original test set (instead of contrastive)")
    parser.add_argument('--ori_dev_test_batch_size', type=int, default=8)
    parser.add_argument('--resume_from_ckpt', type=bool, default=False)
    training_args, other_args = parser.parse_args_into_dataclasses()

    # very important, otherwise only those columns included in model.forward will be kept,
    # and we can not do further processing in this case
    training_args.remove_unused_columns = False

    training_args.alpha = other_args.alpha
    training_args.theta = other_args.theta

    if other_args.model_type == "bart":
        model = BartForContrastive.from_pretrained(other_args.model_name_or_path)
        tokenizer = BartTokenizerFast.from_pretrained(other_args.model_name_or_path)
    elif other_args.model_type == "pegasus":
        model = PegasusForContrastive.from_pretrained(other_args.model_name_or_path)
        tokenizer = PegasusTokenizerFast.from_pretrained(other_args.model_name_or_path)
    elif other_args.model_type == "t5":
        model = T5ForContrastive.from_pretrained(other_args.model_name_or_path)
        tokenizer = T5TokenizerFast.from_pretrained(other_args.model_name_or_path)
    else:
        raise ValueError(f"Model type [{other_args.model_type}] not supported.")

    data_dir = other_args.data_dir

    source_prefix = ''
    if other_args.model_type == "t5":
        source_prefix = "summarize: "

    train_dataset = ContrastiveMultiDataset(
        os.path.join(data_dir, other_args.contrastive_train_file),
        tokenizer,
        other_args.max_input_length,
        other_args.max_target_length,
        source_prefix=source_prefix,
        policy=other_args.contrast_policy,
    )

    dev_dataset = ContrastiveMultiDataset(
        os.path.join(data_dir, other_args.contrastive_dev_file),
        tokenizer,
        other_args.max_input_length,
        other_args.max_target_length,
        source_prefix=source_prefix,
        policy=other_args.contrast_policy
    )

    # load test dataset
    text_column = "text"
    summary_column = "summary"
    padding = "max_length" if other_args.pad_to_max_length else False

    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        inputs = [source_prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=other_args.max_input_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=other_args.max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        # if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        #     labels["input_ids"] = [
        #         [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        #     ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    data_files = {}
    if other_args.test_file is not None:
        data_files["test"] = other_args.test_file
        extension = other_args.test_file.split(".")[-1]
    if other_args.validation_file is not None:
        data_files["val"] = other_args.validation_file
        extension = other_args.validation_file.split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=None,
        use_auth_token=None,
    )
    ori_validation_dataset = raw_datasets["val"]
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        ori_validation_dataset = ori_validation_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=None,
            load_from_cache_file=False,
            desc="Running tokenizer on validation dataset",
        )

    test_dataset = raw_datasets["test"]
    with training_args.main_process_first(desc="prediction dataset map pre-processing"):
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=None,
            load_from_cache_file=False,
            desc="Running tokenizer on test dataset",
        )

    data_collator = DataCollatorForContrastive(tokenizer, model=model)

    label_pad_token_id = -100 if other_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator_for_val_test = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    custom_wandb_callback = CustomWandbCallback()

    metric = load_metric("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if other_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    trainer = MarginContrastTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,  # monitor ROUGE
        ori_validation_dataset=ori_validation_dataset,  # we generate summaries on the original validation set
        test_dataset=test_dataset,
        data_collator=data_collator,
        data_collator_for_val_test=data_collator_for_val_test,
        tokenizer=tokenizer,
        callbacks=[custom_wandb_callback],
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        ori_dev_test_batch_size=other_args.ori_dev_test_batch_size
    )

    trainer.train(resume_from_checkpoint=other_args.resume_from_ckpt)


if __name__ == '__main__':
    main()