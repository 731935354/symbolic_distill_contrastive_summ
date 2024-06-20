# Introduction
This repository contains code for the paper **[Factual Dialogue Summarization via Learning from Large Language Models]()**

# Requirements
* transformers 4.30.2

# Training
We modified the original code from transformers, to generate summaries on the original development/validation set and test set every 500 training steps. This makes it easier for us to conduct hyper-parameter search and checkpoint selection based on a set of quality metrics after the fine-tuning process. However, this slows down the finetuning process. If you don't want this behavior, set **eval_steps** to a large value.

## MLE & SeqDistill
The following command can be used for finetuning on the original or augmented summaries generated from the teacher model.

You need to provide the following parameters:
* MODELNAME: the name (in [huggingface models](https://huggingface.co/models)) or path to the local initial checkpoint for finetuning, such as "facebook/bart-large".
* TRAIN_FILE: a json line file. Each line corresponds to a dictionary with two mandatory fields: **text**, **summary**. Field **text** is a string representing the input dialogue.
* VAL_FILE: same format as TRAIN_FILE
* OUTPUT_DIR: the path to save model generations and model checkpoints


```
export PYTHONPATH="${PYTHONPATH}:/path/to/symbolic_distill_contrastive_summ"

python finetuning/run_summarization.py \
    --model_name_or_path ${MODELNAME} \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ${TRAIN_FILE} \
    --validation_file ${VAL_FILE} \
    --test_file ${TEST_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --predict_with_generate \
    --max_source_length 512 \
    --max_target_length 128 \
    --learning_rate 0.00003 \
    --max_steps 15000 \
    --report_to tensorboard wandb \
    --save_strategy steps \
    --save_total_limit 1 \
    --save_steps 500 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --fp16 False \
    --load_best_model_at_end True \
    --gradient_accumulation_steps 16 \
    --text_column text \
    --summary_column summary
```

## MarginContrast
Code to finetune with MarginContrast. You need to provide the following parameters:
* DATADIR: the folder containing C_TRAIN_FILE and C_DEV_FILE
* C_TRAIN_FILE: the json line file with dialogues and positive + negative summaries. [Here](data/samsum_contrast_200.jsonl) is an example file.
* C_DEV_FILE: same format as C_TRAIN_FILE.
* VAL_FILE: A json file with dialogues and reference summaries. We generate summaries on this set for hyper-parameter and checkpoint selection. 
* TEST_FILE: same format as VAL_FILE. We generate summaries on this set for evaluation. [Here](data/samsum.test.json) is an example file.
* MODELNAME: the name (in [huggingface models](https://huggingface.co/models)) or path to the local initial checkpoint for finetuning, such as "facebook/bart-large".
* MODELTYPE: "bart", "pegasus" or "t5".
* OUTPUT_DIR: the path to save model generations and model checkpoints
* THETA: the margin threshold, such as 15.
* ALPHA: the coefficient of contrastive loss, such as 1.0.

```
export PYTHONPATH="${PYTHONPATH}:/path/to/symbolic_distill_contrastive_summ"

python margin_contrast/margin_contrast_train.py \
--data_dir ${DATADIR} \
--contrastive_train_file ${C_TRAIN_FILE} \
--contrastive_dev_file ${C_DEV_FILE} \
--ori_dev_test_batch_size 4 \
--validation_file ${VAL_FILE} \
--test_file ${TEST_FILE} \
--model_name_or_path ${MODELNAME} \
--model_type ${MODELTYPE} \
--output_dir {OUTPUT_DIR} \
--theta ${THETA} \
--alpha ${ALPHA} \
--max_input_length 512 \
--max_target_length 128 \
--evaluation_strategy steps \
--eval_steps 500 \
--save_strategy steps \
--save_steps 500 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--learning_rate 3e-5 \
--max_steps 15000 \
--weight_decay 0.01 \
--save_total_limit 1 \
--num_train_epochs 10 \
--gradient_accumulation_steps 32 \
--remove_unused_columns False \
--report_to tensorboard wandb \
--do_train \
--do_predict \
--predict_with_generate
```

## PairContrast
Code to finetune a student model with PairContrast. You need to provide the following parameters:
* DATADIR: the folder containing C_TRAIN_FILE and C_DEV_FILE
* C_TRAIN_FILE: the json line file with dialogues and positive + negative summaries. [Here](data/samsum_contrast_200.jsonl) is an example file.
* C_DEV_FILE: same format as C_TRAIN_FILE.
* VAL_FILE: A json file with dialogues and reference summaries. We generate summaries on this set for hyper-parameter and checkpoint selection. 
* TEST_FILE: same format as VAL_FILE. We generate summaries on this set for evaluation. [Here](data/samsum.test.json) is an example file.
* MODELNAME: the name (in [huggingface models](https://huggingface.co/models)) or path to the local initial checkpoint for finetuning, such as "facebook/bart-large".
* MODELTYPE: "bart", "pegasus" or "t5".
* OUTPUT_DIR: the path to save model generations and model checkpoints
* THETA: the coefficient of contrastive loss, such as 1.0.

```
python pair_contrast/pair_contrast_train.py \
--theta ${THETA} \
--data_dir ${DATA_DIR} \
--contrastive_train_file ${C_TRAIN_FILE} \
--contrastive_dev_file ${C_DEV_FILE}.jsonl \
--ori_dev_test_batch_size 4 \
--validation_file ${VAL_FILE} \
--test_file ${TEST_FILE} \
--model_name_or_path ${MODELNAME} \
--model_type ${MODELTYPE} \
--max_input_length 512 \
--max_target_length 128 \
--output_dir ${OUTPUT_DIR} \
--evaluation_strategy steps \
--eval_steps 500 \
--save_strategy steps \
--save_steps 500 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--learning_rate 3e-5 \
--weight_decay 0.01 \
--save_total_limit 1 \
--num_train_epochs 10 \
--max_steps 15000 \
--gradient_accumulation_steps 32 \
--report_to wandb \
--do_train \
--do_predict \
--do_eval \
--predict_with_generate \
--label_smoothing_factor 0
```

# Evaluation
We include code to use AlignScore (for factual consistency) and UniEval (for coherence, fluency and relevance). We recommend creating separate conda environments for them to avoid potential conflict with the model training environment. See [`evaluation/align_score.py`](evaluation/align_score.py) and [`evaluation/unieval.py`](evaluation/unieval.py) for more details. 

We follow the official instruction of [G-Eval](https://github.com/nlpyang/geval) to calculate factual consistency using OpenAI API.

