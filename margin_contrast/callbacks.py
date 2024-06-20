import os
import datetime

from transformers import TrainerCallback, logging, TrainingArguments, TrainerState, TrainerControl
from transformers.integrations import WandbCallback


class CustomWandbCallback(WandbCallback):
    likelihood_log_steps = 2
    gradients_log_steps = 10
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Log log-likelihood of both positive and negative targets in wandb"""
        if state.global_step % self.likelihood_log_steps == 0:
            max_neg_score = state.batch_max_neg_score
            min_pos_score = state.batch_min_pos_score
            required_margin = [(x - y).item() for x, y in zip(min_pos_score, max_neg_score)]
            self._wandb.log({
                f"margin_loss (mean)": state.margin_loss.mean(),
                f"model_loss (mean)": state.model_loss,
                f"train/global_step": state.global_step  # x-axis of chart
            })
            # we record these values for each sample in the batch
            for i in range(len(required_margin)):
                self._wandb.log({  # we only record the value of the first sample in the batch
                    f"max_neg_score_sample{i}": max_neg_score[i],
                    f"min_pos_score_sample{i}": min_pos_score[i],
                    f"required_margin(for a non-negative margin loss)_sample{i}": required_margin[i],
                    f"train/global_step": state.global_step + i  # x-axis of chart
                })


class SaveSummariesCallback(TrainerCallback):
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(1)
        eval_dataloader = kwargs
        model = kwargs['model']
        tokenizer = kwargs['tokenizer']
        eval_dataloader = kwargs['eval_dataloader']
        save_dir = args.logging_dir.split("runs")[0]
        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d_%H_%M")
        summary_file = os.path.join(save_dir, f"{formatted_time}_generated_summaries.txt")