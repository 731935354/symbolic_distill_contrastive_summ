import os
import datetime

from transformers import TrainerCallback, logging, TrainingArguments, TrainerState, TrainerControl
from transformers.integrations import WandbCallback


class ContrastWandbCallback(WandbCallback):
    likelihood_log_steps = 5
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Log contrastive_loss, lm_loss and total_loss in wandb"""
        if state.global_step % self.likelihood_log_steps == 0:
            self._wandb.log({
                f"contrast_loss (mean)": state.contrast_loss,
                f"lm_loss (mean)": state.lm_loss.mean(),
                f"total_loss": state.total_loss,
                f"train/global_step": state.global_step,  # x-axis of chart
            })