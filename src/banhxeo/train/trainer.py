import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from banhxeo import CPU_DEVICE, GPU_DEVICE
from banhxeo.data.torch import TorchTextDataset
from banhxeo.model.neural import NeuralLanguageModel
from banhxeo.train.callbacks import (
    ProgressCallback,
)
from banhxeo.train.config import TrainerConfig
from banhxeo.utils.logging import DEFAULT_LOGGER

# User train step must do:
# 1. Move batch data to the correct device.
# 2. Perform the forward pass.
# 3. Calculate the loss.
# 4. Call loss.backward().
# 5. If using gradient accumulation, divide loss by accumulation_steps before backward.
# 6. If it's time to update weights (i.e., after N accumulation steps):
#    a. Optional: Gradient clipping.
#    b. optimizer.step().
#    c. scheduler.step() (if scheduler exists and stepping per optimizer step).
#    d. optimizer.zero_grad().
# 7. Calculate any desired metrics.
# 8. Return a dictionary containing at least {"loss": loss_item} and other metrics.
TrainStepCallable = Callable[
    [
        "Trainer",  # The trainer instance itself (for access to global_step, etc.)
        NeuralLanguageModel,  # The model
        Dict[str, torch.Tensor],  # The raw batch from DataLoader
    ],
    Dict[str, Any],  # Should return {"loss": itemized_loss, "metric1": val1, ...}
]

# EvalStepCallable can remain simpler, as it doesn't involve optimizers/schedulers
EvalStepCallable = Callable[
    [
        "Trainer",
        NeuralLanguageModel,
        Dict[str, torch.Tensor],  # Raw batch
    ],
    Dict[str, Any],  # Returns {"eval_loss": itemized_loss, "eval_metric1": val1, ...}
]


class Trainer:
    def __init__(
        self,
        model: NeuralLanguageModel,
        config: TrainerConfig,
        train_dataset: TorchTextDataset,
        eval_dataset: TorchTextDataset,
        train_step_fn: TrainStepCallable,
        eval_step_fn: Optional[TrainStepCallable] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.optimizer = None
        self.loss = None
        self.scheduler = None

        self.train_step_fn = train_step_fn
        self.eval_step_fn = eval_step_fn

        self.callbacks = [ProgressCallback()] + kwargs.get("callbacks", [])
        self._invoke_callbacks("on_init_end")

        self.collate_fn = kwargs.get("collate_fn")  # For dataloader

        self.global_step = 0
        self.current_epoch = 0
        self.total_train_steps = 0  # Will be calculated in train()
        self.best_metric = None  # For saving best model based on a metric

        self.device = device
        if self.device is None:
            if GPU_DEVICE is None:
                self.device = CPU_DEVICE
            else:
                self.device = GPU_DEVICE

        # Seed everything
        self._set_seed(self.config.seed)

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)

    def _create_optim_and_loss(self):
        self.optimizer = self.config.optim.get_optimizer(self.model.parameters())
        if self.config.optim.scheduler_name is not None:
            self.scheduler = self.config.optim.get_scheduler(self.optimizer)
        self.loss = self.config.loss.get_loss_function()

    def _invoke_callbacks(self, method_name: str, *args, **kwargs):
        for cb in self.callbacks:
            getattr(cb, method_name)(self, *args, **kwargs)

    def get_callbacks_output(self):
        result = dict()
        for callback in self.callbacks:
            output = callback.get_output()
            if output is not None:
                result[callback.name] = output
        return result

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return self.train_dataset.to_loader(
            batch_size=self.config.per_device_train_batch_size,
            shuffle=self.config.training_shuffle,
            num_workers=os.cpu_count() // 2 if os.cpu_count() else 0,  # type: ignore
        )

    def get_eval_dataloader(self) -> Optional[DataLoader]:
        if self.eval_dataset is None:
            return None

        return self.eval_dataset.to_loader(
            batch_size=self.config.per_device_eval_batch_size,
            shuffle=False,  # We don't need shuffle for eval dataset
            num_workers=os.cpu_count() // 2 if os.cpu_count() else 0,  # type: ignore
        )

    def train(self):
        train_dataloader = self.get_train_dataloader()

        self.total_train_steps = (
            len(train_dataloader)
            // self.config.gradient_accumulation_steps
            * self.config.num_train_epochs
        )

        self._create_optim_and_loss()

        self._invoke_callbacks("on_train_begin")

        train_loss_agg = 0.0

        for epoch in range(1, self.config.num_train_epochs + 1):
            self.current_epoch = epoch
            self._invoke_callbacks("on_epoch_begin", epoch=epoch)
            self.model.train()

            for batch_idx, batch in enumerate(train_dataloader):  # batch is on CPU here
                self._invoke_callbacks(
                    "on_step_begin", global_step=self.global_step, batch_idx=batch_idx
                )

                # User's function is responsible for:
                # - Moving batch to device
                # - Forward pass
                # - Loss calculation
                # - loss.backward()
                # - Gradient accumulation logic (if any, impacting optimizer.step())
                # - optimizer.step(), scheduler.step(), optimizer.zero_grad()
                # - Metric calculation
                step_outputs = self.train_step_fn(
                    self,
                    self.model,
                    batch,
                )

                # Extract loss for aggregation - user MUST return an itemized loss
                current_loss_item = step_outputs.get("loss")
                if current_loss_item is None or not isinstance(
                    current_loss_item, (float, int)
                ):
                    raise ValueError(
                        "train_step_fn must return a dictionary with an itemized 'loss' (float/int)."
                    )

                train_loss_agg += current_loss_item

                # Logging (loss, metrics from step_outputs)
                log_payload = {k: v for k, v in step_outputs.items()}
                self._invoke_callbacks(
                    "on_step_end",
                    global_step=self.global_step,
                    batch_idx=batch_idx,
                    logs=log_payload,  # Callbacks will receive the user's returned dict
                )

                # Evaluation during training (if configured)
                if (
                    self.config.evaluate_during_training
                    and self.config.evaluation_steps
                    and self.global_step > 0
                    and (self.global_step + 1) % self.config.evaluation_steps == 0
                ):  # Check global_step + 1 because global_step isn't incremented yet
                    self.evaluate()
                    # User's train_step_fn should ensure model is back in train mode if eval changes it.
                    # Or, Trainer can enforce model.train() after evaluate() if that's a desired convention.
                    self.model.train()

                self.global_step += 1

            # Calculate average epoch loss
            num_batches_in_epoch = len(train_dataloader)
            avg_epoch_loss = (
                train_loss_agg / num_batches_in_epoch
                if num_batches_in_epoch > 0
                else 0.0
            )

            self._invoke_callbacks(
                "on_epoch_end", epoch=epoch, logs={"avg_epoch_loss": avg_epoch_loss}
            )
            train_loss_agg = 0.0  # Reset for next epoch

        self._invoke_callbacks(
            "on_train_end",
            logs={
                "final_avg_loss": avg_epoch_loss  # type: ignore
                if "avg_epoch_loss" in locals()
                else 0.0
            },
        )
        DEFAULT_LOGGER.info("Training finished.")

    def evaluate(self):
        eval_dataloader = self.get_eval_dataloader()
        if eval_dataloader is None:
            DEFAULT_LOGGER.info("No evaluation dataset provided, skipping evaluation.")
            return None

        all_metrics = {}  # Aggregate metrics
        total_eval_loss = 0
        num_eval_steps = 0

        self.model.eval()
        with torch.inference_mode():
            for batch in eval_dataloader:
                # User's eval_step_fn is responsible for:
                # - Moving batch to device
                # - Forward pass
                # - Loss calculation (optional, for eval loss)
                # - Metric calculation
                step_metrics = self.eval_step_fn(self, self.model, batch)  # type: ignore

                current_eval_loss_item = step_metrics.get("eval_loss")
                if current_eval_loss_item is not None and isinstance(
                    current_eval_loss_item, (float, int)
                ):
                    total_eval_loss += current_eval_loss_item

                num_eval_steps += 1

                for k, v in step_metrics.items():
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].append(v)

        # Average metrics
        final_metrics = {}
        for k, v_list in all_metrics.items():
            try:
                final_metrics[k] = sum(v_list) / len(v_list) if len(v_list) > 0 else 0.0
            except TypeError:  # If v_list contains non-numeric types
                final_metrics[k] = v_list  # Or some other aggregation

        final_metrics["avg_eval_loss"] = (
            total_eval_loss / num_eval_steps if num_eval_steps > 0 else 0.0
        )

        self._invoke_callbacks("on_evaluate", metrics=final_metrics)

    def save_model(self, output_dir: Union[str, Path]):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_model(output_dir)  # Model saves its own config and weights

        # Save optimizer and scheduler state
        if self.optimizer:
            torch.save(self.optimizer.state_dict(), output_dir / "optimizer.pt")
        if self.scheduler:
            torch.save(self.scheduler.state_dict(), output_dir / "scheduler.pt")

        # Save trainer config (optional, as it might be part of experiment tracking)
        with open(output_dir / "trainer_config.json", "w") as f:
            json.dump(self.config.model_dump(), f, indent=2)
