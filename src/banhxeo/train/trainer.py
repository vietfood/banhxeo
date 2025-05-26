import json
import os
import random

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from torch.utils.data import DataLoader

from banhxeo import CPU_DEVICE, GPU_DEVICE
from banhxeo.data.torch import TorchTextDataset
from banhxeo.model.neural import NeuralLanguageModel
from banhxeo.train.callbacks import (
    ProgressCallback,
    TrainerCallback,
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
        "Trainer",  # The trainer instance
        NeuralLanguageModel,  # The model
        Dict[str, torch.Tensor],  # The raw batch from DataLoader (on CPU initially)
    ],
    Dict[
        str, Any
    ],  # Should return {"loss": itemized_loss (float/int), "metric1": val1, ...}
]

# EvalStepCallable can remain simpler, as it doesn't involve optimizers/schedulers
EvalStepCallable = Callable[
    [
        "Trainer",  # The trainer instance
        NeuralLanguageModel,  # The model
        Dict[str, torch.Tensor],  # The raw batch from DataLoader (on CPU initially)
    ],
    Dict[str, Any],  # Returns {"eval_loss": itemized_loss, "eval_metric1": val1, ...}
]


class Trainer:
    """Handles the training and evaluation loop for neural language models.

    The `Trainer` orchestrates the training process, including data loading,
    model optimization, logging, checkpointing, and evaluation, according
    to a provided `TrainerConfig`. It supports custom training and evaluation
    logic via `train_step_fn` and `eval_step_fn` callables.

    Attributes:
        model (NeuralLanguageModel): The model to be trained.
        config (TrainerConfig): Configuration for the training process.
        train_dataset (TorchTextDataset): The dataset for training.
        eval_dataset (Optional[TorchTextDataset]): The dataset for evaluation.
        optimizer (Optional[torch.optim.Optimizer]): The optimizer instance.
        loss_fn (Optional[torch.nn.modules.loss._Loss]): The loss function instance.
            Renamed from `loss` to avoid conflict with loss values.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): The LR scheduler.
        train_step_fn (TrainStepCallable): User-defined function for a single training step.
        eval_step_fn (Optional[EvalStepCallable]): User-defined function for a single eval step.
        callbacks (List[TrainerCallback]): List of callbacks to customize training.
        collate_fn (Optional[Callable]): Custom collate function for DataLoaders.
        device (torch.device): The device (CPU or GPU) where training will occur.
        global_step (int): Total number of training steps (optimizer updates or micro-batches) performed.
        current_epoch (int): The current training epoch (1-indexed).
        total_train_steps (int): The total number of training steps planned across all epochs.
        best_metric (Optional[float]): Stores the best evaluation metric value achieved,
            used for saving the best model (if logic is implemented in a callback).
    """

    def __init__(
        self,
        model: NeuralLanguageModel,
        config: TrainerConfig,
        train_dataset: TorchTextDataset,
        eval_dataset: TorchTextDataset,
        train_step_fn: TrainStepCallable,
        eval_step_fn: Optional[TrainStepCallable] = None,
        device: Optional[Union[str, torch.device]] = None,  # Allow str like "cuda"
        callbacks: Optional[List[TrainerCallback]] = None,
        collate_fn: Optional[Callable[[List[Dict[str, torch.Tensor]]], Any]] = None,
        **kwargs,
    ):
        """Initializes the Trainer.

        Args:
            model: The `NeuralLanguageModel` to train.
            config: The `TrainerConfig` specifying training parameters.
            train_dataset: The training `TorchTextDataset`.
            eval_dataset: Optional evaluation `TorchTextDataset`.
            train_step_fn: A callable that executes a single training step.
                It's responsible for the forward pass, loss calculation, backward pass,
                and optimizer step (including gradient accumulation if desired).
                See `TrainStepCallable` for signature.
            eval_step_fn: An optional callable that executes a single evaluation step.
                See `EvalStepCallable` for signature.
            device: The device to train on ('cuda', 'mps', 'cpu', or torch.device object).
                If None, attempts to use GPU, otherwise CPU.
            callbacks: An optional list of `TrainerCallback` instances.
                `ProgressCallback` is added by default.
            collate_fn: An optional custom collate function for the DataLoaders.
        """
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.optimizer = None
        self.loss_fn = None
        self.scheduler = None

        self.train_step_fn = train_step_fn
        self.eval_step_fn = eval_step_fn

        # Initialize callbacks: add ProgressCallback if not already present by user
        self.callbacks = callbacks if callbacks is not None else []
        if not any(isinstance(cb, ProgressCallback) for cb in self.callbacks):
            self.callbacks.insert(0, ProgressCallback())  # Add to the beginning

        self.collate_fn = collate_fn

        self.global_step: int = 0
        self.current_epoch: int = 0  # Will be 1-indexed during training
        self.total_train_steps: int = 0
        self.best_metric: Optional[float] = (
            None  # Example: for best F1, lower is better for loss
        )

        # Determine and set device
        if isinstance(device, str):
            self.device = torch.device(device)
        elif isinstance(device, torch.device):
            self.device = device
        elif GPU_DEVICE is not None:
            self.device = GPU_DEVICE
        else:
            self.device = CPU_DEVICE
        DEFAULT_LOGGER.info(f"Trainer will use device: {self.device}")

        # Seed everything
        self._set_seed(self.config.seed)

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        self._invoke_callbacks("on_init_end")

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)

    def _create_optim_and_loss(self) -> None:
        """Creates the optimizer, LR scheduler, and loss function from config."""
        if self.optimizer is None:
            self.optimizer = self.config.optim.get_optimizer(self.model.parameters())
            DEFAULT_LOGGER.info(
                f"Optimizer created: {self.optimizer.__class__.__name__}"
            )

        if (
            self.scheduler is None
            and self.config.optim.scheduler_name
            and self.optimizer
        ):
            self.scheduler = self.config.optim.get_scheduler(self.optimizer)
            if self.scheduler:
                DEFAULT_LOGGER.info(
                    f"LR Scheduler created: {self.scheduler.__class__.__name__}"
                )

        if self.loss_fn is None:
            self.loss_fn = self.config.loss.get_loss_function()
            DEFAULT_LOGGER.info(
                f"Loss function created: {self.loss_fn.__class__.__name__}"
            )

    def _invoke_callbacks(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        """Invokes a specified method on all registered callbacks."""
        for cb in self.callbacks:
            try:
                getattr(cb, method_name)(self, *args, **kwargs)
            except Exception as e:
                DEFAULT_LOGGER.error(f"Error in callback {cb.name}.{method_name}: {e}")

    def get_callbacks_output(self) -> Dict[str, Any]:
        """Collects and returns outputs from all callbacks that provide them.

        Returns:
            A dictionary where keys are callback names and values are their outputs
            (obtained via `callback.get_output()`).
        """
        result = {}
        for callback in self.callbacks:
            try:
                output = callback.get_output()
                if output is not None:
                    result[callback.name] = output
            except Exception as e:
                DEFAULT_LOGGER.error(
                    f"Error getting output from callback {callback.name}: {e}"
                )
        return result

    def get_train_dataloader(self) -> DataLoader:
        """Creates and returns the DataLoader for the training set."""
        # Ensure train_dataset is a PyTorch Dataset
        if not isinstance(self.train_dataset, torch.utils.data.Dataset):
            raise TypeError(
                f"train_dataset must be a PyTorch Dataset, got {type(self.train_dataset)}"
            )

        return self.train_dataset.to_loader(
            batch_size=self.config.per_device_train_batch_size,
            shuffle=self.config.training_shuffle,
            num_workers=min(os.cpu_count() or 1, 4) if os.cpu_count() else 0,
            collate_fn=self.collate_fn,
            pin_memory=self.device.type == "cuda",  # Pin memory if using CUDA
        )

    def get_eval_dataloader(self) -> Optional[DataLoader]:
        """Creates and returns the DataLoader for the evaluation set, if available."""
        if self.eval_dataset is None:
            return None

        if not isinstance(self.eval_dataset, torch.utils.data.Dataset):
            raise TypeError(
                f"eval_dataset must be a PyTorch Dataset, got {type(self.eval_dataset)}"
            )

        return self.eval_dataset.to_loader(
            batch_size=self.config.per_device_eval_batch_size,
            shuffle=False,  # Don't shuffle for evaluate dataset
            num_workers=min(os.cpu_count() or 1, 4) if os.cpu_count() else 0,
            collate_fn=self.collate_fn,
            pin_memory=self.device.type == "cuda",
        )

    def train(self):
        """Runs the main training loop.

        The loop iterates over epochs and batches, calling `train_step_fn` for
        each batch. It handles optimizer creation, logging, callbacks, and
        optional evaluation during training.

        Returns:
            A dictionary containing training results, potentially including
            outputs from callbacks.
        """
        train_dataloader = self.get_train_dataloader()

        self.total_train_steps = (
            len(train_dataloader)
            // self.config.gradient_accumulation_steps
            * self.config.num_train_epochs
        )

        if self.total_train_steps == 0:
            DEFAULT_LOGGER.warning(
                "Total training steps is 0. Training will not proceed. "
                "Check dataset size and batch size."
            )
            return {"status": "No training steps."}

        self._create_optim_and_loss()  # Ensure optimizer and loss are ready

        self._invoke_callbacks("on_train_begin")

        overall_training_loss = 0.0
        num_steps_for_loss_agg = 0

        for epoch in range(1, self.config.num_train_epochs + 1):
            self.current_epoch = epoch
            self._invoke_callbacks("on_epoch_begin", epoch=epoch)

            self.model.train()  # Set model to training mode
            epoch_loss_agg = 0.0
            num_batches_in_epoch_processed = 0

            for batch_idx, batch in enumerate(train_dataloader):
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

                epoch_loss_agg += current_loss_item
                overall_training_loss += current_loss_item
                num_steps_for_loss_agg += 1
                num_batches_in_epoch_processed += 1

                # Logging based on TrainerConfig.logging_steps (uses global_step)
                if (self.global_step + 1) % self.config.logging_steps == 0:
                    avg_log_loss = (
                        overall_training_loss / num_steps_for_loss_agg
                        if num_steps_for_loss_agg > 0
                        else 0.0
                    )
                    DEFAULT_LOGGER.info(
                        f"Epoch {epoch}/{self.config.num_train_epochs}, "
                        f"Step {self.global_step + 1}/{self.total_train_steps}: "
                        f"Loss: {current_loss_item:.4f} (Avg over last {self.config.logging_steps} steps: {avg_log_loss:.4f})"
                    )
                    # Reset for next logging period
                    overall_training_loss = 0.0
                    num_steps_for_loss_agg = 0

                self._invoke_callbacks(
                    "on_step_end",
                    global_step=self.global_step,
                    batch_idx=batch_idx,
                    logs=step_outputs,
                )

                # Evaluation during training
                if (
                    self.config.evaluate_during_training
                    and self.eval_dataset is not None
                    and self.config.evaluation_steps is not None
                    and (self.global_step + 1) % self.config.evaluation_steps == 0
                    and (self.global_step + 1)
                    < self.total_train_steps  # Avoid eval on last step if also end of epoch
                ):
                    DEFAULT_LOGGER.info(
                        f"Running evaluation at step {self.global_step + 1}..."
                    )
                    self.evaluate()
                    self.model.train()  # Ensure model is back in train mode

                self.global_step += 1

            # Calculate average epoch loss
            avg_epoch_loss = (
                epoch_loss_agg / num_batches_in_epoch_processed
                if num_batches_in_epoch_processed > 0
                else 0.0
            )
            DEFAULT_LOGGER.info(
                f"End of Epoch {epoch}: Average Training Loss: {avg_epoch_loss:.4f}"
            )
            self._invoke_callbacks(
                "on_epoch_end", epoch=epoch, logs={"avg_epoch_loss": avg_epoch_loss}
            )

        final_logs = {
            "final_avg_epoch_loss": avg_epoch_loss  # type: ignore
            if "avg_epoch_loss" in locals()
            else 0.0
        }
        self._invoke_callbacks("on_train_end", logs=final_logs)
        DEFAULT_LOGGER.info("Training finished.")

        results = {
            "training_summary": final_logs,
            "callback_outputs": self.get_callbacks_output(),
        }
        return results

    @torch.inference_mode()
    def evaluate(self):
        """Runs the evaluation loop on the `eval_dataset`.

        If `eval_dataset` or `eval_step_fn` is not provided, evaluation is skipped.
        Sets the model to evaluation mode (`model.eval()`) and disables gradient calculations.

        Returns:
            A dictionary containing evaluation metrics, or None if evaluation
            is skipped. Keys typically include "avg_eval_loss" and other
            metrics returned by `eval_step_fn`.
        """
        eval_dataloader = self.get_eval_dataloader()
        if eval_dataloader is None or self.eval_step_fn is None:
            DEFAULT_LOGGER.info(
                "No evaluation dataset or eval_step_fn provided. Skipping evaluation."
            )
            return None

        self.model.eval()  # Set model to evaluation mode

        all_eval_metrics_sum = {}
        num_eval_steps = 0

        for batch_idx, batch in enumerate(eval_dataloader):
            # User's eval_step_fn is responsible for:
            # - Moving batch to device
            # - Forward pass
            # - Loss calculation (optional, for eval loss)
            # - Metric calculation
            step_metrics = self.eval_step_fn(self, self.model, batch)  # type: ignore

            for k, v in step_metrics.items():
                if isinstance(v, (float, int, torch.Tensor)):
                    all_eval_metrics_sum[k] += (
                        v.item() if isinstance(v, torch.Tensor) else v
                    )
                # Non-numeric metrics could be collected differently if needed
            num_eval_steps += 1

        # Average metrics
        final_metrics: Dict[str, float] = {}
        if num_eval_steps > 0:
            for k, v_sum in all_eval_metrics_sum.items():
                final_metrics[k] = v_sum / num_eval_steps

        # Standardize "eval_loss" key if present
        if "loss" in final_metrics:  # If eval_step_fn returns "loss"
            final_metrics["avg_eval_loss"] = final_metrics.pop("loss")
        elif "eval_loss" in final_metrics:  # If eval_step_fn returns "eval_loss"
            final_metrics["avg_eval_loss"] = final_metrics.pop("eval_loss")

        DEFAULT_LOGGER.info(f"Evaluation Results: {final_metrics}")
        self._invoke_callbacks("on_evaluate", metrics=final_metrics)

        return final_metrics

    def save_model(self, output_dir: Union[str, Path]) -> None:
        """Saves the model, optimizer, scheduler, and trainer configuration.

        Args:
            output_dir: The directory where components will be saved.
                It will be created if it doesn't exist.
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # 1. Save the model itself (weights and model-specific config)
        self.model.save_model(output_dir_path)

        # 2. Save optimizer state
        if self.optimizer:
            optimizer_path = output_dir_path / "optimizer.pt"
            torch.save(self.optimizer.state_dict(), optimizer_path)
            DEFAULT_LOGGER.info(f"Optimizer state saved to {optimizer_path}")

        # 3. Save scheduler state
        if self.scheduler:
            scheduler_path = output_dir_path / "scheduler.pt"
            torch.save(self.scheduler.state_dict(), scheduler_path)
            DEFAULT_LOGGER.info(f"Scheduler state saved to {scheduler_path}")

        # 4. Save TrainerConfig (Pydantic model)
        trainer_config_path = output_dir_path / "trainer_config.json"
        try:
            with open(trainer_config_path, "w", encoding="utf-8") as f:
                json.dump(self.config.model_dump(mode="json"), f, indent=2)
            DEFAULT_LOGGER.info(f"Trainer configuration saved to {trainer_config_path}")
        except Exception as e:
            DEFAULT_LOGGER.error(f"Failed to save trainer configuration: {e}")

        # Invoke on_save for callbacks so they can save their state too
        self._invoke_callbacks("on_save", checkpoint_dir=output_dir_path)
