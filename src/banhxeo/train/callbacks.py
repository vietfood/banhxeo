from abc import ABCMeta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from banhxeo.utils.logging import DEFAULT_LOGGER

# Forward reference
if TYPE_CHECKING:
    from banhxeo.model.neural import (
        NeuralLanguageModel,
    )
    from banhxeo.train.trainer import Trainer


class TrainerCallback(metaclass=ABCMeta):
    """Base class for Trainer callbacks."""

    def get_output(self) -> Any:
        return None

    def on_init_end(self, trainer: "Trainer"): ...

    def on_train_begin(self, trainer: "Trainer"): ...

    def on_train_end(self, trainer: "Trainer", logs: Optional[Dict[str, float]]): ...

    def on_epoch_begin(self, trainer: "Trainer", epoch: int): ...

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Optional[Dict[str, float]]
    ): ...

    def on_step_begin(self, trainer: "Trainer", global_step: int, batch_idx: int): ...

    def on_step_end(
        self,
        trainer: "Trainer",
        global_step: int,
        batch_idx: int,
        logs: Optional[Dict[str, float]],
    ): ...

    def on_evaluate(self, trainer: "Trainer", metrics: Optional[Dict[str, float]]): ...

    def on_save(self, trainer: "Trainer", checkpoint_dir: Path): ...


class ProgressCallback(TrainerCallback):
    name = "progress"

    def __init__(self):
        self.train_pbar = None
        self.eval_pbar = None

    def on_train_begin(self, trainer: "Trainer"):
        from banhxeo.utils import progress_bar

        self.train_pbar = progress_bar(
            total=trainer.total_train_steps,
            desc="Training",
            unit="step",
            unit_scale=True,
        )

    def on_train_end(self, trainer: "Trainer", logs: Optional[Dict[str, float]]):
        if self.train_pbar:
            self.train_pbar.close()
            self.train_pbar

    def on_step_end(
        self,
        trainer: "Trainer",
        global_step: int,
        batch_idx: int,
        logs: Optional[Dict[str, float]],
    ):
        if self.train_pbar:
            self.train_pbar.update(1)
            if logs:
                self.train_pbar.set_postfix(logs, refresh=True)


class AccuracyCallback(TrainerCallback):
    name = "accuracy"

    def __init__(self, log_step: int = 100):
        super().__init__()
        self.log_step = log_step
        self.correct = 0
        self.total = 0
        self.accs = dict()

    def on_step_end(
        self,
        trainer,
        global_step,
        batch_idx,
        logs,  # Shouldn't be None
    ):
        if logs is None:
            raise ValueError("Log values shouldn't be None for AccuracyCallback")

        if (global_step + 1) % self.log_step == 0:
            acc = (self.correct * 100) / self.total
            # reset
            self.correct = 0
            self.total = 0
            self.accs[global_step + 1] = acc
            DEFAULT_LOGGER.info(f"Accuracy after {global_step + 1} steps: {acc:.2f}%")
        else:
            self.correct += logs["correct"]
            self.total += logs["total"]

    def on_evaluate(self, trainer, metrics):
        acc = (metrics["correct"] * 100) / metrics["total"]  # type: ignore
        DEFAULT_LOGGER.info(f"Testing accuracy: {acc:.2f}%")

    def get_output(self):
        return self.accs


class CheckpointCallback(TrainerCallback):
    name = "checkpoint"

    def on_step_end(
        self,
        trainer: "Trainer",
        global_step: int,
        batch_idx: int,
        logs: Optional[Dict[str, float]],
    ):
        if (
            trainer.config.save_steps
            and global_step % trainer.config.save_steps == 0
            and global_step > 0
        ):
            self._save_checkpoint(trainer, f"checkpoint-{global_step}")

    def on_epoch_end(
        self, trainer: "Trainer", epoch: int, logs: Optional[Dict[str, float]]
    ):
        self._save_checkpoint(trainer, f"checkpoint-epoch-{epoch}")

    def _save_checkpoint(self, trainer: "Trainer", checkpoint_name: str):
        checkpoint_dir = Path(trainer.config.output_dir) / checkpoint_name
        trainer.save_model(checkpoint_dir)
        DEFAULT_LOGGER.info(f"Checkpoint saved to {checkpoint_dir}")

        # Call the on_save hook for other callbacks
        for cb in trainer.callbacks:
            cb.on_save(trainer, checkpoint_dir)
