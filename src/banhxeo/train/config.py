from typing import Any, Dict, Optional

import torch
import torch.optim as optim
from pydantic import BaseModel, Field


class LossConfig(BaseModel):
    name: str = "CrossEntropyLoss"  # e.g., "CrossEntropyLoss", "MSELoss"
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    def get_loss_function(self):
        if self.name not in torch.nn.modules.loss.__all__:
            raise ValueError(f"Optimizer '{self.name}' not found in torch.optim")
        imported = getattr(
            __import__("torch.nn.modules.loss", fromlist=[self.name]), self.name
        )
        return imported(**self.kwargs)


class OptimizerConfig(BaseModel):
    # Optimizer and Scheduler - can be strings or actual classes/functions
    name: str = "AdamW"  # e.g., "AdamW", "SGD"

    # LR Scheduler
    scheduler_name: Optional[str] = None  # e.g. "LambdaLR", ...
    warmup_steps: int = 0

    # Hyperparemeters
    optimizer_kwargs: Dict[str, Any] = Field(default_factory=dict)
    lr_scheduler_kwargs: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def get_optimizer(self, model_parameters):
        # https://stackoverflow.com/questions/6677424/how-do-i-import-variable-packages-in-python-like-using-variable-variables-i
        if self.name not in optim.__all__:
            raise ValueError(f"Optimizer '{self.name}' not found in torch.optim")
        imported = getattr(__import__("torch.optim", fromlist=[self.name]), self.name)
        return imported(model_parameters, **self.optimizer_kwargs)

    def get_scheduler(self, optimizer):
        if self.scheduler_name not in optim.lr_scheduler.__all__:
            raise ValueError(
                f"LR Scheduler '{self.scheduler_name}' not found in torch.optim.lr_scheduler"
            )
        imported = getattr(
            __import__("torch.optim.lr_scheduler", fromlist=[self.scheduler_name]),
            self.scheduler_name,
        )
        return imported(optimizer, **self.lr_scheduler_kwargs)


class TrainerConfig(BaseModel):
    output_dir: str = "./training_output"  # Directory to save checkpoints, logs, etc.
    num_train_epochs: int = 3

    # Batch sizes
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1  # For larger effective batch sizes
    training_shuffle: bool = True

    # Logging and Saving
    logging_steps: int = 100  # Log loss every N steps
    save_steps: Optional[int] = 500  # Save checkpoint every N steps
    save_total_limit: Optional[int] = None  # Limit number of saved checkpoints
    evaluate_during_training: bool = False  # Whether to run eval loop during training
    evaluation_steps: Optional[int] = (
        None  # Evaluate every N steps if evaluate_during_training is True
    )

    # Seed and device
    seed: int = 42

    # Optimizer
    optim: OptimizerConfig

    # Loss
    loss: LossConfig
