from typing import Any, Dict, Optional

import torch
import torch.optim as optim

from pydantic import BaseModel, Field, field_validator


class LossConfig(BaseModel):
    """Configuration for the loss function used during training.

    Attributes:
        name: The name of the loss function class from `torch.nn.modules.loss`
            (e.g., "CrossEntropyLoss", "MSELoss"). Defaults to "CrossEntropyLoss".
        kwargs: A dictionary of keyword arguments to be passed to the
            loss function's constructor (e.g., `{"weight": torch.tensor([0.1, 0.9])}`).
            Defaults to an empty dictionary.
    """

    name: str = "CrossEntropyLoss"  # e.g., "CrossEntropyLoss", "MSELoss"
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    def get_loss_function(self):
        """Instantiates and returns the configured PyTorch loss function.

        Returns:
            An instance of the specified PyTorch loss function.

        Raises:
            ValueError: If the specified `name` is not a valid loss function
                in `torch.nn.modules.loss`.
            AttributeError: If the specified `name` is not found.
        """
        if self.name not in torch.nn.modules.loss.__all__:
            raise ValueError(
                f"Optimizer '{self.name}' not found in torch.nn.module.loss"
            )
        imported = getattr(
            __import__("torch.nn.modules.loss", fromlist=[self.name]), self.name
        )
        return imported(**self.kwargs)


class OptimizerConfig(BaseModel):
    """Configuration for the optimizer and learning rate scheduler.

    Attributes:
        name: The name of the optimizer class from `torch.optim`
            (e.g., "AdamW", "SGD"). Defaults to "AdamW".
        scheduler_name: Optional name of the learning rate scheduler class
            from `torch.optim.lr_scheduler` (e.g., "LambdaLR", "ReduceLROnPlateau").
            Defaults to None (no scheduler).
        warmup_steps: Number of initial steps during which the learning rate
            is linearly warmed up from 0 to its initial value. This is typically
            implemented by custom logic or a scheduler like `transformers.get_linear_schedule_with_warmup`.
            Note: This field is declarative; actual warmup logic needs to be
            implemented in the training loop or via a specific scheduler. Defaults to 0.
        optimizer_kwargs: Keyword arguments for the optimizer's constructor
            (e.g., `{"lr": 1e-3, "weight_decay": 0.01}`). Defaults to an empty dict.
        lr_scheduler_kwargs: Keyword arguments for the LR scheduler's constructor.
            Defaults to an empty dict.
    """

    name: str = "AdamW"  # e.g., "AdamW", "SGD"

    # LR Scheduler
    scheduler_name: Optional[str] = None  # e.g. "LambdaLR", ...

    # Hyperparemeters
    optimizer_kwargs: Dict[str, Any] = Field(default_factory=dict)
    lr_scheduler_kwargs: Dict[str, Any] = Field(default_factory=dict)

    class Config:  # noqa: D106
        arbitrary_types_allowed = True

    def get_optimizer(self, model_parameters):
        """Instantiates and returns the configured PyTorch optimizer.

        Args:
            model_parameters: An iterable of model parameters to optimize,
                typically `model.parameters()`.

        Returns:
            An instance of the specified PyTorch optimizer.

        Raises:
            ValueError: If the specified `name` is not a valid optimizer
                in `torch.optim`.
        """
        # https://stackoverflow.com/questions/6677424/how-do-i-import-variable-packages-in-python-like-using-variable-variables-i

        if self.name not in optim.__all__:
            raise ValueError(f"Optimizer '{self.name}' not found in torch.optim")
        imported = getattr(__import__("torch.optim", fromlist=[self.name]), self.name)
        return imported(model_parameters, **self.optimizer_kwargs)

    def get_scheduler(self, optimizer):
        """Instantiates and returns the configured PyTorch learning rate scheduler.

        Args:
            optimizer: The PyTorch optimizer instance for which to create the scheduler.

        Returns:
            An instance of the specified LR scheduler, or None if `scheduler_name` is not set.

        Raises:
            ValueError: If `scheduler_name` is set but not found in
                `torch.optim.lr_scheduler`.
        """
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
    """Configuration for the `Trainer`.

    Specifies parameters for the training process, including directories,
    epochs, batch sizes, logging, saving, evaluation, and optimization settings.

    Attributes:
        output_dir: Directory to save checkpoints, logs, and other training artifacts.
            Defaults to "./training_output".
        num_train_epochs: Total number of training epochs to perform. Defaults to 3.
        per_device_train_batch_size: Batch size per GPU/CPU for training. Defaults to 8.
        per_device_eval_batch_size: Batch size per GPU/CPU for evaluation. Defaults to 8.
        gradient_accumulation_steps: Number of updates steps to accumulate gradients
            before performing a backward/update pass. Effective batch size will be
            `per_device_train_batch_size * num_devices * gradient_accumulation_steps`.
            Defaults to 1.
        training_shuffle: Whether to shuffle the training data at each epoch.
            Defaults to True.
        logging_steps: Log training loss and metrics every N global steps.
            Defaults to 100.
        save_steps: Save a checkpoint every N global steps. If None, checkpoints
            are only saved at the end of epochs (if `CheckpointCallback` is used).
            Defaults to 500.
        save_total_limit: If set, limits the total number of saved checkpoints.
            Older checkpoints will be deleted. If None, all checkpoints are kept.
            Needs to be implemented in `CheckpointCallback` or `Trainer`. Defaults to None.
        evaluate_during_training: Whether to run evaluation on the eval dataset
            during training. Defaults to False.
        evaluation_steps: If `evaluate_during_training` is True, evaluate every
            N global steps. If None, evaluation might occur at epoch ends if
            controlled by callbacks. Defaults to None.
        seed: Random seed for initialization and data shuffling. Defaults to 42.
        optim: An `OptimizerConfig` instance defining the optimizer and LR scheduler.
        loss: A `LossConfig` instance defining the loss function.
    """

    output_dir: str = "./training_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    training_shuffle: bool = True
    logging_steps: int = 100
    save_steps: Optional[int] = 500
    save_total_limit: Optional[int] = None  # Note: Needs implementation for deletion
    evaluate_during_training: bool = False
    evaluation_steps: Optional[int] = None
    seed: int = 42
    optim: OptimizerConfig
    loss: LossConfig

    @field_validator("gradient_accumulation_steps")
    @classmethod
    def check_grad_acc_steps(cls, v: int) -> int:
        """Validates gradient_accumulation_steps."""
        if v < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1.")
        return v
