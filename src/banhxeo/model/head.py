from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        # Optionally, specify what key this head expects from the base model's output
        self.expected_input_key: Optional[str] = None

    @abstractmethod
    def forward(
        self, base_model_output_feature: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError


class ClassificationHead(BaseHead):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout_rate: float = 0.1,
    ):
        super().__init__(input_dim)
        self.num_classes = num_classes
        self.expected_input_key = (
            "pooled_output"  # Or "last_hidden_state" if CLS token is taken
        )

        layers = []
        current_dim = input_dim
        if hidden_dim:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(
        self, base_model_output_feature: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Args:
            pooled_output (torch.Tensor): Tensor of shape (batch_size, input_dim),
                                          typically the pooled output of the base model.
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        return self.classifier(base_model_output_feature)
