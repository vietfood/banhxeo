from typing import Dict, List, Optional

import einops
import torch

from jaxtyping import Integer
from pydantic import Field, model_validator
from torch import nn
from typing_extensions import Self

from banhxeo.core.vocabulary import Vocabulary
from banhxeo.model.neural import NeuralLanguageModel, NeuralModelConfig
from banhxeo.utils.logging import DEFAULT_LOGGER


class MLPConfig(NeuralModelConfig):
    """Configuration for the Multi-Layer Perceptron (MLP) model.

    Attributes:
        output_size: The number of output units (e.g., number of classes for
            classification).
        hidden_sizes: A list of integers, where each integer is the number of
            units in a hidden layer. E.g., `[256, 128]` for two hidden layers.
        activation_fn: The activation function to use in hidden layers.
            Supported: "relu", "tanh", "gelu", "sigmoid". Defaults to "relu".
        dropout_rate: Dropout rate to apply after activation in hidden layers.
            Must be between 0.0 and 1.0. Defaults to 0.0.
        aggregate_strategy: Strategy to aggregate token embeddings from a sequence
            into a single vector before passing to the MLP. Supported:
            "average", "max", "sum". "concat_window" is planned.
            Defaults to "average".
        window_size: Required if `aggregate_strategy` is "concat_window".
            Specifies the window size for concatenating embeddings.
        embedding_dim (from NeuralModelConfig): Dimension of the input embeddings.
    """

    output_size: int
    hidden_sizes: List[int] = Field(
        default_factory=list
    )  # e.g., [256, 128] -> two hidden layers
    activation_fn: str = "relu"
    dropout_rate: float = 0.0

    aggregate_strategy: str = "average"
    window_size: Optional[int] = None

    @model_validator(mode="after")
    def check_valid(self) -> Self:
        """Validates the MLP configuration."""
        supported_aggregates = ["average", "max", "sum", "concat_window"]
        if self.aggregate_strategy not in supported_aggregates:
            raise ValueError(
                f"Unsupported aggregation strategy: {self.aggregate_strategy}. "
                f"Supported: {supported_aggregates}"
            )
        if self.aggregate_strategy == "concat_window" and (
            self.window_size is None or self.window_size <= 0
        ):
            raise ValueError(
                "If aggregate_strategy is 'concat_window', a positive 'window_size' must be provided."
            )

        activation_options = ["relu", "tanh", "gelu", "sigmoid"]
        if self.activation_fn.lower() not in activation_options:
            raise ValueError(
                f"Unsupported activation_fn: {self.activation_fn}. Supported: {activation_options}"
            )
        self.activation_fn = self.activation_fn.lower()

        if not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError(
                "dropout_rate must be between 0.0 and 1.0 (exclusive of 1.0)"
            )
        return self


class MLP(NeuralLanguageModel):
    """Multi-Layer Perceptron model for sequence classification or regression.

    This model takes token embeddings, aggregates them into a single vector
    using a specified strategy (e.g., averaging), and then passes this
    vector through one or more fully connected layers to produce an output.
    """

    ConfigClass = MLPConfig  # for Loading

    def __init__(
        self,
        vocab: Vocabulary,
        output_size: int,
        embedding_dim: int = 128,
        hidden_sizes: List[int] = [256],
        **kwargs,
    ):
        """Initializes the MLP model.

        Args:
            vocab: The vocabulary instance.
            output_size: The dimensionality of the output layer.
            embedding_dim: The dimensionality of the input token embeddings.
                Defaults to 128.
            hidden_sizes: A list of hidden layer sizes. Defaults to `[256]`.
            **kwargs: Additional arguments for `MLPConfig`, such as
                `activation_fn`, `dropout_rate`, `aggregate_strategy`.
        """
        super().__init__(
            vocab=vocab,
            model_config=MLPConfig(
                output_size=output_size,
                hidden_sizes=hidden_sizes,
                embedding_dim=embedding_dim,
                activation_fn=kwargs.get("activation_fn", "relu"),
                dropout_rate=kwargs.get("dropout_rate", 0.0),
                aggregate_strategy=kwargs.get("aggregate_strategy", "average"),
                window_size=kwargs.get("window_size"),
            ),
        )
        self.config: MLPConfig

        self.embedding = nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=self.config.embedding_dim,  # type: ignore
            padding_idx=vocab.pad_id,
        )

        if self.config.aggregate_strategy == "concat_window":
            current_input_dim = self.config.window_size * self.config.embedding_dim  # type: ignore
        else:  # "average", "max", "sum" all result in embedding_dim
            current_input_dim = self.config.embedding_dim

        self.layers = nn.Sequential()

        # Hidden Layers
        if self.config.hidden_sizes:
            for i, hidden_dim in enumerate(self.config.hidden_sizes):
                self.layers.add_module(
                    f"linear_{i}",
                    nn.Linear(current_input_dim, hidden_dim),  # type: ignore
                )
                if self.config.activation_fn == "relu":
                    self.layers.add_module(f"activation_{i}", nn.ReLU())

                elif self.config.activation_fn == "tanh":
                    self.layers.add_module(f"activation_{i}", nn.Tanh())

                elif self.config.activation_fn == "gelu":
                    self.layers.add_module(f"activation_{i}", nn.GELU())

                elif self.config.activation_fn == "sigmoid":
                    self.layers.add_module(f"activation_{i}", nn.Sigmoid())

                if self.config.dropout_rate > 0.0:
                    self.layers.add_module(
                        f"dropout_{i}", nn.Dropout(self.config.dropout_rate)
                    )
                current_input_dim = hidden_dim  # For the next layer

        # Output Layer
        self.layers.add_module(
            "output_linear",
            nn.Linear(current_input_dim, self.config.output_size),  # type: ignore
        )

    def forward(
        self,
        input_ids: Integer[torch.Tensor, "batch seq"],  # noqa: F722
        attention_mask: Optional[Integer[torch.Tensor, "batch seq"]] = None,  # noqa: F722
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Performs the forward pass of the MLP.

        Args:
            input_ids: Tensor of token IDs with shape `(batch_size, seq_len)`.
            attention_mask: Optional tensor indicating valid tokens (1) and padding (0)
                with shape `(batch_size, seq_len)`. If None and `padding_idx`
                is set for embeddings, a warning is issued and all tokens are
                assumed valid.
            **kwargs: Additional keyword arguments (ignored by this base forward pass).

        Returns:
            A dictionary containing:
                - "logits": Output logits from the MLP, shape `(batch_size, output_size)`.

        Raises:
            NotImplementedError: If an unsupported `aggregate_strategy` is configured
                                (though `check_valid` should catch this).
        """
        if attention_mask is None and self.embedding.padding_idx is not None:
            DEFAULT_LOGGER.warning(
                "Fallback: Assume all token is valid because attention_mask is None"
            )
            #  Not ideal for padded sequences.
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)

        # input_ids: (batch_size, seq_len)
        # embeddings: (batch_size, seq_len, embedding_dim)
        embeddings = self.embedding(input_ids)

        # attention_mask: (batch_size, seq_len) -> (batch_size, seq_len, 1) for broadcasting
        mask_expanded = einops.rearrange(
            attention_mask, "batch seq -> batch seq 1"
        ).float()  # type: ignore

        if self.config.aggregate_strategy == "average":
            # Corrected average with einops
            masked_embeddings = embeddings * mask_expanded
            summed_embeddings = einops.reduce(
                masked_embeddings, "batch seq dim -> batch dim", "sum"
            )

            # Count non-padded tokens per batch item
            # mask_expanded is (batch, seq, 1) and contains 0s or 1s
            num_valid_tokens = einops.reduce(
                mask_expanded, "batch seq 1 -> batch 1", "sum"
            )
            num_valid_tokens = torch.clamp(
                num_valid_tokens, min=1e-9
            )  # Avoid division by zero

            aggregated_embeddings = summed_embeddings / num_valid_tokens
        elif self.config.aggregate_strategy == "max":
            # To correctly max-pool with padding, set padding embeddings to a very small number
            # before max-pooling, so they are not chosen.
            aggregated_embeddings = einops.reduce(
                embeddings.masked_fill((mask_expanded == 0), float("inf")),
                "batch seq dim -> batch dim",
                "max",
            )
        elif self.config.aggregate_strategy == "sum":
            aggregated_embeddings = einops.reduce(
                embeddings * mask_expanded, "batch seq dim -> batch dim", "sum"
            )

        logits = self.layers(aggregated_embeddings)  # type: ignore
        return {"logits": logits}

    def generate_sequence(self, *args, **kwargs) -> str:  # noqa: D102
        raise NotImplementedError("MLP is not an autoregressive model.")
