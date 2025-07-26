from typing import List, Literal

import einops
import jax
import jax.nn as jnn
from flax import linen as nn
from jax import numpy as jnp
from jaxtyping import Integer


class MLP(nn.Module):
    vocab_size: int
    output_size: int
    embedding_dim: int
    hidden_sizes: List[int]
    pad_id: int = 0
    activation_fn: Literal["relu", "tanh", "gelu", "sigmoid"] = "relu"
    dropout_rate: float = 0.0
    aggregate_strategy: str = "mean"

    @nn.compact
    def __call__(
        self,
        input_ids: Integer[jax.Array, "batch seq"],  # noqa: F722
        attention_mask: Integer[jax.Array, "batch seq"],  # noqa: F722
        dropout: bool = True,
    ):
        embeddings = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embedding_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )(input_ids)

        # attention_mask: (batch_size, seq_len) -> (batch_size, seq_len, 1) for broadcasting
        mask_expanded = einops.rearrange(
            attention_mask, "batch seq -> batch seq 1"
        ).astype(jnp.float32)

        match self.aggregate_strategy:
            case "mean":
                # sum then divide by valid-token count
                summed = einops.reduce(
                    embeddings * mask_expanded, "batch seq dim -> batch dim", "sum"
                )
                count = einops.reduce(
                    mask_expanded, "batch seq 1 -> batch 1", "sum"
                ).clip(min=1e-9)
                x = summed / count
            case "sum":
                x = einops.reduce(
                    embeddings * mask_expanded, "batch seq dim -> batch dim", "sum"
                )
            case "max":
                neg_inf = jnp.finfo(embeddings.dtype).min
                emb_masked = jnp.where(
                    mask_expanded == 1.0,  # Check if token is valid
                    embeddings,
                    neg_inf,
                )
                x = einops.reduce(emb_masked, "batch seq dim -> batch dim", "max")
            case "concat":
                # flatten seq into feature dim
                x = einops.rearrange(
                    embeddings * mask_expanded, "batch seq dim -> batch (seq dim)"
                )
            case _:
                raise ValueError(
                    f"Unknown aggregate_strategy={self.aggregate_strategy!r}"
                )

        for hidden_dim in self.hidden_sizes:
            x = nn.Dense(hidden_dim)(x)
            x = getattr(jnn, self.activation_fn.lower())(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not dropout)

        logits = nn.Dense(self.output_size)(x)
        return logits
