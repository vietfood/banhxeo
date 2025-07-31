from typing import Callable, Optional

import flax.linen as nn
import jax
from jaxtyping import Float


class MLP(nn.Module):
    hidden_dim: int
    bias: bool = True
    multiplier: int = 4
    activation: Callable = jax.nn.relu
    dtype: Optional[str] = None

    @nn.compact
    def __call__(
        self,
        x: Float[jax.Array, "batch seq_len hidden_dim"],  # noqa: F722
        deterministic: bool = False,
    ):
        x = nn.Dense(
            self.multiplier * self.hidden_dim,
            dtype=self.dtype,
            use_bias=self.bias,
            name="c_fc",
        )(x)
        x = self.activation(x, approximate=True)
        x = nn.Dense(
            self.hidden_dim, dtype=self.dtype, use_bias=self.bias, name="c_proj"
        )(x)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic)

        return x
