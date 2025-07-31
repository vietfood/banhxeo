import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxtyping import Float


class RNNCell(nn.Module):
    hidden_dim: int
    bias: bool = False

    @nn.compact
    def __call__(
        self,
        h_prev: Float[jax.Array, "batch hidden_dim"],
        input_t: Float[jax.Array, "batch embed_dim"],
    ):
        input_gate = nn.Dense(
            features=self.hidden_dim, use_bias=self.bias, name="input_gate"
        )

        hidden_gate = nn.Dense(
            features=self.hidden_dim, use_bias=self.bias, name="hidden_gate"
        )

        h_next = jnn.tanh(input_gate(input_t) + hidden_gate(h_prev))

        return h_next


class LSTMCell(nn.Module):
    hidden_dim: int
    bias: bool = False

    @nn.compact
    def __call__(
        self,
        h_prev: Float[jax.Array, "batch hidden_dim"],
        c_prev: Float[jax.Array, "batch hidden_dim"],
        input_t: Float[jax.Array, "batch embed_dim"],
    ):
        # pre-defined new gate
        forget_gate = nn.Dense(
            features=self.hidden_dim, use_bias=self.bias, name="forget_gate"
        )

        input_gate = nn.Dense(
            features=self.hidden_dim, use_bias=self.bias, name="input_gate"
        )

        output_gate = nn.Dense(
            features=self.hidden_dim, use_bias=self.bias, name="output_gate"
        )

        candidate_gate = nn.Dense(
            features=self.hidden_dim, use_bias=self.bias, name="candidate_gate"
        )

        # concat on dim axis
        concat_inp = jnp.concat([h_prev, input_t], axis=1)

        forget = jnn.sigmoid(forget_gate(concat_inp))
        inp = jnn.sigmoid(input_gate(concat_inp))
        out = jnn.sigmoid(output_gate(concat_inp))

        # new candidate cell state
        candidate_cell = jnn.tanh(candidate_gate(concat_inp))

        # new cell state
        new_cell_state = forget * c_prev + inp * candidate_cell

        # new_state
        new_hidden_state = out * jnp.tanh(new_cell_state)

        return new_hidden_state, new_cell_state
