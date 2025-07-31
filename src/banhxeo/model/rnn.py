from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Float, Integer

from banhxeo.core.layer.cell import RNNCell


class RNN(nn.Module):
    hidden_dim: int
    pad_id: int
    bias: bool = True

    @nn.compact
    def __call__(
        self,
        input_ids: Integer[jax.Array, "batch seq"],
        attention_mask: Integer[jax.Array, "batch seq"],
    ):
        def calculate_hidden_state(
            carry_h: Float[jax.Array, "batch hidden_dim"],
            x: Tuple[Float[jax.Array, "batch hidden_dim"], Integer[jax.Array, "batch"]],
        ):
            input_t, mask_t = x

            h_next = jnp.where(
                (mask_t != self.pad_id)[:, None],
                RNNCell(hidden_dim=self.hidden_dim, bias=self.bias)(carry_h, input_t),
                carry_h,
            )

            # The scan function needs to return (new_carry, y_output)
            return (h_next, h_next)

        # create embeddings
        embeddings = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embedding_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )(input_ids)

        # We need to swap the batch and seq_len axes.
        # Why? Because we want to iterate over the sequence.
        inputs_swapped = jnp.swapaxes(embeddings, 0, 1)
        mask_swapped = jnp.swapaxes(attention_mask, 0, 1)
        scan_inputs = (inputs_swapped, mask_swapped)

        # create first hidden state
        h_initial = jnp.zeros(shape=(embeddings.shape[0], self.hidden_dim))

        final_h, all_h_swapped = nn.scan(
            calculate_hidden_state,
            variable_broadcast="params",
            split_rngs={"params": False},
        )(h_initial, scan_inputs)

        # Our all hidden states has shape (seq_len, batch, hidden_dim)
        all_h = jnp.swapaxes(all_h_swapped, 1, 0)

        return all_h, final_h
