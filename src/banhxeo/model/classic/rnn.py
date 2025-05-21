from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Integer
from pydantic import model_validator
from typing_extensions import Self

from banhxeo.core.vocabulary import Vocabulary
from banhxeo.model.neural import NeuralLanguageModel, NeuralModelConfig


class RNNConfig(NeuralModelConfig):
    hidden_size: int  # For one cell
    bidirectional: bool = False  # Bidirectional RNN
    num_layers: int = 1
    dropout: float = 0.0
    bias: bool = False

    @model_validator(mode="after")
    def check_config(self) -> Self:
        if self.num_layers <= 0:
            raise ValueError("Number of layers is 1 minimum")
        if self.bidirectional and self.num_layers > 1:
            raise ValueError("Current RNN didn't support stacked bidirectional")
        return self


class RNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias: bool):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_gate = nn.Linear(input_size, hidden_size, bias=bias)
        self.hidden_gate = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(
        self, input_t: torch.Tensor, h_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = input_t.shape[0]

        if h_prev is None:
            h_prev = torch.zeros(
                batch_size, self.hidden_size, device=input_t.device, dtype=input_t.dtype
            )

        h_next = F.tanh(self.input_gate(input_t) + self.hidden_gate(h_prev))

        return h_next


class RNN(NeuralLanguageModel):
    """
    Use batch_first as default
    """

    def __init__(
        self,
        vocab: Vocabulary,
        embedding_dim: int,
        hidden_size: int,
        bias: bool = False,
    ):
        super().__init__(
            model_config=RNNConfig(
                embedding_dim=embedding_dim, hidden_size=hidden_size, bias=bias
            ),
            vocab=vocab,
        )
        self.config: RNNConfig

        self.input_size = self.config.embedding_dim
        self.hidden_size = hidden_size

        self.embedding_layer = nn.Embedding(
            num_embeddings=self.vocab.vocab_size,
            embedding_dim=self.config.embedding_dim,
            padding_idx=self.vocab.pad_id,
        )

        if self.config.num_layers > 1:
            raise ValueError("Stacked RNN isn't implemented now")
            # self.rnn_cells = nn.ModuleDict()
            # for layer in range(self.config.num_layers):
            #     self.rnn_cells[f"layer_{layer}"] = RNNCell(
            #         self.input_size, self.hidden_size, self.config.bias
            #     )
        else:
            self.rnn_cells = RNNCell(
                self.input_size, self.hidden_size, self.config.bias
            )

    def forward(
        self,
        input_ids: Integer[torch.Tensor, "batch seq"],  # noqa: F722
        attention_mask: Optional[Integer[torch.Tensor, "batch seq"]] = None,  # noqa: F722
        **kwargs,
    ):
        outputs_list = []

        # get original sequences length (we need cpu tensor for pack_padded_sequence)
        original_seqs_len = einops.reduce(
            attention_mask, "batch seq -> batch", "sum"
        ).to("cpu")  # type: ignore

        # Attention_mask: [batch, seq]
        # Input_ids: [batch, seq]
        # Embeddings: [batch, seq, embed_dim]
        embeddings = self.embedding_layer(input_ids)

        # Pack input for RNN
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
            input=embeddings,
            lengths=original_seqs_len,  # type: ignore
            batch_first=True,
            enforce_sorted=False,
        )

        # Then unpack the input (again)
        inputs, batch_sizes, sorted_indices, unsorted_indices = packed_inputs

        # Batch size at the first (longest) time step
        effective_batch_size = batch_sizes[0].item()

        # Create initial hidden_state
        h_prev = torch.zeros(
            effective_batch_size,  # type: ignore
            self.hidden_size,
            device=inputs.device,
            dtype=inputs.dtype,
        )

        max_seq_len = batch_sizes.size(0)

        # For packed sequence, inputs is already (total_tokens_across_all_seqs, input_size)
        # We need to slice it based on batch_sizes
        last_processed_idx = 0

        for t in range(max_seq_len):
            # Get the current batch size for this time step
            current_batch_size = batch_sizes[t].item()

            # input_t will have shape (current_batch_size, input_size)
            input_t = inputs[
                last_processed_idx : last_processed_idx + current_batch_size
            ]
            last_processed_idx += current_batch_size

            # We need to ensure h_prev is also sliced to current_batch_size
            # This is important because as sequences end, the effective batch size shrinks
            h_prev_t = h_prev[:current_batch_size]

            # Pass through our RNN cell
            h_next_t = self.rnn_cells(input_t, h_prev_t)

            outputs_list.append(h_next_t)

            if current_batch_size < h_prev.shape[0]:
                # If batch size shrunk, carry over old hidden states for sequences that ended
                # This is tricky. The simplest is to ensure h_prev is always sized for max batch size
                # and only update the relevant parts.
                h_prev = torch.cat((h_next_t, h_prev[current_batch_size:]), dim=0)
            else:
                h_prev = h_next_t

        outputs_packed_data = torch.cat(outputs_list, dim=0)

        # Re-pack the outputs
        outputs = nn.utils.rnn.PackedSequence(
            outputs_packed_data, batch_sizes, sorted_indices, unsorted_indices
        )

        # For packed inputs, we need to unsort this h_prev
        final_hidden_states = h_prev.clone()  # It's already for the max_batch_size
        h_n = einops.rearrange(
            final_hidden_states[unsorted_indices],
            "batch hidden -> 1 batch hidden",  # PyTorch's nn.RNN h_n output is (num_layers * num_directions, batch_size, hidden_size)
        )

        return {"hidden_states": outputs, "last_hidden_state": h_n}
