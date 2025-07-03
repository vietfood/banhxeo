from __future__ import annotations

import jax
from jax import numpy as jnp


class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
        **kwargs,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.seed = seed

        self.config = kwargs

        self.indices = jnp.arange(len(dataset))
        self.current_step = 0
        self.total_steps = self._calculate_num_steps()

        self._on_batch_end()

    def _calculate_num_steps(self) -> int:
        """Calculates the number of steps per epoch."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _on_batch_end(self):
        self.current_step = 0
        if self.shuffle:
            rng = jax.random.key(self.seed)
            jax.random.permutation(key=rng, x=self.indices)
            self.seed += 1  # Ensure different shuffle next epoch

    def __len__(self) -> int:
        return self.total_steps

    def __iter__(self) -> DataLoader:
        self._on_batch_end()
        return self

    def __next__(self):
        if self.current_step >= self.total_steps:
            raise StopIteration

        start_idx = self.current_step * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_indices = self.indices[start_idx:end_idx]

        if len(batch_indices) < self.batch_size and not self.drop_last:
            pass

        if hasattr(self.dataset, "__getitems__"):
            batch_samples = self.dataset.__getitems__(batch_indices)
        else:
            batch_samples = [self.dataset[i] for i in batch_indices]

        self.current_step += 1
        return batch_samples
