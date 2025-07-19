from __future__ import annotations

from typing import Any, Dict, Iterator, List

import jax
from jax import numpy as jnp

from banhxeo import USE_TORCH

try:
    import torch
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import Dataset as TorchDataset
except ImportError:
    if USE_TORCH:
        raise ImportError(
            "PyTorch is not installed, but USE_TORCH is True. Please install it (`pip install torch`) or set USE_TORCH=False."
        )
    USE_TORCH = False
    # Define dummy types for type hints if torch is not installed
    TorchDataLoader = type("TorchDataLoader", (), {})
    TorchDataset = type("TorchDataset", (), {})


def _torch_collate_fn(dataset):
    def collate_fn(indices: List[int]) -> Dict[str, jax.Array]:
        return dataset.__getitems__(indices)

    return collate_fn


class NaiveDataLoader:
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

    def __iter__(self) -> NaiveDataLoader:
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


class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
        num_workers: int = 0,  # Torch-specific
        **kwargs,
    ):
        self.dataset = dataset

        if USE_TORCH:

            class TorchDummyDataset(TorchDataset):  # type: ignore
                """Dummy dataset to let's DataLoader work with our TextDataset"""

                def __init__(self, length):
                    self.length = length

                def __len__(self):
                    return self.length

                def __getitem__(self, index):
                    return index

            adapter = TorchDummyDataset(len(dataset))
            collate_fn = _torch_collate_fn(dataset)

            generator = torch.Generator()  # type: ignore
            generator.manual_seed(seed)

            self._loader = TorchDataLoader(
                dataset=adapter,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
                collate_fn=collate_fn,
                generator=generator,
                **kwargs,
            )
        else:
            if num_workers > 0:
                print(
                    "Warning: `num_workers > 0` has no effect when USE_TORCH=False. Using single-process native loader."
                )

            self._loader = NaiveDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=seed,
                **kwargs,
            )

    def __len__(self) -> int:
        return len(self._loader)  # type: ignore

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Returns an iterator over the batches."""
        return iter(self._loader)  # type: ignore
