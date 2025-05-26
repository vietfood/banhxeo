from typing import Dict

import torch

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from banhxeo.data.base import BaseTextDataset
from banhxeo.data.config import TorchDatasetConfig


class TorchTextDataset(TorchDataset):
    """A PyTorch Dataset wrapper for `BaseTextDataset`.

    This class handles the transformation of raw text samples from a
    `BaseTextDataset` into tokenized and numericalized PyTorch tensors,
    ready for model consumption. It applies tokenization, vocabulary mapping,
    and any specified text transformations.

    Attributes:
        text_dataset (BaseTextDataset): The underlying raw text dataset.
        config (TorchDatasetConfig): Configuration specifying tokenizer, vocabulary,
            text processing, and label handling.
    """

    def __init__(
        self,
        text_dataset: BaseTextDataset,
        config: TorchDatasetConfig,
    ):
        """Initializes the TorchTextDataset.

        Args:
            text_dataset: The instance of `BaseTextDataset` containing raw data.
            config: A `TorchDatasetConfig` object that defines how to process
                the raw data into tensors.
        """
        self.text_dataset = text_dataset
        self.config = config

    def __len__(self) -> int:  # noqa: D105
        return len(self.text_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retrieves and processes a single sample into a tensor dictionary.

        Fetches a raw sample from `text_dataset`, applies configured text
        transformations, tokenizes and encodes it using the tokenizer and
        vocabulary from `self.config`. If `is_classification` is True,
        it also processes and includes the label.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A dictionary containing:
                - "input_ids": LongTensor of token IDs.
                - "attention_mask": LongTensor indicating valid tokens (1) vs padding (0).
                - "labels" (optional): LongTensor of the label ID, if
                  `config.is_classification` is True.

        Raises:
            ValueError: If `is_classification` is True but a label cannot be
                obtained or mapped for the sample.
            IndexError: If `idx` is out of bounds for the underlying `text_dataset`.
        """
        raw_sample = self.text_dataset[idx]

        if isinstance(raw_sample, tuple):
            raw_text, raw_label = raw_sample
        elif isinstance(raw_sample, dict):
            raw_text = raw_sample[self.config.text_column_name]
            raw_label = (
                raw_sample.get(self.config.label_column_name)
                if self.config.label_column_name
                else None
            )
        else:  # Assuming raw_sample is just text
            raw_text = raw_sample
            raw_label = None

        if not isinstance(raw_text, str):
            raise ValueError(
                f"Expected raw_text to be a string, but got {type(raw_text)} for sample {idx}."
            )

        text = self.config.transforms(raw_text)  # type: ignore

        # Encode text
        encoded_output = self.config.tokenizer.encode(
            text, self.config.vocab, self.config.tokenizer_config
        )

        output_dict: Dict[str, torch.Tensor] = {
            "input_ids": torch.tensor(encoded_output["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                encoded_output["attention_mask"], dtype=torch.long
            ),
        }

        if self.config.is_classification:
            if raw_label is None:
                raise ValueError(
                    f"Label is None for sample {idx}, but is_classification is True."
                )
            if self.config.label_map:
                label_id = self.config.label_map.get(str(raw_label))
                if label_id is None:
                    raise ValueError(
                        f"Label '{raw_label}' not found in label_map: {self.config.label_map.keys()}"
                    )
            elif isinstance(raw_label, int):
                label_id = raw_label
            else:
                raise ValueError(
                    f"Label must be an int, castable to int, or label_map must be provided. Got {raw_label} ({type(raw_label)})"
                )
            output_dict["labels"] = torch.tensor(label_id, dtype=torch.long)

        return output_dict

    def to_loader(
        self, batch_size: int, num_workers: int, shuffle: bool = True, **kwargs
    ):
        """Creates a PyTorch DataLoader for this dataset.

        Args:
            batch_size: Number of samples per batch.
            num_workers: Number of subprocesses to use for data loading.
            shuffle: Whether to shuffle the data at every epoch. If `sampler`
                is provided, `shuffle` must be False (or will be ignored).
            collate_fn: Custom function to merge a list of samples to form a
                mini-batch of Tensor(s). If None, uses default PyTorch collate_fn.
                Note: Default collate_fn works well if `__getitem__` returns a
                dictionary of tensors, which this class does.
            sampler: Defines the strategy to draw samples from the dataset.
                If specified, `shuffle` must be False.
            **kwargs: Additional arguments passed directly to `torch.utils.data.DataLoader`.
                      (e.g., `pin_memory`, `drop_last`).

        Returns:
            A PyTorch DataLoader instance.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=kwargs.get("collate_fn"),
            sampler=kwargs.get("sampler"),
        )
