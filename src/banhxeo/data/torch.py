from typing import Dict

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from banhxeo.data.base import BaseTextDataset
from banhxeo.data.config import TorchDatasetConfig


class TorchTextDataset(TorchDataset):
    """
    A PyTorch Dataset that wraps a TextDataset, applying tokenization,
    numericalization, and transformations.
    """

    def __init__(
        self,
        text_dataset: BaseTextDataset,
        config: TorchDatasetConfig,
    ):
        self.text_dataset = text_dataset
        self.config = config

    def __len__(self) -> int:
        return len(self.text_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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

        text = self.config.transforms(raw_text)  # type: ignore

        encoded_output = self.config.tokenizer.encode(
            text, self.config.vocab, self.config.tokenizer_config
        )

        output_dict = {
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
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=kwargs.get("collate_fn"),
            sampler=kwargs.get("sampler"),
        )
