from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Optional, Union

from banhxeo import DEFAULT_SEED
from banhxeo.core.tokenizer import ProcessConfig, Tokenizer
from banhxeo.data.base import BaseTextDataset
from banhxeo.data.loader import DataLoader


@dataclass
class ArrayDatasetConfig:
    # For tokenizer
    tokenizer: Tokenizer
    encode_config: ProcessConfig

    return_tensors: Optional[Literal["jax", "np"]] = None

    # For classification
    is_classification: bool = False
    label_map: Dict[str, int] = field(default_factory=lambda: {"pos": 1, "neg": 0})


class ArrayTextDataset:
    def __init__(
        self,
        base_dataset: BaseTextDataset,
        config: ArrayDatasetConfig,
    ):
        self.base_dataset = base_dataset
        self.config = config

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.__getitems__([index])

    def __getitems__(self, indices: Iterable[int]) -> Dict[str, Any]:
        batch_texts = []
        batch_labels = []

        for idx in indices:
            raw_sample = self.base_dataset[idx]
            if isinstance(raw_sample, tuple):
                raw_text, raw_label = raw_sample
            elif isinstance(raw_sample, dict):
                raw_text = raw_sample[self.base_dataset.config.text_column]
                raw_label = (
                    raw_sample.get(self.base_dataset.config.label_column)
                    if self.base_dataset.config.label_column
                    else None
                )
            else:  # Assuming raw_sample is just text
                raw_text = raw_sample
                raw_label = None

            if not isinstance(raw_text, str):
                raise ValueError(
                    f"Expected raw_text to be a string, but got {type(raw_text)} for sample {idx}."
                )

            batch_texts.append(raw_text)
            batch_labels.append(raw_label)

        outputs = self.config.tokenizer(
            batch_texts,
            return_tensors=self.config.return_tensors,
            **self.config.encode_config.dict(),
        )

        if self.config.is_classification:
            labels = [0] * len(batch_labels)
            for idx, raw_label in enumerate(batch_labels):
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
                labels[idx] = label_id

            if self.config.return_tensors == "jax":
                import jax.numpy as jnp

                labels = jnp.array(labels, dtype=jnp.int32)
            elif self.config.return_tensors == "np":
                import numpy as np

                labels = np.array(labels, dtype=np.int32)

            return {**outputs, "labels": labels}  # type: ignore
        else:
            return outputs  # type: ignore

    def get(self, index: Union[Iterable[int], int]):
        return (
            self.__getitems__(index)
            if isinstance(index, Iterable)
            else self.__getitem__(index)
        )

    def to_loader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = DEFAULT_SEED,
        num_workers=8,
        **kwargs,
    ):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            num_workers=num_workers,
            **kwargs,
        )
