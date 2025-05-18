from typing import Any, Dict, List, Optional

import torch

from banhxeo.dataset.cls import ClsDatasetConfig, TextClsDataset
from banhxeo.dataset.raw.imdb import IMDBDataset


class IMDBClsDataset(TextClsDataset):
    def __init__(
        self,
        data: IMDBDataset,
        config: ClsDatasetConfig,
        label_map: Optional[Dict[str, int]] = {"pos": 1, "neg": 0},
    ):
        super().__init__(data=data, config=config, label_map=label_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw_text, label = self.data[idx]
        text = self.config.transforms(raw_text)  # type: ignore

        output = self.config.tokenizer.encode(
            text, self.config.vocab, self.config.tokenizer_config
        )

        if self.label_map:
            label_id = self.label_map.get(label)
            if label_id is None:
                raise ValueError(
                    f"Label '{label}' not found in label_map: {self.label_map.keys()}"
                )
        elif isinstance(label, int):
            label_id = label
        else:
            raise ValueError(
                f"Label must be an int or label_map must be provided. Got {label} ({type(label)})"
            )

        return {
            "text": raw_text,
            "processed_text": text,
            "input_ids": torch.tensor(output["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(output["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }

    def __getitems__(self, indexes: List[int]) -> List[Dict[str, torch.Tensor]]:
        return [self.__getitem__(idx) for idx in indexes]
