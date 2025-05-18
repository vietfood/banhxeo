from typing import Any, Dict, List, Optional

import torch

from banhxeo.dataset.classification import ClsfDatasetConfig, TextClsfDataset
from banhxeo.dataset.raw import IMDBRawDataset


class IMDBDataset(TextClsfDataset):
    def __init__(
        self,
        data: IMDBRawDataset,
        config: ClsfDatasetConfig,
        label_map: Optional[Dict[str, int]] = {"pos": 1, "neg": 0},
    ):
        super().__init__(data=data, config=config, label_map=label_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # For some convenients
        transforms = self.config.transforms
        vocab = self.config.vocab
        tokenizer = self.config.tokenizer
        max_len = self.config.max_len

        # Get raw data
        raw_text, label = self.data[idx]
        text = transforms(raw_text)  # type: ignore

        # Create tokens
        tokens = [vocab.sos_tok] + tokenizer(text)[: max_len - 2] + [vocab.eos_tok]
        input_ids = vocab.tokens_to_ids(tokens)  # type: ignore

        attention_mask = [1] * len(input_ids)
        padding_length = self.config.max_len - len(input_ids)

        if padding_length < 0:  # Should not happen if truncation is correct
            # This might indicate an issue with max_tokens_for_text calculation or token list lengths
            # For robustness, clip input_ids and attention_mask if they somehow exceed max_len
            input_ids = input_ids[: self.config.max_len]
            attention_mask = attention_mask[: self.config.max_len]
            padding_length = 0  # No padding needed if already over or at max_len

        input_ids += [self.config.vocab.pad_id] * padding_length
        attention_mask += [0] * padding_length

        # Convert label to one-hot vector
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
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }

    def __getitems__(self, indexes: List[int]) -> List[Dict[str, torch.Tensor]]:
        return [self.__getitem__(idx) for idx in indexes]
