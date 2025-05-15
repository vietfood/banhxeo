from typing import Any, Dict, Optional

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
        raw_item = self.data[idx]
        raw_text, label = raw_item["content"], raw_item["label"]
        text = transforms(raw_text)  # type: ignore

        # Creat tokens and input_ids
        tokens = [vocab.sos_tok] + tokenizer(text)[: max_len - 2] + [vocab.eos_tok]
        input_ids = vocab.tokens_to_ids(tokens)

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

        return {"input_ids": input_ids, "labels": label_id}
