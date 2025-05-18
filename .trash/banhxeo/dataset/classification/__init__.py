from typing import Dict, List, Optional, Union

import torch
from pydantic import BaseModel, field_validator
from torch.utils.data import Dataset

from banhxeo.core.tokenizer import Tokenizer
from banhxeo.core.vocabulary import Vocabulary
from banhxeo.dataset.raw import RawTextDataset
from banhxeo.dataset.transforms import ComposeTransforms, Transforms


class ClsfDatasetConfig(BaseModel):
    tokenizer: Tokenizer
    vocab: Vocabulary
    max_len: int = 256
    add_special_tokens: bool = False
    padding: Union[bool, str] = (
        False  # False = "do_not_pad", True = "longest", "max_length"
    )
    truncation: bool = False  # True = truncate to max_length
    transforms: Union[List[Transforms], ComposeTransforms] = []

    @field_validator("transforms")
    @classmethod
    def ensure_compose_transforms(cls, v):
        if isinstance(v, list):
            return ComposeTransforms(v)
        return v

    class Config:
        arbitrary_types_allowed = True


class TextClsfDataset(Dataset):
    """
    PyTorch Dataset for text classification tasks.
    Takes raw data, tokenizes, numericalizes, and applies padding/truncation.
    """

    def __init__(
        self,
        data: RawTextDataset,
        config: ClsfDatasetConfig,
        label_map: Optional[Dict[str, int]],  # e.g., {"positive": 1, "negative": 0}
    ):
        self.config = config
        self.label_map = label_map
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Inherit class should implement this
        raise NotImplementedError()

    def __getitems__(self, indexes: List[int]) -> List[Dict[str, torch.Tensor]]:
        # Inherit class should implement this
        raise NotImplementedError()


from .imdb import IMDBDataset  # noqa: E402
