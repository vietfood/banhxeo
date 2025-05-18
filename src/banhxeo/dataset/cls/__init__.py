from typing import Dict, List, Optional, Union

import torch
from pydantic import BaseModel, field_validator
from torch.utils.data import Dataset

from banhxeo.core.tokenizer import Tokenizer, TokenizerConfig
from banhxeo.core.vocabulary import Vocabulary
from banhxeo.dataset.raw import RawTextDataset
from banhxeo.dataset.transforms import ComposeTransforms, Transforms


class ClsDatasetConfig(BaseModel):
    tokenizer: Tokenizer
    tokenizer_config: TokenizerConfig
    vocab: Vocabulary
    transforms: Union[List[Transforms], ComposeTransforms] = []

    @field_validator("transforms")
    @classmethod
    def ensure_compose_transforms(cls, v):
        if isinstance(v, list):
            return ComposeTransforms(v)
        return v

    class Config:
        arbitrary_types_allowed = True


class TextClsDataset(Dataset):
    """
    PyTorch Dataset for text classification tasks.
    Takes raw data, tokenizes, numericalizes, and applies padding/truncation.
    """

    def __init__(
        self,
        data: RawTextDataset,
        config: ClsDatasetConfig,
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
