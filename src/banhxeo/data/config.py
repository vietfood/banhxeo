from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, field_validator

from banhxeo.core.tokenizer import Tokenizer, TokenizerConfig
from banhxeo.core.vocabulary import Vocabulary
from banhxeo.data.transforms import ComposeTransforms, Transforms


@dataclass
class DatasetSplit:
    train: int
    test: int
    val: Optional[int] = None


@dataclass
class DownloadDatasetFile:
    name: str
    ext: str
    source: Optional[str] = None


class DatasetConfig(BaseModel):
    name: str

    # For file-based datasets:
    url: Optional[str] = None
    file_info: Optional[DownloadDatasetFile] = None
    md5: Optional[str] = None

    # For HF datasets:
    hf_path: Optional[str] = None
    hf_name: Optional[str] = None  # For subsets/configurations of HF datasets
    text_column: str = "text"  # Default text column for HF
    label_column: Optional[str] = "label"  # Default label column for HF

    # Common
    split: Optional[DatasetSplit] = None  # Keep DatasetSplit if used


class TorchDatasetConfig(BaseModel):
    tokenizer: Tokenizer
    tokenizer_config: TokenizerConfig
    vocab: Vocabulary
    is_classification: bool = False
    transforms: Union[List[Transforms], ComposeTransforms] = []
    label_map: Optional[Dict[str, int]]

    # For Hugging face
    text_column_name: str = "text"
    label_column_name: Optional[str] = "label"

    @field_validator("transforms")
    @classmethod
    def ensure_compose_transforms(cls, v):
        if isinstance(v, list):
            return ComposeTransforms(v)
        return v

    class Config:
        arbitrary_types_allowed = True
