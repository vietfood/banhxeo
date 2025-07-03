import os
import shutil
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jax
import polars as pl
from datasets import Dataset
from jax import numpy as jnp
from pydantic import BaseModel, field_validator

from banhxeo import DEFAULT_SEED
from banhxeo.core.tokenizer import EncodeConfig, Tokenizer
from banhxeo.data.loader import DataLoader
from banhxeo.data.transforms import ComposeTransforms, Transforms
from banhxeo.utils.file import check_md5, download_archive, extract_archive
from banhxeo.utils.logging import default_logger


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


class TextDatasetConfig(BaseModel):
    transforms: Union[List[Transforms], ComposeTransforms] = []

    # For classification
    is_classification: bool = False
    label_map: Dict[str, int] = {"pos": 1, "neg": 0}

    # For tokenizer
    tokenizer: Tokenizer
    encode_config: EncodeConfig

    @field_validator("transforms")
    @classmethod
    def ensure_compose_transforms(cls, v):
        if isinstance(v, list):
            return ComposeTransforms(v)
        return v

    class Config:
        arbitrary_types_allowed = True


class BaseTextDataset(metaclass=ABCMeta):
    def __init__(
        self,
        root_dir: Optional[str],
        split_name: str,
        config: DatasetConfig,
        seed: int,
        download: bool = True,
    ):
        if not root_dir:
            default_logger.warning(
                "root_dir is None or empty. Using current working directory as root."
            )
            self.root_path = Path.cwd()
        else:
            self.root_path = Path(root_dir).resolve()

        self.config = config
        self.dataset_base_path = self.root_path / "datasets" / self.config.name

        if self.config.url or self.config.file_info:  # Dataset might be local only
            self.dataset_base_path.mkdir(parents=True, exist_ok=True)
            default_logger.info(
                f"Dataset '{self.config.name}' will be stored in/loaded from: {self.dataset_base_path}"
            )

        self.split_name = split_name
        self.seed = seed
        self.max_workers = min(
            32, (os.cpu_count() or 1) + 4
        )  # Ensure os.cpu_count() not None

        # Data loading logic will be in subclasses
        self._data: Any = None

        if download and self.config.url and self.config.file_info:
            try:
                self._download_and_extract_data()
            except Exception as e:
                default_logger.error(
                    f"Failed to download/extract {self.config.name}: {e}. "
                    "Dataset may not be available."
                )
                raise e

    def _download_and_extract_data(self):
        if self.config.file_info is None:
            raise

        archive_file_path = (
            self.dataset_base_path
            / f"{self.config.file_info.name}.{self.config.file_info.ext}"
        )

        # Download file if not exist
        if not archive_file_path.is_file():
            default_logger.info(
                f"Downloading {self.config.name} dataset to {str(archive_file_path)}..."
            )
            try:
                download_archive(
                    self.config.file_info.source,
                    self.config.url,  # type: ignore
                    archive_file_path,
                )
            except Exception as e:
                default_logger.error(f"Cannot download {self.config.url}: {e}")
                if archive_file_path.exists():
                    archive_file_path.unlink()
                raise e

        # Check MD5 (can raise exception)
        check_md5(self.config.md5, self.config.name, archive_file_path)

        # File extract
        extracted_data_dir_path = self.dataset_base_path / self.config.file_info.name
        if not extracted_data_dir_path.is_dir():
            default_logger.info(
                f"Extracting file {archive_file_path.name} to {str(self.dataset_base_path)}..."
            )
            try:
                extract_archive(
                    self.config.file_info.ext,
                    archive_file_path,
                    self.dataset_base_path,
                    extracted_data_dir_path,
                )
                default_logger.info(
                    f"Successfully extracted to {str(extracted_data_dir_path)}"
                )
            except Exception as e:
                default_logger.error(
                    f"Error unpacking archive {str(archive_file_path)}: {e}"
                )
                if extracted_data_dir_path.exists():
                    shutil.rmtree(extracted_data_dir_path, ignore_errors=True)
                raise e

    def __len__(self) -> int:
        return len(self._data) if self._data is not None else 0

    def __getitem__(self, index: int):
        raise NotImplementedError("Subclass should implement this")

    def get_all_texts(self) -> List[str]:
        if self._data is None:
            default_logger.warning(
                f"Dataset '{self.config.name}' (split: {self.split_name}) "
                "is not loaded. Returning empty list of texts."
            )
            return []

        if isinstance(self._data, Dataset):
            return self._data[self.config.text_column]
        elif isinstance(self._data, pl.DataFrame):
            return self._data[self.config.text_column].to_list()
        else:  # slow path
            texts = []
            for i in range(len(self)):
                try:
                    sample = self[i]
                    if isinstance(sample, str):
                        texts.append(sample)
                    elif (
                        isinstance(sample, tuple)
                        and len(sample) > 0
                        and isinstance(sample[0], str)
                    ):
                        texts.append(sample[0])
                    elif isinstance(sample, dict) and self.config.text_column in sample:
                        text_content = sample[self.config.text_column]
                        if isinstance(text_content, str):
                            texts.append(text_content)
                        else:
                            default_logger.warning(
                                f"Sample {i} text column '{self.config.text_column}' "
                                f"is not a string: {type(text_content)}. Skipping."
                            )
                    else:
                        default_logger.warning(
                            f"Cannot extract text from sample {i} of type {type(sample)}. Skipping."
                        )
                except Exception as e:
                    default_logger.warning(
                        f"Error processing sample {i} in get_all_texts: {e}. Skipping."
                    )
            return texts

    def get_data(self):
        if self._data is None:
            default_logger.warning(
                f"Dataset '{self.config.name}' (split: {self.split_name}) "
                "is not loaded. Returning None."
            )
        return self._data

    @abstractmethod
    def _build_data(self):
        """Loads dataset-specific data into `self._data`"""
        raise NotImplementedError("Subclasses must implement _build_data()")


class HFDataset(BaseTextDataset):
    @classmethod
    def load(
        cls,
        hf_path: str,
        hf_name: Optional[str] = None,
        root_dir: Optional[str] = None,
        split_name: str = "train",
        text_column: str = "text",
        label_column: Optional[str] = "label",
        seed: int = DEFAULT_SEED,
        **hf_load_kwargs,
    ):
        dataset_name = hf_path.split("/")[-1]
        if hf_name:
            dataset_name = f"{dataset_name}_{hf_name}"

        config = DatasetConfig(
            name=f"hf_{dataset_name}_{split_name.replace('[', '_').replace(']', '')}",
            hf_path=hf_path,
            hf_name=hf_name,
            text_column=text_column,
            label_column=label_column,
        )

        result = cls(
            root_dir=root_dir,
            split_name=split_name,
            config=config,
            seed=seed,
            download=False,
        )

        default_logger.info(
            f"Loading Hugging Face dataset: {hf_path} (name: {hf_name}, split: {split_name})"
        )

        from datasets import load_dataset

        try:
            result._data = load_dataset(
                hf_path, name=hf_name, split=split_name, **(hf_load_kwargs or {})
            )
            default_logger.info(f"Loaded {len(result._data)} samples.")  # type: ignore
        except Exception as e:
            default_logger.error(f"Failed to load Hugging Face dataset: {e}")
            result._data = None
            raise e

        if result._data:
            if text_column not in result._data.column_names:  # type: ignore
                raise ValueError(
                    f"Text column '{text_column}' not found in dataset features: {result._data.column_names}"  # type: ignore
                )
            if label_column and label_column not in result._data.column_names:  # type: ignore
                # If label_column is optional and not found, it's not an error, just means no labels.
                default_logger.warning(
                    f"Specified label column '{label_column}' not found in dataset features: {result._data.column_names}. "  # type: ignore
                    "Proceeding without labels for this column."
                )
                result.config.label_column = None

        # Convert to Jax
        result._data = result._data.with_format("jax")

        return result

    def __getitem__(self, index):
        return self.get_data()[index]


class TextDataset:
    def __init__(
        self,
        base_dataset: BaseTextDataset,
        config: TextDatasetConfig,
    ):
        self.base_dataset = base_dataset
        self.config = config

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitems__(self, indices: List[int]) -> Dict[str, jax.Array]:
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

            text = self.config.transforms(raw_text)  # type: ignore

            batch_texts.append(text)
            batch_labels.append(raw_label)

        outputs = self.config.tokenizer(
            batch_texts, return_array=True, **self.config.encode_config.model_dump()
        )

        if self.config.is_classification:
            labels = jnp.full(shape=len(batch_labels), fill_value=0, dtype=jnp.int64)
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

            return {**outputs, "labels": labels}  # type: ignore
        else:
            return outputs  # type: ignore

    def to_loader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = DEFAULT_SEED,
        **kwargs,
    ):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
            **kwargs,
        )
