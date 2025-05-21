import os
import shutil
from abc import ABCMeta
from pathlib import Path
from typing import Any, List, Optional

from banhxeo import DEFAULT_SEED
from banhxeo.core.tokenizer import Tokenizer, TokenizerConfig
from banhxeo.core.vocabulary import Vocabulary
from banhxeo.data.config import DatasetConfig, TorchDatasetConfig
from banhxeo.utils.file import check_md5, download_archive, extract_archive
from banhxeo.utils.logging import DEFAULT_LOGGER


class BaseTextDataset(metaclass=ABCMeta):
    """
    - This is the *raw* dataset, will return text (string) not Tensor, convert to Pytorch dataset by using `to_torch_dataset`.
    - Can load hugging face dataset by using `load_from_huggingface`.
    - ABC Class for all text datasets in banhxeo library. Cannot be directly instantiated itself.
    """

    def __init__(
        self,
        root_dir: Optional[str],
        split_name: str,
        config: DatasetConfig,
        seed: int,
        download: bool = True,
    ):
        if not root_dir:
            DEFAULT_LOGGER.warning(
                "root_dir is None or empty string, use current directory as default"
            )
            self.root_path = Path.cwd()
        else:
            self.root_path = Path(root_dir).resolve()

        self.dataset_base_path = self.root_path / "datasets" / config.name

        # Create base path only if it's a dataset needing download
        if config.url or config.file_info:
            if not self.dataset_base_path.exists():
                DEFAULT_LOGGER.info(
                    f"Creating folder for dataset {config.name} in {self.dataset_base_path.name}"
                )
                self.dataset_base_path.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.split_name = split_name
        self.seed = seed
        self.max_workers = min(32, os.cpu_count() + 4)  # type: ignore

        # Download and extract raw data
        if download and self.config.url and self.config.file_info:
            self._download_and_extract_data()

        # Data loading logic will be in subclasses
        self._data: Any = None

    def _download_and_extract_data(self):
        if self.config.file_info is None:
            raise

        archive_file_path = (
            self.dataset_base_path
            / f"{self.config.file_info.name}.{self.config.file_info.ext}"
        )

        # Download file if not exist
        if not archive_file_path.is_file():
            DEFAULT_LOGGER.info(
                f"Downloading {self.config.name} dataset to {archive_file_path.name}..."
            )
            try:
                download_archive(
                    self.config.file_info.source,
                    self.config.url,  # type: ignore
                    archive_file_path,
                )
            except Exception as e:
                DEFAULT_LOGGER.error(f"Cannot download {self.config.url}: {e}")
                if archive_file_path.exists():
                    archive_file_path.unlink()
                raise e

        # Check MD5 (can raise exception)
        check_md5(self.config.md5, self.config.name, archive_file_path)

        # File extract
        extracted_data_dir_path = self.dataset_base_path / self.config.file_info.name
        if not extracted_data_dir_path.is_dir():
            DEFAULT_LOGGER.info(
                f"Extracting file {archive_file_path.name} to {self.dataset_base_path.name}..."
            )
            try:
                extract_archive(
                    self.config.file_info.ext,
                    archive_file_path,
                    self.dataset_base_path,
                    extracted_data_dir_path,
                )
                DEFAULT_LOGGER.info(
                    f"Successfully extracted to {extracted_data_dir_path.name}"
                )
            except Exception as e:
                DEFAULT_LOGGER.error(
                    f"Error unpacking archive {archive_file_path.name}: {e}"
                )
                if extracted_data_dir_path.exists():
                    shutil.rmtree(extracted_data_dir_path, ignore_errors=True)
                raise e

    def __len__(self) -> int:
        return len(self._data) if self._data is not None else 0

    def __getitem__(self, index: int) -> Any:
        # Should return a raw data sample, e.g., (text, label) or dict (hugging face)
        if self._data is None:
            raise IndexError("Dataset not loaded properly.")
        sample = self._data[index]
        return sample

    def get_all_texts(self) -> List[str]:
        texts = []
        for i in range(len(self)):
            sample = self[i]
            if isinstance(sample, tuple):
                texts.append(sample[0])
            elif isinstance(sample, str):
                texts.append(sample)
            else:  # If sample is a dict (hugging face)
                texts.append(sample[self.config.text_column])
        return texts

    def get_data(self) -> Any:
        return self._data

    def to_torch_dataset(
        self,
        tokenizer: Tokenizer,
        vocab: Vocabulary,
        **kwargs,
    ):
        # Create tokenizer config
        tokenizer_config = TokenizerConfig(
            add_special_tokens=kwargs.get("add_special_tokens", False),
            max_length=kwargs.get("max_length"),
            truncation=kwargs.get("truncation", False),
            padding=kwargs.get("padding", False),
        )

        # Create torch dataset config
        torch_config = TorchDatasetConfig(
            tokenizer=tokenizer,
            tokenizer_config=tokenizer_config,
            vocab=vocab,
            is_classification=kwargs.get("is_classification", False),
            transforms=kwargs.get("transforms", []),
            label_map=kwargs.get("label_map"),
            text_column_name=kwargs.get("text_column_name", "text"),
            label_column_name=kwargs.get("label_column_name", "label"),
        )

        from banhxeo.data.torch import TorchTextDataset

        return TorchTextDataset(self, config=torch_config)

    @classmethod
    def load_from_huggingface(
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
            name=f"hf_{dataset_name}_{split_name.replace('[', '_').replace(']', '')}",  # Unique name
            hf_path=hf_path,
            hf_name=hf_name,
            text_column=text_column,
            label_column=label_column,
        )

        result = cls(root_dir, split_name, config, seed, download=False)

        DEFAULT_LOGGER.info(
            f"Loading Hugging Face dataset: {hf_path} (name: {hf_name}, split: {split_name})"
        )

        from datasets import load_dataset

        try:
            result._data = load_dataset(
                hf_path, name=hf_name, split=split_name, **(hf_load_kwargs or {})
            )
            DEFAULT_LOGGER.info(f"Loaded {len(result._data)} samples.")
        except Exception as e:
            DEFAULT_LOGGER.error(f"Failed to load Hugging Face dataset: {e}")
            result._data = None
            raise

        if result._data:
            if text_column not in result._data.column_names:
                raise ValueError(
                    f"Text column '{text_column}' not found in dataset features: {result._data.column_names}"
                )
            if label_column and label_column not in result._data.column_names:
                raise ValueError(
                    f"Label column '{label_column}' not found in dataset features: {result._data.column_names}"
                )

        return result

    @staticmethod
    def inspect_huggingface_dataset(hf_path: str, hf_name: Optional[str] = None):
        from datasets import (
            get_dataset_split_names,
            load_dataset_builder,
        )

        ds_builder = load_dataset_builder(hf_path, name=hf_name)
        print(ds_builder.info.description)
        print(f"Features: {ds_builder.info.features}")
        print(f"Splits: {str(get_dataset_split_names(hf_path, name=hf_name))}")
