import os
import shutil
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import polars as pl

from banhxeo.utils.file import check_md5, download_archive, extract_archive
from banhxeo.utils.logging import DEFAULT_LOGGER


@dataclass
class DatasetSplit:
    train: int
    test: int
    val: Optional[int] = None


@dataclass
class RawDatasetFile:
    name: str
    ext: str
    source: Optional[str] = None


@dataclass
class RawDatasetConfig:
    name: str
    url: str
    file: RawDatasetFile
    md5: Optional[str] = None
    split: Optional[DatasetSplit] = None


class RawTextDataset(metaclass=ABCMeta):
    """
    - This is the *raw* dataset, will return text (string) not Tensor
    - ABC Class for Text Datasets
    - Cannot be directly instantiated itself.
    """

    def __init__(
        self,
        root_dir: Optional[str],
        split: str,
        config: RawDatasetConfig,
        seed: int,
    ):
        if not root_dir:
            DEFAULT_LOGGER.warning(
                "root_dir is None or empty string, use current directory as default"
            )
            self.root_path = Path.cwd()
        else:
            self.root_path = Path(root_dir).resolve()

        self.config = config

        self.max_workers = min(32, os.cpu_count() + 4)  # type: ignore

        self.split = split

        self.dataset_base_path = self.root_path / "datasets" / self.config.name

        if not self.dataset_base_path.exists():
            DEFAULT_LOGGER.info(
                f"Creating folder for dataset {self.config.name} in {self.dataset_base_path.name}"
            )
            self.dataset_base_path.mkdir(parents=True, exist_ok=True)

        self.seed = seed

        # Download and extract raw data
        self._download_and_extract_data()

    def _download_and_extract_data(self):
        archive_file_path = (
            self.dataset_base_path / f"{self.config.file.name}.{self.config.file.ext}"
        )

        # Download file if not exist
        if not archive_file_path.is_file():
            DEFAULT_LOGGER.info(
                f"Downloading {self.config.name} dataset to {archive_file_path.name}..."
            )
            try:
                download_archive(
                    self.config.file.source, self.config.url, archive_file_path
                )
            except Exception as e:
                DEFAULT_LOGGER.error(f"Cannot download {self.config.url}: {e}")
                if archive_file_path.exists():
                    archive_file_path.unlink()
                raise e

        # Check MD5 (can raise exception)
        check_md5(self.config.md5, self.config.name, archive_file_path)

        # File extract
        extracted_data_dir_path = self.dataset_base_path / self.config.file.name
        if not extracted_data_dir_path.is_dir():
            DEFAULT_LOGGER.info(
                f"Extracting file {archive_file_path.name} to {self.dataset_base_path.name}..."
            )
            try:
                extract_archive(
                    self.config.file.ext,
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

    @property
    def data(self) -> pl.DataFrame:
        raise NotImplementedError()

    @property
    def text_data(self) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def to_torch_dataset(self, tokenizer, vocab, **kwargs):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()
