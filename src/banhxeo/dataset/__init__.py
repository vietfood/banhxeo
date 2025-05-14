import hashlib
import os
import shutil
import warnings
from abc import ABCMeta
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import requests
from torch.utils.data import Dataset
from tqdm import tqdm

from banhxeo.dataset.transforms import ComposeTransforms, Transforms


@dataclass
class DatasetSplit:
    train: int
    test: int
    val: Optional[int] = None


@dataclass
class DatasetFile:
    name: str
    ext: str


@dataclass
class DatasetConfig:
    name: str
    url: str
    file: DatasetFile
    md5: Optional[str] = None
    split: Optional[DatasetSplit] = None


class TextDataset(Dataset, metaclass=ABCMeta):
    """
    - ABC Class for Text Datasets, inherit from pytorch Dataset
    - Cannot be directly instantiated itself.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        transforms: Union[List[Transforms], ComposeTransforms],
        config: DatasetConfig,
        seed: int,
    ):
        """_summary_

        Args:
            root_dir (str): _description_
            split (str): _description_
            config (DatasetConfig): _description_
            seed (int): _description_

        Raises:
            ValueError: _description_
        """
        if not root_dir:
            raise ValueError("Root directory path cannot be empty.")

        self.root_path = Path(root_dir).resolve()
        self.config = config

        self.max_workers = min(32, os.cpu_count() + 4)  # type: ignore

        self.split = split

        self.dataset_base_path = self.root_path / "datasets" / self.config.name

        if not self.dataset_base_path.exists():
            print(
                f"Creating folder for dataset {self.config.name} in {self.dataset_base_path}"
            )
            self.dataset_base_path.mkdir(parents=True, exist_ok=True)

        if isinstance(transforms, ComposeTransforms):
            self.transforms = transforms
        else:
            self.transforms = ComposeTransforms(transforms)

        self.seed = seed

        # Download and extract raw data
        self._download_data()

    def _download_data(self):
        """_summary_

        Raises:
            ValueError: _description_
            ValueError: _description_
        """

        archive_file_path = (
            self.dataset_base_path / f"{self.config.file.name}.{self.config.file.ext}"
        )

        # Download file if not exist
        if not archive_file_path.is_file():
            print(f"Downloading {self.config.name} dataset to {archive_file_path}...")
            try:
                response = requests.get(self.config.url, stream=True)
                response.raise_for_status()  # Raise if error

                with archive_file_path.open(mode="wb") as file:
                    for chunk in tqdm(
                        response.iter_content(chunk_size=1),
                        unit="B",
                        unit_scale=True,
                        desc=f"Downloading {archive_file_path.name}",
                    ):
                        if chunk:
                            file.write(chunk)
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {self.config.url}: {e}")
                if archive_file_path.exists():
                    archive_file_path.unlink()
                raise

        # Check MD5
        if not self.config.md5:
            warnings.warn(
                f"MD5 checksum not provided for dataset {self.config.name}. Extraction is potentially unsafe."
            )
            accept = input("Do you want to continue (yes/no): ").strip().lower()
            if accept != "yes":
                if archive_file_path.is_file():
                    archive_file_path.unlink()
                    print(f"Removed {archive_file_path}")
                raise ValueError(
                    f"Extraction aborted by user for dataset {archive_file_path}."
                )
        else:
            print(f"Verifying MD5 for {archive_file_path.name}...")

            with archive_file_path.open(mode="rb") as f:
                file_hash = hashlib.md5()
                while chunk := f.read(8192):  # Read in chunks for large files
                    file_hash.update(chunk)
                archive_md5 = file_hash.hexdigest()

            if archive_md5 != self.config.md5:
                # remove file
                archive_file_path.unlink()
                print(f"Removed corrupted file {archive_file_path}")
                raise ValueError(
                    f"MD5 mismatch for {archive_file_path}. Expected {self.config.md5}, got {archive_md5}. File may be corrupted."
                )

            print("MD5 checksum verified.")

        extracted_data_dir_path = self.dataset_base_path / self.config.file.name
        if not extracted_data_dir_path.is_dir():
            print(f"Extracting file {archive_file_path} to {self.dataset_base_path}...")
            try:
                shutil.unpack_archive(archive_file_path, self.dataset_base_path)
                print(f"Successfully extracted to {extracted_data_dir_path}")
            except Exception as e:
                print(f"Error unpacking archive {archive_file_path}: {e}")
                if extracted_data_dir_path.exists():
                    # Clean up
                    shutil.rmtree(extracted_data_dir_path, ignore_errors=True)
                raise

    def get_data(self):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()
