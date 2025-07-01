import concurrent.futures
from glob import glob
from pathlib import Path
from typing import Dict, Optional

import polars as pl

from banhxeo import DEFAULT_SEED
from banhxeo.data.base import (
    DatasetConfig,
    DatasetSplit,
    DownloadDatasetFile,
    TextDataset,
)
from banhxeo.utils import progress_bar
from banhxeo.utils.logging import default_logger

IMDB_DATASET_CONFIG = DatasetConfig(
    name="IMDB",
    url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    file_info=DownloadDatasetFile(name="aclImdb", ext="tar.gz"),
    md5="7c2ac02c03563afcf9b574c7e56c153a",
    split=DatasetSplit(train=25000, test=25000),
    text_column="content",
    label_column="label",
    label_map={"pos": 1, "neg": 0},
)


class IMDBDataset(TextDataset):
    def __init__(
        self,
        root_dir: Optional[str] = None,
        split_name: str = "train",  # Renamed from 'split'
        seed: int = DEFAULT_SEED,
    ):
        super().__init__(root_dir, split_name, IMDB_DATASET_CONFIG, seed, download=True)

        # Load data using polars
        self._data = self._build_data()

    def _read_file_data(self, file_path: str) -> Optional[Dict]:
        try:
            p = Path(file_path)

            name_split = p.stem.split("_")
            if len(name_split) != 2:
                default_logger.warning(
                    f"Skipping file with unexpected name format: {p.name}"
                )
                return None

            file_id = name_split[0]
            rating = name_split[1]
            label = p.parts[-2]

            with open(file_path, mode="r", encoding="utf-8") as f:
                content = f.read()

            return {
                "id": file_id,
                "rating": int(rating),
                "content": content,
                "label": label,
            }

        except FileNotFoundError:
            default_logger.warning(f"File not found: {file_path}")
            return None
        except Exception as e:
            default_logger.error(f"Cannot reading file {file_path}: {e}")
            return None

    def _read_data_folder(self, folder):
        results = []
        file_paths = list(glob(f"{folder}/*.txt"))

        if not file_paths:
            default_logger.warning(f"No .txt files found in {folder}")
            return []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_path = {
                executor.submit(self._read_file_data, path): path for path in file_paths
            }
            for future in (
                pbar := progress_bar(
                    concurrent.futures.as_completed(future_to_path),
                    unit="file",
                    unit_scale=True,
                    total=len(file_paths),  # type: ignore
                )
            ):
                path = future_to_path[future]
                pbar.set_description(f"Processing file {Path(path).name}")
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    default_logger.error(
                        f"Error processing file {path} in thread pool: {e}"
                    )

        return results

    def _build_data(self) -> pl.DataFrame:
        all_data = []

        data_path = self.dataset_base_path / self.config.file_info.name  # type: ignore
        split_path = data_path / self.split_name
        for label in ["pos", "neg"]:
            folder_path = split_path / label
            default_logger.info(f"Reading data from: {folder_path}")
            if folder_path.is_dir():
                folder_data = self._read_data_folder(str(folder_path))
                all_data.extend(folder_data)
            else:
                default_logger.debug(f"Directory not found: {folder_path}")

        if not all_data:
            return pl.DataFrame({})

        df = pl.DataFrame(all_data)

        # fmt: off
        df = df.with_columns([
            pl.col("id").cast(pl.String),
            pl.col("rating").cast(pl.Int8), 
            pl.col("content").cast(pl.String),
            pl.col("label").cast(pl.Categorical), 
        ])
        # fmt: on

        return df

    def __getitem__(self, index: int):
        row = self.get_data().row(index, named=True)
        return row
