from typing import Optional

import polars as pl

from banhxeo import DEFAULT_SEED
from banhxeo.data.base import (
    DatasetConfig,
    DatasetSplit,
    DownloadDatasetFile,
    TextDataset,
)

AMAZON_REVIEW_FULL_CONFIG = DatasetConfig(
    name="AmazonReview",
    url="https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA",
    file_info=DownloadDatasetFile(
        "amazon_review_full_csv", ext="tar.gz", source="drive"
    ),
    md5="57d28bd5d930e772930baddf36641c7c",
    split=DatasetSplit(train=3000000, test=650000),
    text_column="Review",
    label_column="Polarity",
)


class AmazonReviewFullDataset(TextDataset):
    def __init__(
        self,
        root_dir: Optional[str] = None,
        split_name: str = "train",
        seed: int = DEFAULT_SEED,
    ):
        super().__init__(
            root_dir, split_name, AMAZON_REVIEW_FULL_CONFIG, seed, download=True
        )

        self._data = self._build_data()

    def _build_data(self):
        # https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews
        columns = ["Polarity", "Title", "Review"]
        csv_path = self.dataset_base_path / self.config.file_info.name  # type: ignore
        if self.split_name == "train":
            df = pl.read_csv(
                csv_path / "train.csv", has_header=False, new_columns=columns
            )
        elif self.split_name == "test":
            df = pl.read_csv(
                csv_path / "test.csv", has_header=False, new_columns=columns
            )
        else:
            raise ValueError("AmazonReviewDataset supports train and test splits only")
        return df

    def __getitem__(self, index: int):
        row = self.get_data().row(index, named=True)
        return row
