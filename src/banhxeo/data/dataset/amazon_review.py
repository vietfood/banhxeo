import concurrent.futures
from glob import glob
from pathlib import Path
from typing import Dict, Optional

import polars as pl

from banhxeo import DEFAULT_SEED
from banhxeo.data.base import BaseTextDataset
from banhxeo.data.config import DatasetConfig, DatasetSplit, DownloadDatasetFile
from banhxeo.utils import progress_bar
from banhxeo.utils.logging import DEFAULT_LOGGER

AMAZON_REVIEW_CONFIG = DatasetConfig(
    name="AmazonReview",
    url="https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA",
    file_info=DownloadDatasetFile(
        "amazon_review_full_csv", ext="tar.gz", source="drive"
    ),
    md5="57d28bd5d930e772930baddf36641c7c",
    split=DatasetSplit(train=3000000, test=650000),
)
