import concurrent.futures
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

from banhxeo import DEFAULT_SEED
from banhxeo.dataset.raw import (
    DatasetSplit,
    RawDatasetConfig,
    RawDatasetFile,
    RawTextDataset,
)
from banhxeo.utils.logging import DEFAULT_LOGGER

WIKITEXT103_CONFIG = RawDatasetConfig(
    name="Wikitext103",
    url="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip",
    file=RawDatasetFile("wikitext-103-v1", ext="zip"),
    md5="9ddaacaf6af0710eda8c456decff7832",
    split=DatasetSplit(train=1801350, val=3760, test=4358),
)


class WikiText103(RawTextDataset):
    def __init__(self, root_dir=None, split="train", seed: int = DEFAULT_SEED):
        super().__init__(root_dir, split, WIKITEXT103_CONFIG, seed)
