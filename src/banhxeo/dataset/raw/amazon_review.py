from banhxeo import DEFAULT_SEED
from banhxeo.dataset.raw import (
    DatasetSplit,
    RawDatasetConfig,
    RawDatasetFile,
    RawTextDataset,
)
from banhxeo.utils.logging import DEFAULT_LOGGER

AMAZON_REVIEW_CONFIG = RawDatasetConfig(
    name="AmazonReview",
    url="https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZVhsUnRWRDhETzA",
    file=RawDatasetFile("amazon_review_full_csv", ext="tar.gz", source="drive"),
    md5="57d28bd5d930e772930baddf36641c7c",
    split=DatasetSplit(train=3000000, test=650000),
)


class AmazonReviewDataset(RawTextDataset):
    def __init__(self, root_dir=None, split="train", seed: int = DEFAULT_SEED):
        super().__init__(root_dir, split, AMAZON_REVIEW_CONFIG, seed)

    def to_torch_dataset(self, tokenizer, vocab, **kwargs):
        raise NotImplementedError()
