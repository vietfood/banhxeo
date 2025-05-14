from pathlib import Path
from typing import Dict, List, Optional, Union

from banhxeo import DEFAULT_SEED
from banhxeo.dataset import DatasetConfig, DatasetFile, DatasetSplit, TextDataset
from banhxeo.dataset.transforms import ComposeTransforms, Transforms

IMDB_CONFIG = DatasetConfig(
    name="IMDB",
    url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    file=DatasetFile(name="aclImdb_v1", ext="tar.gz"),
    md5="7c2ac02c03563afcf9b574c7e56c153a",
    split=DatasetSplit(train=25000, test=25000),
)


class IMDBDataset(TextDataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transforms: Union[List[Transforms], ComposeTransforms] = [],
        seed: int = DEFAULT_SEED,
    ):
        super().__init__(root_dir, split, transforms, IMDB_CONFIG, seed)

    def _read_file_data(self, file_path: str) -> Optional[Dict]:
        """Reads a single file and returns its data, or None on error."""
        try:
            p = Path(file_path)

            name_split = p.stem.split("_")
            if len(name_split) != 2:
                print(f"Warning: Skipping file with unexpected name format: {p.name}")
                return None

            file_id = name_split[0]
            rating = name_split[1]
            label = p.parts[-2]

            with open(file_path, mode="r", encoding="utf-8") as f:
                content = f.read()
                content = self.transforms(content)

            return {
                "id": file_id,
                "rating": int(rating),
                "content": content,
                "label": label,
            }

        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
            return None
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
