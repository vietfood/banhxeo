from typing import Dict, List, Optional, Union

from banhxeo.core.tokenizer import Tokenizer
from banhxeo.core.vocabulary import Vocabulary
from banhxeo.dataset.classification import ClsfDatasetConfig, IMDBDataset
from banhxeo.dataset.raw import IMDBRawDataset, RawTextDataset
from banhxeo.dataset.transforms import ComposeTransforms, Transforms
from banhxeo.utils.logging import DEFAULT_LOGGER


def load_classification_dataset(
    name: str,
    tokenizer: Tokenizer,
    vocab: Vocabulary,
    split: str = "train",
    root_dir: Optional[
        str
    ] = None,  # If is not None, load data from scratch, else use input data
    data: Optional[RawTextDataset] = None,  # If is None, load data from root dir
    label_map: Optional[Dict[str, int]] = None,
    max_len: int = 256,
    add_special_tokens: bool = False,
    padding: Union[bool, str] = (
        False  # False = "do_not_pad", True = "longest", "max_length"
    ),
    truncation: bool = False,  # True = truncate to max_length
    transforms: Union[List[Transforms], ComposeTransforms] = [],
):
    if data is None and root_dir is None:
        raise ValueError(
            "Either 'data' (pre-loaded RawTextDataset) or 'root_dir' must be provided."
        )

    if data and root_dir:
        DEFAULT_LOGGER.warning("Both 'data' and 'root_dir' provided. Using 'data'.")

    config = ClsfDatasetConfig(
        tokenizer=tokenizer,
        vocab=vocab,
        max_len=max_len,
        transforms=transforms,
        padding=padding,
        add_special_tokens=add_special_tokens,
        truncation=truncation,
    )

    dataset_name = name.strip().lower()
    current_data: Optional[RawTextDataset] = data

    if data:
        current_data = data
        if hasattr(current_data, "config") and (
            current_data.config.name.strip().lower() != dataset_name
        ):
            DEFAULT_LOGGER.warning(
                f"Provided input data name '{current_data.config.name}' "
                f"differs from requested dataset_name '{name}'. Proceeding with provided data."
            )
    elif root_dir:  # data is None
        if dataset_name == "imdb":
            current_data = IMDBRawDataset(root_dir=root_dir, split=split)
        else:
            raise ValueError(f"Dataset {name} is not supported yet")

    if dataset_name == "imdb":
        final_label_map = label_map if label_map is not None else {"pos": 1, "neg": 0}
        if not isinstance(current_data, IMDBRawDataset):
            raise TypeError(
                f"Expected IMDBRawDataset for 'imdb', got {type(current_data)}"
            )

        return IMDBDataset(data=current_data, config=config, label_map=final_label_map)
    else:
        raise ValueError(f"Dataset {name} is not supported yet")
