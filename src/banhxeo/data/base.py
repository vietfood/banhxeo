import os
import shutil

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, List, Optional

from banhxeo import DEFAULT_SEED
from banhxeo.core.tokenizer import Tokenizer, TokenizerConfig
from banhxeo.core.vocabulary import Vocabulary
from banhxeo.data.config import DatasetConfig, TorchDatasetConfig
from banhxeo.utils.file import check_md5, download_archive, extract_archive
from banhxeo.utils.logging import DEFAULT_LOGGER


class BaseTextDataset(metaclass=ABCMeta):
    """Abstract base class for raw text datasets.

    This class handles common dataset operations like downloading, extracting,
    and providing an interface to access raw text samples. It's designed
    to be subclassed for specific datasets. Raw datasets return text strings,
    which can then be converted to PyTorch Datasets using `to_torch_dataset`.

    Attributes:
        root_path (Path): The root directory for storing datasets.
        dataset_base_path (Path): The specific directory for this dataset
            (e.g., `root_path/datasets/MyDatasetName`).
        config (DatasetConfig): Configuration for the dataset, including name,
            URL, file info, etc.
        split_name (str): The name of the dataset split (e.g., "train", "test").
        seed (int): Random seed, primarily for reproducibility if subsampling
            or shuffling is involved at this stage.
        max_workers (int): Maximum number of workers for parallel processing tasks
            (e.g., file reading in subclasses).
        _data (Any): Internal storage for the loaded dataset samples. Subclasses
            are responsible for populating this.
    """

    def __init__(
        self,
        root_dir: Optional[str],
        split_name: str,
        config: DatasetConfig,
        seed: int,
        download: bool = True,
    ):
        """Initializes the BaseTextDataset.

        Args:
            root_dir: The root directory where datasets are stored. If None,
                defaults to the current working directory.
            split_name: The name of the dataset split (e.g., "train", "test").
            config: A `DatasetConfig` object containing metadata for the dataset.
            seed: A random seed for reproducibility.
            download: If True, attempts to download and extract the dataset
                if it's not already present and `config.url` is provided.
        """
        if not root_dir:
            DEFAULT_LOGGER.warning(
                "root_dir is None or empty. Using current working directory as root."
            )
            self.root_path = Path.cwd()
        else:
            self.root_path = Path(root_dir).resolve()

        self.config = config
        self.dataset_base_path = self.root_path / "datasets" / self.config.name

        if self.config.url or self.config.file_info:  # Dataset might be local only
            self.dataset_base_path.mkdir(parents=True, exist_ok=True)
            DEFAULT_LOGGER.info(
                f"Dataset '{self.config.name}' will be stored in/loaded from: {self.dataset_base_path}"
            )

        self.split_name = split_name
        self.seed = seed
        self.max_workers = min(
            32, (os.cpu_count() or 1) + 4
        )  # Ensure os.cpu_count() not None

        # Data loading logic will be in subclasses
        self._data: Any = None

        if download and self.config.url and self.config.file_info:
            try:
                self._download_and_extract_data()
            except Exception as e:
                DEFAULT_LOGGER.error(
                    f"Failed to download/extract {self.config.name}: {e}. "
                    "Dataset may not be available."
                )

    def _download_and_extract_data(self):
        """Downloads and extracts the dataset archive if specified in config.

        Handles MD5 checksum verification if `config.md5` is provided.
        The archive is downloaded to `dataset_base_path` and extracted there.

        Raises:
            ValueError: If `config.file_info` is missing when download is attempted,
                        or if MD5 checksum fails and user aborts (or if strict
                        checking is enforced).
            Exception: Propagates exceptions from download or extraction utilities.
        """
        if self.config.file_info is None:
            raise

        archive_file_path = (
            self.dataset_base_path
            / f"{self.config.file_info.name}.{self.config.file_info.ext}"
        )

        # Download file if not exist
        if not archive_file_path.is_file():
            DEFAULT_LOGGER.info(
                f"Downloading {self.config.name} dataset to {str(archive_file_path)}..."
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
                f"Extracting file {archive_file_path.name} to {str(self.dataset_base_path)}..."
            )
            try:
                extract_archive(
                    self.config.file_info.ext,
                    archive_file_path,
                    self.dataset_base_path,
                    extracted_data_dir_path,
                )
                DEFAULT_LOGGER.info(
                    f"Successfully extracted to {str(extracted_data_dir_path)}"
                )
            except Exception as e:
                DEFAULT_LOGGER.error(
                    f"Error unpacking archive {str(archive_file_path)}: {e}"
                )
                if extracted_data_dir_path.exists():
                    shutil.rmtree(extracted_data_dir_path, ignore_errors=True)
                raise e

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Relies on `self._data` being populated by the subclass.

        Returns:
            The total number of samples.
        """
        return len(self._data) if self._data is not None else 0

    def __getitem__(self, index: int) -> Any:
        """Retrieves a single raw sample from the dataset.

        The format of the returned sample depends on the specific dataset subclass
        (e.g., a tuple of (text, label), a dictionary, or just text).

        Args:
            index: The index of the sample to retrieve.

        Returns:
            The raw data sample at the given index.

        Raises:
            IndexError: If the index is out of bounds or data is not loaded.
            NotImplementedError: If the subclass does not implement this method.
        """
        if self._data is None:
            raise IndexError(
                f"Dataset {self.config.name} (split: {self.split_name}) not loaded properly."
            )
        raise NotImplementedError("Subclasses must implement __getitem__()")

    def get_all_texts(self) -> List[str]:
        """Extracts all text content from the dataset.

        Iterates through the dataset using `__getitem__` and extracts the text
        portion from each sample. Assumes samples are either strings,
        tuples where the first element is text, or dictionaries with a
        `self.config.text_column`.

        Returns:
            A list of all text strings in the dataset.
        """
        if self._data is None:
            DEFAULT_LOGGER.warning(
                f"Dataset '{self.config.name}' (split: {self.split_name}) "
                "is not loaded. Returning empty list of texts."
            )
            return []

        texts: List[str] = []
        for i in range(len(self)):
            try:
                sample = self[i]
                if isinstance(sample, str):
                    texts.append(sample)
                elif (
                    isinstance(sample, tuple)
                    and len(sample) > 0
                    and isinstance(sample[0], str)
                ):
                    texts.append(sample[0])
                elif isinstance(sample, dict) and self.config.text_column in sample:
                    text_content = sample[self.config.text_column]
                    if isinstance(text_content, str):
                        texts.append(text_content)
                    else:
                        DEFAULT_LOGGER.warning(
                            f"Sample {i} text column '{self.config.text_column}' "
                            f"is not a string: {type(text_content)}. Skipping."
                        )
                else:
                    DEFAULT_LOGGER.warning(
                        f"Cannot extract text from sample {i} of type {type(sample)}. Skipping."
                    )
            except Exception as e:
                DEFAULT_LOGGER.warning(
                    f"Error processing sample {i} in get_all_texts: {e}. Skipping."
                )
        return texts

    def get_data(self) -> Any:
        """Returns the internal data structure holding all samples.

        The type of this structure (`self._data`) depends on the subclass
        (e.g., list, Polars DataFrame, Hugging Face Dataset).

        Returns:
            The raw, loaded dataset.
        """
        if self._data is None:
            DEFAULT_LOGGER.warning(
                f"Dataset '{self.config.name}' (split: {self.split_name}) "
                "is not loaded. Returning None."
            )
        return self._data

    @abstractmethod
    def _build_data(self) -> Any:
        """Loads dataset-specific data into `self._data`.

        This method is responsible for reading files from disk (after potential
        download/extraction) and structuring them into a format accessible by
        `__getitem__`. It should be called by the subclass's `__init__`.

        Returns:
            The loaded data structure (e.g., list, DataFrame).

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement _build_data()")

    def to_torch_dataset(
        self,
        tokenizer: Tokenizer,
        vocab: Vocabulary,
        **kwargs,
    ):
        """Converts this raw text dataset into a `TorchTextDataset`.

        This method sets up the necessary configurations for tokenization,
        numericalization, and transformations to prepare the data for
        PyTorch models.

        Args:
            tokenizer: The `Tokenizer` instance to use.
            vocab: The `Vocabulary` instance for ID mapping.
            **kwargs: Additional configuration options:
                add_special_tokens (bool): Passed to `TokenizerConfig`. Defaults to False.
                max_length (Optional[int]): Passed to `TokenizerConfig`. Defaults to None.
                truncation (bool): Passed to `TokenizerConfig`. Defaults to False.
                padding (Union[bool, str]): Passed to `TokenizerConfig`. Defaults to False.
                is_classification (bool): Whether this is for a classification task.
                    Defaults to False.
                transforms (Union[List["Transforms"], "ComposeTransforms"]): Preprocessing
                    transforms to apply to text before tokenization. Defaults to [].
                label_map (Optional[Dict[str, int]]): Mapping for labels if
                    `is_classification` is True. Defaults to `self.config.label_map`.
                text_column_name (str): Name of the text column. Defaults to
                    `self.config.text_column`.
                label_column_name (Optional[str]): Name of the label column.
                    Defaults to `self.config.label_column`.

        Returns:
            A `TorchTextDataset` instance ready for use with PyTorch DataLoaders.
        """
        from banhxeo.data.torch import TorchTextDataset

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
            label_map=kwargs.get("label_map", self.config.label_map),
            text_column_name=kwargs.get("text_column_name", self.config.text_column),
            label_column_name=kwargs.get("label_column_name", self.config.label_column),
        )

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
        """Loads a dataset from Hugging Face Datasets Hub.

        This classmethod creates an instance of the calling `BaseTextDataset`
        subclass (or `BaseTextDataset` itself if called directly, though subclasses
        are typical) and populates its `_data` attribute with the loaded
        Hugging Face dataset.

        Args:
            hf_path: The path or name of the dataset on Hugging Face Hub (e.g., "imdb", "glue").
            hf_name: The specific configuration or subset of the dataset (e.g., "cola" for "glue").
            root_dir: Root directory for dataset caching (can be managed by HF Datasets).
                If provided, used to construct a unique `DatasetConfig.name`.
            split_name: The dataset split to load (e.g., "train", "test", "validation", "train[:10%]").
            text_column: The name of the column containing text data in the HF dataset.
            label_column: The name of the column containing label data. Can be None.
            seed: Random seed, primarily for dataset config naming consistency.
            **hf_load_kwargs: Additional keyword arguments to pass to
                `datasets.load_dataset()` (e.g., `cache_dir`, `num_proc`).

        Returns:
            An instance of the class this method is called on, with `_data`
            populated by the Hugging Face dataset.

        Raises:
            ImportError: If `datasets` library is not installed.
            ValueError: If specified `text_column` or `label_column` are not
                found in the loaded dataset.
            Exception: Propagates exceptions from `datasets.load_dataset()`.
        """
        dataset_name = hf_path.split("/")[-1]
        if hf_name:
            dataset_name = f"{dataset_name}_{hf_name}"

        config = DatasetConfig(
            name=f"hf_{dataset_name}_{split_name.replace('[', '_').replace(']', '')}",
            hf_path=hf_path,
            hf_name=hf_name,
            text_column=text_column,
            label_column=label_column,
        )

        result = cls(
            root_dir=root_dir,
            split_name=split_name,
            config=config,
            seed=seed,
            download=False,
        )

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
                # If label_column is optional and not found, it's not an error, just means no labels.
                DEFAULT_LOGGER.warning(
                    f"Specified label column '{label_column}' not found in dataset features: {result._data.column_names}. "
                    "Proceeding without labels for this column."
                )
                result.config.label_column = None

        return result

    @staticmethod
    def inspect_huggingface_dataset(hf_path: str, hf_name: Optional[str] = None):
        """Prints information about a Hugging Face dataset.

        Displays the dataset description, features, and available splits.
        Useful for exploring a dataset before loading it.

        Args:
            hf_path: The path or name of the dataset on Hugging Face Hub.
            hf_name: The specific configuration or subset of the dataset.

        Raises:
            ImportError: If `datasets` library is not installed.
        """
        from datasets import (
            get_dataset_split_names,
            load_dataset_builder,
        )

        DEFAULT_LOGGER.info(
            f"Inspecting Hugging Face dataset: {hf_path} (config: {hf_name})"
        )

        ds_builder = load_dataset_builder(hf_path, name=hf_name)
        print("\n--- Dataset Description ---")
        print(
            ds_builder.info.description
            if ds_builder.info.description
            else "No description provided."
        )

        print("\n--- Dataset Features ---")
        print(ds_builder.info.features)

        print("\n--- Available Splits ---")
        try:
            splits = get_dataset_split_names(hf_path, name=hf_name)
            print(splits)
        except Exception as e:
            print(
                f"Could not retrieve split names (dataset might require specific config): {e}"
            )
