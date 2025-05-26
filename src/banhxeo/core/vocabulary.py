import json

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, computed_field, field_validator

from banhxeo.core.tokenizer import Tokenizer
from banhxeo.utils import progress_bar
from banhxeo.utils.logging import DEFAULT_LOGGER


class VocabConfig(BaseModel):
    """Configuration for vocabulary settings, especially special tokens.

    This configuration defines the string representations for various special
    tokens and the minimum frequency for corpus tokens. The order in the
    `special_tokens` property attempts to follow Hugging Face conventions
    to facilitate consistent ID assignment (e.g., PAD=0, UNK=1).

    Attributes:
        min_freq: Minimum frequency for a token from the corpus to be included
            in the vocabulary. Defaults to 1.
        pad_tok: Padding token string. Crucial for sequence padding.
            Defaults to "<PAD>".
        unk_tok: Unknown token string, for out-of-vocabulary words.
            Defaults to "<UNK>".
        bos_tok: Beginning-of-sentence token string. Often used by generative models.
            Defaults to "<BOS>".
        sep_tok: Separator token string. Used to separate sequences (e.g., in BERT
            for sentence pairs) or as an end-of-sentence token.
            Defaults to "<SEP>".
        mask_tok: Mask token string (e.g., for Masked Language Modeling like BERT).
            Optional. Defaults to None.
        cls_tok: Classification token string (e.g., the first token in BERT sequences
            for classification tasks). Optional. Defaults to None.
        resv_tok: Reserved token string for future use or custom purposes. Optional.
            Defaults to None.
    """

    min_freq: int = 1
    # Define tokens that are almost always present and have conventional low IDs
    pad_tok: str = "<PAD>"
    unk_tok: str = "<UNK>"

    # Tokens common in many models
    bos_tok: str = "<BOS>"  # Beginning of Sentence/Sequence
    sep_tok: str = "<SEP>"  # Separator / End of Sentence/Sequence

    # Model-specific or optional tokens
    cls_tok: Optional[str] = None  # Often BERT-specific, sometimes same as BOS
    mask_tok: Optional[str] = None  # BERT-specific
    resv_tok: Optional[str] = None  # For any reserved tokens

    @field_validator("min_freq", mode="before")
    @classmethod
    def check_min_freq_positive(cls, value: int) -> int:  # Renamed for clarity
        """Validates that min_freq is at least 1."""
        if value < 1:
            raise ValueError("Minimum frequency (min_freq) must be at least 1.")
        return value

    @computed_field
    @property
    def special_tokens(self) -> List[str]:
        """Returns a list of all configured special tokens in a conventional order.

        This order is designed to facilitate common ID assignments when building
        a vocabulary sequentially (e.g., PAD token getting ID 0, UNK token ID 1).
        The actual IDs depend on the `Vocabulary.build()` process.

        Conventional Order (if token is defined):
        1. `pad_tok` (aims for ID 0)
        2. `unk_tok` (aims for ID 1)
        3. `cls_tok` (if defined, common for BERT-like models)
        4. `sep_tok` (common separator or EOS for many models)
        5. `mask_tok` (if defined, for MLM)
        6. `bos_tok` (if defined and distinct, for generative start)
        7. `resv_tok` (if defined)

        Returns:
            A list of special token strings, excluding any that are None.
        """
        ordered_tokens = []
        # Critical tokens first for low IDs
        if self.pad_tok:
            ordered_tokens.append(self.pad_tok)
        if self.unk_tok:
            ordered_tokens.append(self.unk_tok)

        # Common structural tokens
        if self.cls_tok:
            ordered_tokens.append(self.cls_tok)
        if self.sep_tok:
            ordered_tokens.append(self.sep_tok)  # Often also EOS
        if self.mask_tok:
            ordered_tokens.append(self.mask_tok)

        # BOS might be distinct from CLS in some contexts (e.g. pure generative)
        # If CLS is defined and used as BOS, this might be redundant or for different use.
        if (
            self.bos_tok and self.bos_tok not in ordered_tokens
        ):  # Avoid adding if already present (e.g. if cls_tok == bos_tok)
            ordered_tokens.append(self.bos_tok)

        if self.resv_tok and self.resv_tok not in ordered_tokens:
            ordered_tokens.append(self.resv_tok)

        # Deduplicate while preserving order (in case user sets, e.g., bos_tok = sep_tok)
        # Though Pydantic models usually enforce distinctness if fields are distinct.
        # This is more if a user sets multiple config attributes to the *same string value*.
        final_tokens = []
        seen = set()
        for token in ordered_tokens:
            if token not in seen:
                final_tokens.append(token)
                seen.add(token)
        return final_tokens

    def special_token_idx(self, token: str) -> int:
        """Gets the predefined index of a special token within the `special_tokens` list.

        Note: This implies a fixed ordering of special tokens. The actual ID in a
        `Vocabulary` instance depends on how it was built. Use `Vocabulary.token_to_id[token]`
        for the actual ID. This method is more about the config's view.

        Args:
            token: The special token string.

        Returns:
            The index of the token in the `special_tokens` list.

        Raises:
            ValueError: If the token is not found in the `special_tokens` list.
        """
        try:
            return self.special_tokens.index(token)
        except ValueError:
            raise ValueError(
                f"Token '{token}' is not a configured special token in this VocabConfig."
            )


DEFAULT_VOCAB_CONFIG = VocabConfig()


class Vocabulary:
    """Manages token-to-ID mapping, special tokens, and vocabulary building.

    Provides functionalities to build a vocabulary from a corpus,
    load/save it, and convert between tokens and their numerical IDs.
    The assignment of IDs to special tokens during `build()` is guided by
    the order in `VocabConfig.special_tokens`.

    Attributes:
        vocab_config (VocabConfig): Configuration for special tokens and vocabulary
            building parameters like minimum frequency.
        tokenizer (Optional[Tokenizer]): The tokenizer associated with this vocabulary,
            used during the build process.
        _idx_to_token (List[str]): A list mapping token IDs to tokens.
        _token_to_idx (Dict[str, int]): A dictionary mapping tokens to token IDs.
        _word_counts (Optional[Dict[str, int]]): Raw token counts from the corpus
            used to build the vocabulary (populated after `build()` is called).
    """

    def __init__(self, vocab_config: Optional[VocabConfig] = None):
        """Initializes the Vocabulary.

        Args:
            vocab_config: Configuration for the vocabulary. If None,
                uses `DEFAULT_VOCAB_CONFIG`.
        """
        self.vocab_config = vocab_config if vocab_config else DEFAULT_VOCAB_CONFIG

        self.tokenizer = None
        self._idx_to_token = list()
        self._token_to_idx = dict()
        self._word_counts = None

    @classmethod
    def load(cls, path: Union[Path, str], tokenizer: Tokenizer):
        """Loads a vocabulary from a JSON file.

        The JSON file should contain the vocabulary config, the tokenizer class name
        used for building, and the token-to-ID mappings.

        Args:
            path: Path to the vocabulary JSON file.
            tokenizer: The tokenizer instance that was used or is compatible with
                this vocabulary. Its class name will be checked against the saved one.

        Returns:
            An instance of Vocabulary loaded with data from the file.

        Raises:
            ValueError: If the path is invalid, or if the provided tokenizer
                class does not match the one saved with the vocabulary.
            FileNotFoundError: If the vocabulary file does not exist.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Vocabulary file not found at: {path}")

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

            loaded_config = VocabConfig.model_validate(data["config"])
            vocab = cls(loaded_config)
            vocab._token_to_idx = data["token_to_idx"]
            vocab._idx_to_token = data["idx_to_token"]

            saved_tokenizer_name = data.get("tokenizer", "Unknown")
            if tokenizer.__class__.__name__ != saved_tokenizer_name:
                DEFAULT_LOGGER.warning(
                    f"Tokenizer mismatch: Loaded vocabulary was built with "
                    f"'{saved_tokenizer_name}', but provided tokenizer is "
                    f"'{tokenizer.__class__.__name__}'. Ensure compatibility."
                )
            vocab.tokenizer = tokenizer
        DEFAULT_LOGGER.info(f"Vocabulary loaded from {path}")
        return vocab

    @classmethod
    def build(
        cls,
        corpus: List[str],  # Expect a list of string
        tokenizer: Tokenizer,
        **kwargs,
    ):
        """Builds a vocabulary from a list of text sentences.

        This method tokenizes the corpus, counts token frequencies,
        adds special tokens, and then adds corpus tokens that meet the
        minimum frequency requirement.

        Args:
            corpus: A list of strings, where each string is a sentence or document.
            tokenizer: The tokenizer instance to use for tokenizing the corpus.
            **kwargs: Additional arguments to override `VocabConfig` defaults.
                Supported keys: `min_freq`, `pad_tok`, `unk_tok`, `sep_tok`,
                `bos_tok`, `mask_tok`, `cls_tok`, `resv_tok`.

        Returns:
            A new Vocabulary instance built from the corpus.
        """
        """Builds a vocabulary from a list of text sentences.

        This method tokenizes the corpus, counts token frequencies,
        adds special tokens (in the order defined by `VocabConfig.special_tokens`
        to facilitate conventional ID assignment), and then adds corpus tokens
        that meet the minimum frequency requirement.

        Args:
            corpus: A list of strings, where each string is a sentence or document.
            tokenizer: The tokenizer instance to use for tokenizing the corpus.
            **kwargs: Arguments to override `VocabConfig` defaults when creating
                the vocabulary's configuration (e.g., `min_freq`, `pad_tok`).

        Returns:
            A new Vocabulary instance built from the corpus.
        """
        current_config_dict = DEFAULT_VOCAB_CONFIG.model_dump()
        # Update with any kwargs passed that are valid VocabConfig fields
        for k, v in kwargs.items():
            if k in VocabConfig.model_fields:
                current_config_dict[k] = v
            else:
                DEFAULT_LOGGER.warning(
                    f"Ignoring unknown kwarg '{k}' during VocabConfig creation for Vocabulary.build()."
                )

        vocab_config_instance = VocabConfig(**current_config_dict)
        vocab = cls(vocab_config_instance)
        vocab.tokenizer = tokenizer

        # Add special tokens first, in the order defined by the config's property.
        for token_str in vocab.vocab_config.special_tokens:
            if token_str and token_str not in vocab._token_to_idx:
                idx = len(vocab._idx_to_token)
                vocab._idx_to_token.append(token_str)
                vocab._token_to_idx[token_str] = idx

        token_counts = defaultdict(int)

        DEFAULT_LOGGER.info("Tokenizing corpus and counting token frequencies...")
        for idx_sample, text_sample in progress_bar(
            enumerate(corpus),
            unit=" sentence",
            unit_scale=True,
            total=len(corpus),
            desc="Building vocabulary",
        ):
            try:
                tokens = tokenizer(text_sample)
                for token in tokens:
                    token_counts[token] += 1
            except Exception as e:
                DEFAULT_LOGGER.warning(
                    f"Tokenizer failed on sample {idx_sample} ('{text_sample[:50]}...'): {e}. Skipping."
                )

        vocab._word_counts = dict(token_counts)  # Store raw counts

        # Add tokens from corpus, sorted by frequency (descending)
        # This ensures more frequent tokens get lower IDs (after special tokens)
        sorted_tokens_by_freq = sorted(
            token_counts.items(), key=lambda item: item[1], reverse=True
        )

        min_freq = vocab.vocab_config.min_freq
        added_corpus_tokens = 0
        for token, count in sorted_tokens_by_freq:
            if (
                count >= min_freq and token not in vocab._token_to_idx
            ):  # Check it's not already a special token
                idx = len(vocab._idx_to_token)
                vocab._idx_to_token.append(token)
                vocab._token_to_idx[token] = idx
                added_corpus_tokens += 1

        DEFAULT_LOGGER.info(
            f"Vocabulary built: {len(vocab._idx_to_token)} unique tokens "
            f"({len(vocab.vocab_config.special_tokens)} special, {added_corpus_tokens} from corpus) "
            f"from {len(corpus)} sentences. Min frequency: {min_freq}."
        )

        return vocab

    def _check_built(self) -> None:
        """Raises ValueError if the vocabulary is not yet built."""
        if not self._idx_to_token or not self._token_to_idx:
            raise ValueError("Vocabulary not built yet. Call build() or load() first.")

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a single token to its ID.

        Internal method. Users should generally use `tokens_to_ids`.
        If the token is not in the vocabulary, returns the ID of the UNK token.

        Args:
            token: The token string.

        Returns:
            The integer ID of the token.

        Raises:
            ValueError: If the vocabulary has not been built.
        """
        self._check_built()
        return self._token_to_idx.get(token, self.unk_id)

    def _convert_id_to_token(self, idx: int) -> str:
        """Converts a single ID to its token string.

        Internal method. Users should generally use `ids_to_tokens`.

        Args:
            idx: The integer ID.

        Returns:
            The token string.

        Raises:
            ValueError: If the vocabulary has not been built.
            IndexError: If the ID is out of the vocabulary range.
        """
        self._check_built()
        if 0 <= idx < len(self._idx_to_token):
            return self._idx_to_token[idx]
        else:
            DEFAULT_LOGGER.warning(
                f"ID {idx} is out of vocabulary range (0-{len(self) - 1}). "
                f"Returning UNK token '{self.unk_tok}' instead."
            )
            return (
                self.unk_tok
            )  # Or raise IndexError as before if strictness is preferred

    def save(self, path: Union[str, Path]) -> None:
        """Saves the vocabulary to a JSON file.

        The saved file includes the vocabulary configuration, the name of the
        tokenizer class used for building (if any), the token-to-ID mappings,
        and raw word counts.

        Args:
            path: The file path where the vocabulary will be saved.

        Raises:
            ValueError: If the vocabulary has not been built yet (is empty).
            IOError: If there's an issue writing the file.
        """
        self._check_built()

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data_to_save = {
            "config": self.vocab_config.model_dump(mode="json"),
            "tokenizer": self.tokenizer.__class__.__name__
            if self.tokenizer
            else "Unknown",
            "token_to_idx": self._token_to_idx,
            "idx_to_token": self._idx_to_token,
            "_word_counts": self._word_counts,  # Save raw counts
        }

        DEFAULT_LOGGER.info(f"Saving vocabulary to {save_path}...")
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
            DEFAULT_LOGGER.info(f"Vocabulary successfully saved to {save_path}.")
        except IOError as e:
            DEFAULT_LOGGER.error(f"Failed to save vocabulary to {save_path}: {e}")
            raise

    @property
    def vocab_size(self) -> int:
        """Returns the total number of unique tokens in the vocabulary (including special tokens)."""
        # self._check_built() # get_vocab will do this
        return len(self.get_vocab())

    def get_vocab(self) -> List[str]:  # Changed return type from List[int] to List[str]
        """Returns the list of all tokens in the vocabulary, ordered by ID.

        Returns:
            A list of all token strings in the vocabulary.

        Raises:
            ValueError: If the vocabulary has not been built.
        """
        self._check_built()
        return self._idx_to_token  # This is already the list of tokens by ID

    @property
    def unk_id(self) -> int:
        """Returns the ID of the unknown token (<UNK>)."""
        # self._check_built() # _convert_token_to_id will do this
        return self._convert_token_to_id(self.vocab_config.unk_tok)

    @property
    def pad_id(self) -> int:
        """Returns the ID of the padding token (<PAD>)."""
        return self._convert_token_to_id(self.vocab_config.pad_tok)

    @property
    def bos_id(self) -> int:
        """Returns the ID of the beginning-of-sentence token (<BOS>)."""
        return self._convert_token_to_id(self.vocab_config.bos_tok)

    @property
    def sep_id(self) -> int:
        """Returns the ID of the separator/end-of-sentence token (<SEP>)."""
        return self._convert_token_to_id(self.vocab_config.sep_tok)

    @property
    def sep(self) -> int:  # This was an int property, probably an alias for sep_id
        """Alias for `sep_id`."""
        return self.sep_id

    @property
    def unk_tok(self) -> str:
        """Returns the unknown token string (e.g., "<UNK>")."""
        return self.vocab_config.unk_tok

    @property
    def bos_toks(self) -> List[str]:  # Name implies plural, but was returning single
        """Returns a list containing the beginning-of-sentence token string.

        Typically used when prepending tokens to a sequence.
        """
        return [self.vocab_config.bos_tok] if self.vocab_config.bos_tok else []

    @property
    def sep_toks(self) -> List[str]:  # Name implies plural, but was returning single
        """Returns a list containing the separator/end-of-sentence token string.

        Typically used when appending tokens to a sequence.
        """
        return [self.vocab_config.sep_tok] if self.vocab_config.sep_tok else []

    @property
    def token_to_idx(self) -> Dict[str, int]:
        """Returns the dictionary mapping tokens to their IDs.

        Raises:
            ValueError: If the vocabulary has not been built.
        """
        self._check_built()
        return self._token_to_idx

    @property
    def idx_to_token(self) -> List[str]:
        """Returns the list mapping IDs to their token strings.

        Raises:
            ValueError: If the vocabulary has not been built.
        """
        self._check_built()
        return self._idx_to_token

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Converts a list of token strings to a list of their corresponding IDs.

        Unknown tokens are mapped to the `unk_id`.

        Args:
            tokens: A list of token strings.

        Returns:
            A list of integer IDs.
        """
        # self._check_built() # _convert_token_to_id will do this
        return [self._convert_token_to_id(token) for token in tokens]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Converts a list of token IDs to their corresponding token strings.

        IDs outside the vocabulary range might be mapped to the `unk_tok` or raise an error,
        depending on the internal `_convert_id_to_token` implementation.

        Args:
            ids: A list of integer IDs.

        Returns:
            A list of token strings.
        """
        # self._check_built() # _convert_id_to_token will do this
        return [
            self._convert_id_to_token(id_val) for id_val in ids
        ]  # Renamed id to id_val

    def __len__(self) -> int:
        """Returns the total number of unique tokens in the vocabulary.

        Equivalent to `self.vocab_size`.
        """
        # self._check_built() # vocab_size will do this
        return self.vocab_size

    def get_word_counts(self) -> Dict[str, int]:
        """Returns the raw counts of tokens observed in the corpus during `build()`.

        If the vocabulary was loaded or not built from a corpus, this might be
        empty or reflect counts from the original build.

        Returns:
            A dictionary mapping token strings to their raw frequencies.

        Raises:
            ValueError: If the vocabulary has not been built (and thus `_word_counts`
                        is not populated).
        """
        self._check_built()  # Ensures _word_counts would have been populated if built
        if self._word_counts is None:
            # This case might happen if loaded from an old vocab file without _word_counts
            DEFAULT_LOGGER.warning(
                "Word counts are not available for this vocabulary instance."
            )
            return {}
        return self._word_counts
