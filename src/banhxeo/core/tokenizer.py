from __future__ import annotations

from abc import ABCMeta
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Union

from pydantic import BaseModel, model_validator
from typing_extensions import Self

from banhxeo.utils.logging import DEFAULT_LOGGER


class TokenizerConfig(BaseModel):
    """Configuration for tokenizers.

    Attributes:
        add_special_tokens: Whether to add special tokens like BOS/EOS.
        max_length: Maximum sequence length. If specified, truncation or padding
            might be applied.
        truncation: Whether to truncate sequences longer than `max_length`.
        padding: Strategy for padding. Can be:
            - False or "do_not_pad": No padding.
            - True or "max_length": Pad to `max_length`.
    """

    add_special_tokens: bool = False
    max_length: Optional[int] = None
    truncation: bool = False
    padding: Union[bool, Literal["do_not_pad", "max_length"]] = (
        False  # False = "do_not_pad", True = "max_length"
    )

    @model_validator(mode="after")
    def check_padding(self) -> Self:
        """Validates padding configuration against max_length."""
        if isinstance(self.padding, str):
            if self.padding == "max_length" and self.max_length is None:
                raise ValueError(
                    "If padding is max_length, you must provide value for max_length parameter"
                )
        elif isinstance(self.padding, bool):
            self.padding = "max_length" if self.padding else "do_not_pad"
        return self


class Tokenizer(metaclass=ABCMeta):
    """Abstract Base Class for all tokenizers.

    Defines the core interface for tokenization, encoding, and managing
    tokenizer-specific data like pre-trained models or vocabularies.
    """

    def __call__(self, text_or_texts: Union[List[str], str], **kwargs):
        """Tokenizes a single text or a list of texts.

        Args:
            text_or_texts: A single string or a list of strings to tokenize.
            **kwargs: Additional arguments passed to the `tokenize` method.

        Returns:
            If input is a single string, returns a list of tokens.
            If input is a list of strings, returns a list of lists of tokens.
        """
        if isinstance(text_or_texts, str):
            return self.tokenize(text_or_texts, **kwargs)
        elif isinstance(text_or_texts, list):
            return [self.tokenize(text, **kwargs) for text in text_or_texts]

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Tokenizes a single string into a list of tokens.

        This method should be implemented by all concrete tokenizer subclasses.
        The base implementation provides a simple regex-based tokenizer as a fallback.

        Args:
            text: The input string to tokenize.
            **kwargs: Subclass-specific tokenization arguments.

        Returns:
            A list of string tokens.
        """
        # Default is regrex tokenizer
        import re

        pattern = r"\w+(?:'\w+)?|[^\w\s]"
        return re.findall(pattern, text)

    def encode(
        self,
        text: str,
        vocab: Vocabulary,  # type: ignore  # noqa: F821
        config: TokenizerConfig,
        **kwargs,
    ) -> Dict[str, List[int]]:
        """Converts a text string into a dictionary of encoded features.

        This base implementation handles tokenization, addition of special tokens,
        truncation, and padding based on the provided configuration.
        Subclasses can override this for more specialized encoding logic.

        Args:
            text: The input string to encode.
            vocab: The vocabulary instance for mapping tokens to IDs.
            config: TokenizerConfig object specifying encoding parameters.
            **kwargs: Additional arguments, potentially passed to the `tokenize` method.

        Returns:
            A dictionary containing:
                - "input_ids": List of token IDs.
                - "attention_mask": List of 0s and 1s indicating padding.

        Raises:
            ValueError: If padding is 'max_length' but `config.max_length` is not set.
        """
        # Tokenize text
        tokens = self.tokenize(text, **kwargs)  # kwargs might be passed to tokenize

        # Add special tokens
        if config.add_special_tokens:
            bos_tokens = vocab.bos_toks
            sep_tokens = vocab.sep_toks

            # Truncate with special tokens
            if config.truncation and config.max_length is not None:
                max_text_tokens = config.max_length - len(bos_tokens) - len(sep_tokens)
                if max_text_tokens < 0:
                    DEFAULT_LOGGER.warning(
                        f"max_length ({config.max_length}) is too small for special tokens. "
                        "Resulting sequence might be only special tokens or empty."
                    )
                    tokens = []
                else:
                    tokens = tokens[:max_text_tokens]
                tokens = bos_tokens + tokens + sep_tokens
        # Truncate without special tokens
        elif (
            config.truncation
            and config.max_length is not None
            and len(tokens) > config.max_length
        ):
            tokens = tokens[: config.max_length]

        input_ids = vocab.tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        output = {"input_ids": input_ids, "attention_mask": attention_mask}

        if config.padding == "max_length":
            pad_len = config.max_length - len(input_ids)  # type: ignore
            if pad_len > 0:
                input_ids += [vocab.pad_id] * pad_len
                attention_mask += [0] * pad_len
            elif (
                pad_len < 0
            ):  # Sequence is longer than max_length after adding special tokens
                # This implies truncation didn't fully reduce it, or max_length is very small
                DEFAULT_LOGGER.warning(
                    f"Sequence length ({len(input_ids)}) is greater than max_length ({config.max_length}) "
                    "even after potential truncation. Output will be truncated to max_length."
                )
            output["input_ids"] = input_ids
            output["attention_mask"] = attention_mask

        return output

    def batch_encode(
        self,
        texts: List[str],
        vocab: Vocabulary,  # noqa: F821 # type: ignore
        config: TokenizerConfig,
        **kwargs,
    ) -> List[Dict[str, List[int]]]:
        """Encodes a batch of text strings.

        Args:
            texts: A list of strings to encode.
            vocab: The vocabulary instance.
            config: TokenizerConfig object.
            **kwargs: Additional arguments for the `encode` method.

        Returns:
            A list of dictionaries, where each dictionary is the output of `encode`
            for the corresponding text.
        """
        return [self.encode(text, vocab, config, **kwargs) for text in texts]

    def train_from_iterator(
        self,
        iterator: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        special_tokens: List[str] = ["<pad>", "<unk>", "<bos>", "<eos>"],
        **kwargs,
    ) -> None:
        """Trains the tokenizer from an iterator of texts.

        This is primarily for tokenizers that learn a vocabulary or merges,
        such as BPE or WordPiece. Simpler tokenizers might implement this
        as a no-operation.

        Args:
            iterator: An iterable yielding text strings.
            vocab_size: The desired vocabulary size.
            min_frequency: The minimum frequency for a token to be included.
            special_tokens: A list of special tokens to include.
            **kwargs: Tokenizer-specific training arguments.

        Raises:
            NotImplementedError: If the tokenizer does not support training.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support training from an iterator."
        )

    # --- Saving and Loading (Essential for trained/complex tokenizers) ---

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        """Saves the tokenizer's state to a directory.

        This should save any learned vocabulary, merges, or configuration
        necessary to reload the tokenizer.

        Args:
            save_directory: Path to the directory where the tokenizer will be saved.
            **kwargs: Additional saving arguments.

        Raises:
            NotImplementedError: If saving is not implemented.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support save_pretrained."
        )

    @classmethod
    def from_pretrained(cls, load_directory: Union[str, Path], **kwargs) -> "Tokenizer":
        """Loads a tokenizer from a previously saved directory.

        Args:
            load_directory: Path to the directory from which to load.
            **kwargs: Additional loading arguments.

        Returns:
            An instance of the tokenizer.

        Raises:
            NotImplementedError: If loading is not implemented.
        """
        raise NotImplementedError(f"{cls.__name__} does not support from_pretrained.")


class NLTKTokenizer(Tokenizer):
    """A tokenizer that uses NLTK's TreebankWordTokenizer.

    Falls back to a regex-based tokenizer if NLTK is not installed.
    """

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Tokenizes text using NLTK's TreebankWordTokenizer.

        Args:
            text: The input string.
            **kwargs: Ignored (NLTK tokenizer doesn't take extra args here).

        Returns:
            A list of tokens.
        """
        try:
            from nltk.tokenize.treebank import TreebankWordTokenizer

            t = TreebankWordTokenizer()
            return t.tokenize(text)

        except ImportError:
            DEFAULT_LOGGER.warning(
                "NLTK not found. Falling back to regex-based tokenizer. "
                "Install NLTK for TreebankWordTokenizer: `pip install nltk`"
            )
            return super().tokenize(text, **kwargs)

    def detokenize(self, tokens: List[str], **kwargs) -> str:
        """Detokenizes a list of tokens using NLTK's TreebankWordDetokenizer.

        Args:
            tokens: A list of string tokens.
            **kwargs: Ignored.

        Returns:
            The detokenized string.
        """
        try:
            from nltk.tokenize.treebank import TreebankWordDetokenizer

            detokenizer = TreebankWordDetokenizer()
            return detokenizer.detokenize(tokens)
        except ImportError:
            DEFAULT_LOGGER.warning(
                "NLTK not found. Falling back to space-joining detokenizer. "
                "Install NLTK for TreebankWordDetokenizer: `pip install nltk`"
            )
            return " ".join(tokens)
