from __future__ import annotations

from abc import ABCMeta
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Union

from pydantic import BaseModel, model_validator
from typing_extensions import Self

from banhxeo.utils.logging import DEFAULT_LOGGER


class TokenizerConfig(BaseModel):
    add_special_tokens: bool = False
    max_length: Optional[int] = None
    truncation: bool = False
    padding: Union[bool, Literal["do_not_pad", "max_length"]] = (
        False  # False = "do_not_pad", True = "max_length"
    )

    @model_validator(mode="after")
    def check_padding(self) -> Self:
        if isinstance(self.padding, str):
            if self.padding == "max_length" and self.max_length is None:
                raise ValueError(
                    "If padding is max_length, you must provide value for max_length parameter"
                )
        elif isinstance(self.padding, bool):
            self.padding = "max_length" if self.padding else "do_not_pad"
        return self


class Tokenizer(metaclass=ABCMeta):
    """
    Abstract Base Class for all tokenizers.
    Defines the core interface for tokenization and related functionalities.
    """

    def __call__(self, text_or_texts: Union[List[str], str], **kwargs):
        if isinstance(text_or_texts, str):
            return self.tokenize(text_or_texts, **kwargs)
        elif isinstance(text_or_texts, list):
            return [self.tokenize(text, **kwargs) for text in text_or_texts]

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """This is base implementation of Tokenizer tokenize, each child class should implement their own version"""
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
        """This is base implementation of Tokenizer encode, each child class should implement their own version"""

        # Tokenize text
        tokens = self.tokenize(text, **kwargs)  # kwargs might be passed to tokenize

        # Add special tokens
        if config.add_special_tokens:
            bos_tokens = vocab.bos_toks
            sep_tokens = vocab.sep_toks

            # Truncate with special tokens
            if config.truncation and config.max_length is not None:
                max_text_tokens = config.max_length - len(bos_tokens) - len(sep_tokens)
                if max_text_tokens < 0:  # max_length too small for special tokens
                    tokens = []  # Or raise error, or just use special tokens
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
        """This is base implementation of Tokenizer batch_encode, each child class should implement their own version"""
        return [self.encode(text, vocab, config, **kwargs) for text in texts]

    def train_from_iterator(
        self,
        iterator: Iterable[str],
        vocab_size: int,
        min_frequency: int = 2,
        special_tokens: List[str] = ["<pad>", "<unk>", "<bos>", "<eos>"],
        **kwargs,
    ) -> None:
        """
        Trains the tokenizer (e.g., learns merges for BPE, or word pieces) from an iterator of texts.
        This is mainly for tokenizers that have a trainable vocabulary component (like BPE, WordPiece).
        Simple tokenizers like WordTokenizer or CharTokenizer might not need this or implement it as a no-op.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support training from an iterator."
        )

    # --- Saving and Loading (Essential for trained/complex tokenizers) ---

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        """
        Saves the tokenizer's configuration and any learned vocabulary/merges
        to a directory, so it can be reloaded.
        """
        ...
        raise NotImplementedError()

    @classmethod
    def from_pretrained(cls, load_directory: Union[str, Path], **kwargs) -> Tokenizer:
        """
        Loads a tokenizer from a previously saved directory.
        """
        raise NotImplementedError("Waiting to be implemented")


class NLTKTokenizer(Tokenizer):
    def tokenize(self, text: str, **kwargs) -> List[str]:
        try:
            from nltk.tokenize.treebank import TreebankWordTokenizer

            t = TreebankWordTokenizer()
            return t.tokenize(text)

        except ImportError:
            DEFAULT_LOGGER.warning(
                "You need to install `nltk` to use `NLTKTokenizer`. Fallback to Regrex Tokenizer"
            )
            return super().tokenize(text)

    def detokenize(self, tokens: List[str], **kwargs) -> str:
        try:
            from nltk.tokenize.treebank import TreebankWordDetokenizer

            d = TreebankWordDetokenizer()
            return d.detokenize(tokens)

        except ImportError:
            DEFAULT_LOGGER.warning(
                "You need to install `nltk` to use `NLTKTokenizer`. Fallback to Space Join Detokenizer"
            )
            return " ".join(tokens)
