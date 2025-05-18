from __future__ import annotations

from abc import ABCMeta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from banhxeo.utils.logging import DEFAULT_LOGGER


class Tokenizer(metaclass=ABCMeta):
    """
    Abstract Base Class for all tokenizers.
    Defines the core interface for tokenization and related functionalities.
    """

    def __call__(self, text_or_texts: Union[List[str], str], **kwargs):
        """
        Allows the tokenizer instance to be called directly.
        Can handle a single string or a list of strings.

        Args:
            text_or_texts (Union[str, List[str]]): A single text string or a list of text strings.
            **kwargs: Additional keyword arguments passed to the `tokenize` method.

        Returns:
            Union[List[str], List[List[str]]]:
                - If input is a single string, returns a list of tokens.
                - If input is a list of strings, returns a list of lists of tokens.
        """
        if isinstance(text_or_texts, str):
            return self.tokenize(text_or_texts, **kwargs)
        elif isinstance(text_or_texts, list):
            return [self.tokenize(text, **kwargs) for text in text_or_texts]

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenizes a single string of text into a list of token strings.

        Args:
            text (str): The input text to tokenize.
            **kwargs: Additional keyword arguments specific to the tokenizer's implementation
                      (e.g., add_special_tokens=True for some advanced tokenizers).

        Returns:
            List[str]: A list of token strings.
        """
        # Default is regrex tokenizer
        import re

        pattern = r"\w+(?:'\w+)?|[^\w\s]"
        return re.findall(pattern, text)

    def encode(
        self,
        text: str,
        vocab: Vocabulary,  # type: ignore  # noqa: F821
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: Union[bool, str] = False,  # False, 'max_length'
        **kwargs,
    ) -> Dict[str, List[int]]:
        """
        Tokenizes text and converts tokens to their corresponding IDs using a vocabulary.
        Optionally adds special tokens, truncates, and pads.

        This is a convenience method. For simple tokenizers, it might just call
        `tokenize` and then `vocab.tokens_to_ids`. More advanced tokenizers (like HF wrappers)
        might have more sophisticated logic here.

        Args:
            text (str): The input text.
            vocab (Vocabulary): The vocabulary instance to map tokens to IDs.
            add_special_tokens (bool): Whether to add special tokens (e.g., BOS, EOS).
                                       The tokenizer needs to know which ones and how.
            max_length (Optional[int]): Maximum length for truncation or padding.
            truncation (bool): Whether to truncate the sequence if it exceeds max_length.
            padding (Union[bool, str]): Whether/how to pad. 'max_length' pads to max_length.
            **kwargs: Additional arguments for `tokenize` or specific encoding logic.

        Returns:
            Dict[str, List[int]]: A dictionary, typically with 'input_ids'.
                                  Could also include 'attention_mask' if padding is applied.
        """
        tokens = self.tokenize(text, **kwargs)  # kwargs might be passed to tokenize

        if truncation and max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]

        if add_special_tokens:
            tokens = [vocab.sos_tok] + tokens + [vocab.eos_tok]

        input_ids = vocab.tokens_to_ids(tokens)
        output = dict()

        if padding == "max_length":
            if max_length is not None:
                attention_mask = [1] * len(input_ids)
                pad_len = max_length - len(input_ids)
                if pad_len > 0:
                    input_ids += [vocab.pad_id] * pad_len
                    attention_mask += [0] * pad_len
                output["input_ids"] = input_ids  # Update if padded
                output["attention_mask"] = attention_mask
            else:
                # TODO: Add error message
                raise ValueError()
        
        elif padding == True or padding == ""

        return output

    def batch_encode(
        self,
        texts: List[str],
        vocab: Vocabulary,  # noqa: F821 # type: ignore
        **kwargs,
    ) -> List[Dict[str, List[int]]]:
        """
        Applies `encode` to a batch of texts.
        Subclasses might override this for efficiency (e.g., batch processing in HF tokenizers).

        Args:
            texts (List[str]): A list of text strings.
            vocab (Vocabulary): The vocabulary instance.
            **kwargs: Additional arguments for `encode`.

        Returns:
            List[Dict[str, List[int]]]: A list of dictionaries, one for each input text.
        """
        return [self.encode(text, vocab, **kwargs) for text in texts]

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

        Args:
            iterator (Iterable[str]): An iterator yielding raw text strings.
            vocab_size (int): The desired vocabulary size (for subword tokenizers).
            min_frequency (int): The minimum frequency for a token to be included.
            special_tokens (List[str]): List of special tokens to include.
            **kwargs: Tokenizer-specific training arguments.

        Raises:
            NotImplementedError: If the tokenizer is not trainable.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support training from an iterator."
        )

    # --- Saving and Loading (Essential for trained/complex tokenizers) ---

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        """
        Saves the tokenizer's configuration and any learned vocabulary/merges
        to a directory, so it can be reloaded.

        Args:
            save_directory (Union[str, Path]): Directory where the tokenizer files will be saved.
            **kwargs: Additional arguments.
        """
        ...
        raise NotImplementedError()

    @classmethod
    def from_pretrained(cls, load_directory: Union[str, Path], **kwargs) -> Tokenizer:
        """
        Loads a tokenizer from a previously saved directory.

        Args:
            load_directory (Union[str, Path]): Directory from which to load the tokenizer.
            **kwargs: Additional arguments.

        Returns:
            Tokenizer: An instance of the tokenizer.
        """
        ...


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
