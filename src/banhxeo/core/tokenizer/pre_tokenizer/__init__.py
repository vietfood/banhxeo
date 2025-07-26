from __future__ import annotations

import functools
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from banhxeo.core.tokenizer.config import Token
from banhxeo.core.tokenizer.normalizers import NormalizedString
from banhxeo.utils.logging import default_logger


@dataclass
class Split:
    normalized: NormalizedString
    tokens: Optional[List[Token]] = None


@dataclass
class PreTokenizedString:
    splits: List[Split]


class PreTokenizer(ABC):
    @abstractmethod
    def pre_tokenize(self, pre_tokenized: PreTokenizedString) -> PreTokenizedString:
        """
        Takes a PreTokenizedString and returns a new, further-split PreTokenizedString.
        """


class RegexPreTokenizer(PreTokenizer):
    def __init__(self, pattern):
        self.pattern = pattern

    def pre_tokenize(self, pre_tokenized: PreTokenizedString) -> PreTokenizedString:
        new_splits = []
        for split in pre_tokenized.splits:
            for match in re.finditer(
                self.pattern,
                (normalized_str := split.normalized).normalized,
                re.UNICODE,
            ):
                new_splits.append(Split(normalized=normalized_str[match.span()[0] : match.span()[1]]))  # type: ignore
        return PreTokenizedString(splits=new_splits)


class WhiteSpacePreTokenizer(RegexPreTokenizer):
    def __init__(self):
        super().__init__(pattern=r"\S+")


class PunctuationPreTokenizer(RegexPreTokenizer):
    def __init__(self):
        """Reference: https://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation"""
        super().__init__(pattern=r"\w+|[^\w\s]")


class NLTKPreTokenizer(WhiteSpacePreTokenizer):
    def __init__(self):
        super().__init__()

        self._tokenizer = None
        try:
            from nltk.tokenize.treebank import TreebankWordTokenizer

            self._tokenizer = TreebankWordTokenizer()
        except ImportError:
            # fallback to whitespace Tokenizer
            default_logger.warning(
                "NLTK not found. Falling back to whitespace pre-tokenizer. "
                "Install NLTK for NLTKPreTokenizer: `pip install nltk`"
            )

    def pre_tokenize(self, pre_tokenized: PreTokenizedString) -> PreTokenizedString:
        if self._tokenizer is None:
            return super().pre_tokenize(pre_tokenized)

        new_splits = []
        for split in pre_tokenized.splits:
            tokenized_offsets = self._tokenizer.span_tokenize(
                split.normalized.normalized
            )
            for offset in tokenized_offsets:
                new_splits.append(Split(normalized=split.normalized[offset[0] : offset[1]]))  # type: ignore

        return PreTokenizedString(splits=new_splits)


class SequencePreTokenizer(PreTokenizer):
    def __init__(self, sequences: List[PreTokenizer] | PreTokenizer = []):
        self.sequences = sequences if isinstance(sequences, list) else [sequences]

    def add(self, pre_tokenizer: PreTokenizer):
        self.sequences.append(pre_tokenizer)

    def pre_tokenize(self, pre_tokenized: PreTokenizedString) -> PreTokenizedString:
        result = pre_tokenized
        for pre_tokenizer in self.sequences:
            result = pre_tokenizer.pre_tokenize(result)
        return result


@functools.lru_cache(maxsize=None)
def bytes_to_unicode():
    """
    Taken from: https://github.com/openai/gpt-2/blob/master/src/encoder.py
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class ByteLevelPreTokenizer(PreTokenizer):
    def __init__(
        self,
        add_prefix_space: bool = True,
    ):
        self.add_prefix_space = add_prefix_space
        self.bytes_to_unicode = bytes_to_unicode()

    def pre_tokenize(self, pre_tokenized: PreTokenizedString) -> PreTokenizedString:
        def bytes_transform(s: str):
            return "".join([self.bytes_to_unicode[byte] for byte in s.encode("utf-8")])

        new_splits = []

        for split in pre_tokenized.splits:
            normalized = split.normalized
            if self.add_prefix_space and not normalized.normalized.startswith(" "):
                normalized = normalized.prepend(bytes_to_unicode()[ord(" ")])
            normalized = normalized.transform(bytes_transform)
            new_splits.append(Split(normalized=normalized))

        return PreTokenizedString(splits=new_splits)


PRE_TOKENIZER_FACTORY = {
    "nltk": NLTKPreTokenizer,
    "whitespace": WhiteSpacePreTokenizer,
    "punctuation": PunctuationPreTokenizer,
    "regex": RegexPreTokenizer,
    "sequence": SequencePreTokenizer,
    "bytelevel": ByteLevelPreTokenizer,
}
