from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from banhxeo.core.tokenizer import Token
from banhxeo.core.tokenizer.normalizer import NormalizedString


@dataclass
class Split:
    normalized: NormalizedString
    tokens: Optional[List[Token]] = None


@dataclass
class PreTokenizedString:
    splits: List[Split]


class PreTokenizer(ABC):
    @abstractmethod
    def pre_tokenize(self, normalized: NormalizedString) -> PreTokenizedString:
        """
        Splits a NormalizedString into a list of smaller NormalizedStrings.
        """


class WhitespacePreTokenizer(PreTokenizer):
    def pre_tokenize(self, normalized: NormalizedString) -> PreTokenizedString:
        return PreTokenizedString(
            splits=[
                Split(normalized=normalized[match.span()[0] : match.span()[1]])  # type: ignore
                for match in re.finditer(r"\S+", normalized.normalized, re.UNICODE)
            ]
        )


class PunctuationPreTokenizer(PreTokenizer):
    def pre_tokenize(self, normalized: NormalizedString) -> PreTokenizedString:
        """Reference: https://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation"""
        return PreTokenizedString(
            splits=[
                Split(normalized=normalized[match.span()[0] : match.span()[1]])  # type: ignore
                for match in re.finditer(
                    r"\w+|[^\w\s]", normalized.normalized, re.UNICODE
                )
            ]
        )
