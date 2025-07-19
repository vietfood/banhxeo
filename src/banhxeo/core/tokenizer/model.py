from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from banhxeo.core.tokenizer import Token
from banhxeo.core.tokenizer.pre_tokenizer import PreTokenizedString


class TokenizerModel(ABC):
    @abstractmethod
    def tokenize(self, pre_tokenized_str: PreTokenizedString):
        """
        Populates tokens for each split in the PreTokenizedString, in-place.
        """


class WordLevelModel(TokenizerModel):
    """
    TODO: Implement this
    """
