from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Iterable, List

from banhxeo.core.tokenizer.config import SpecialTokens
from banhxeo.core.tokenizer.pre_tokenizer import PreTokenizedString
from banhxeo.utils.logging import default_logger


class TokenizerModel(ABC):
    def __init__(self, special_tokens: SpecialTokens):
        self.special_tokens = special_tokens
        self.vocab = defaultdict(int)  # token to id
        self.inverse_vocab = []  # id to token
        self.trained = False

    @abstractmethod
    def tokenize(self, pre_tokenized_str: PreTokenizedString): ...

    @abstractmethod
    def detokenize(self, token_ids: List[int]) -> List[str]: ...

    @abstractmethod
    def train(
        self,
        corpus: Iterable[PreTokenizedString],
        progress: bool = True,
        **kwargs,
    ): ...

    @classmethod
    @abstractmethod
    def from_config(cls, config): ...

    def get_vocab(self):
        if not self.trained:
            default_logger.warning(
                "Tokenizer hasn't been trained. Return None Vocabulary"
            )
            return None
        return self.vocab


from banhxeo.core.tokenizer.model.bpe import BPEModel
from banhxeo.core.tokenizer.model.word import WordLevelModel

MODEL_FACTORY = {"bpe": BPEModel, "word": WordLevelModel}
