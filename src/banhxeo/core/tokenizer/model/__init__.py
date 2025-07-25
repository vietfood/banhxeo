from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

from banhxeo.core.tokenizer.pre_tokenizer import PreTokenizedString


class TokenizerModel(ABC):
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


from banhxeo.core.tokenizer.model.bpe import BPEModel
from banhxeo.core.tokenizer.model.word import WordLevelModel

MODEL_FACTORY = {"bpe": BPEModel, "word": WordLevelModel}
