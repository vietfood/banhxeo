from __future__ import annotations

from typing import Iterable

from banhxeo.core.tokenizer import SpecialTokens, Tokenizer, default_special_tokens
from banhxeo.core.tokenizer.decoder import ByteLevelDecoder, NLTKDecoder
from banhxeo.core.tokenizer.model import BPEModel, WordLevelModel
from banhxeo.core.tokenizer.normalizer import (
    NFCNormalizer,
    SequenceNormalizer,
    StripNormalizer,
)
from banhxeo.core.tokenizer.post_processor import GPTPostProcessor
from banhxeo.core.tokenizer.pre_tokenizer import (
    ByteLevelPreTokenizer,
    NLTKPreTokenizer,
    PunctuationPreTokenizer,
    SequencePreTokenizer,
    WhiteSpacePreTokenizer,
)


class SimpleTokenizer(Tokenizer):
    def __init__(self, special_tokens: SpecialTokens = default_special_tokens):
        super().__init__(
            SequenceNormalizer([StripNormalizer(), NFCNormalizer()]),
            SequencePreTokenizer([WhiteSpacePreTokenizer(), PunctuationPreTokenizer()]),
            WordLevelModel(special_tokens),
            GPTPostProcessor(special_tokens),
            NLTKDecoder(),
        )


class NLTKTokenizer(Tokenizer):
    def __init__(self, special_tokens: SpecialTokens = default_special_tokens):
        super().__init__(
            SequenceNormalizer([StripNormalizer(), NFCNormalizer()]),
            NLTKPreTokenizer(),
            WordLevelModel(special_tokens),
            GPTPostProcessor(special_tokens),
            NLTKDecoder(),
        )


class GPTTokenizer(Tokenizer):
    def __init__(self, special_tokens: SpecialTokens = default_special_tokens):
        super().__init__(
            SequenceNormalizer([StripNormalizer(), NFCNormalizer()]),
            ByteLevelPreTokenizer(),
            BPEModel(special_tokens),
            GPTPostProcessor(special_tokens),
            ByteLevelDecoder(),
        )
