from __future__ import annotations

from banhxeo.core.tokenizer import Tokenizer
from banhxeo.core.tokenizer.config import SpecialTokens, default_special_tokens
from banhxeo.core.tokenizer.decoder import (
    ByteLevelDecoder,
    NLTKDecoder,
    WhiteSpaceDecoder,
)
from banhxeo.core.tokenizer.model import BPEModel, WordLevelModel
from banhxeo.core.tokenizer.normalizers import (
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
            WhiteSpaceDecoder(),
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


class GPT2_Tokenizer(Tokenizer):
    def __init__(self, special_tokens: SpecialTokens = default_special_tokens):
        super().__init__(
            SequenceNormalizer([StripNormalizer(), NFCNormalizer()]),
            ByteLevelPreTokenizer(add_prefix_space=False),
            BPEModel(special_tokens, add_boundary_word="</w>"),
            GPTPostProcessor(special_tokens),
            ByteLevelDecoder(),
        )
