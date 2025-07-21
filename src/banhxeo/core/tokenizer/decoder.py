from abc import ABC, abstractmethod
from typing import List

from banhxeo.core.tokenizer.pre_tokenizer import bytes_to_unicode
from banhxeo.utils.logging import default_logger


class Decoder(ABC):
    @abstractmethod
    def decode(self, tokens: List[str], **kwargs) -> str:
        """Decodes a list of token strings into a single string."""


class ByteLevelDecoder(Decoder):
    def __init__(self):
        self.unicode_to_bytes = {
            value: key for key, value in bytes_to_unicode().items()
        }

    def decode(self, tokens: List[str], **kwargs) -> str:
        tokens_str = "".join(tokens)
        tokens_bytes = bytes(
            [self.unicode_to_bytes[tokens_chr] for tokens_chr in tokens_str]
        )
        return tokens_bytes.decode(encoding="utf-8", errors="replace")


class WhiteSpaceDecoder(Decoder):
    def decode(self, tokens: List[str], **kwargs) -> str:
        return " ".join(tokens)


class NLTKDecoder(Decoder):
    def __init__(self):
        self._detokenizer = None
        try:
            from nltk.tokenize.treebank import TreebankWordDetokenizer

            self._detokenizer = TreebankWordDetokenizer()
        except ImportError:
            # fallback to whitespace Tokenizer
            default_logger.warning(
                "NLTK not found. Falling back to join decoder."
                "Install NLTK for NLTKPreTokenizer: `pip install nltk`"
            )

    def decode(self, tokens: List[str], **kwargs) -> str:
        if self._detokenizer is None:
            return "".join(tokens)
        else:
            return self._detokenizer.detokenize(tokens)
