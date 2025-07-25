from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List

from banhxeo.core.tokenizer.config import SpecialTokens, Token
from banhxeo.core.tokenizer.model import TokenizerModel
from banhxeo.core.tokenizer.pre_tokenizer import PreTokenizedString
from banhxeo.utils import progress_bar
from banhxeo.utils.file import default_logger


class WordLevelModel(TokenizerModel):
    def __init__(self, special_tokens: SpecialTokens):
        self.special_tokens = special_tokens

        self.vocab = defaultdict(int)  # token to id
        self.inverse_vocab = []

        self.trained = False

    def tokenize(self, pre_tokenized_str: PreTokenizedString):
        if not self.vocab:
            raise ValueError(
                "You haven't build Tokenizer. Please use `WordLevelModel.train(...)`"
            )

        for split in pre_tokenized_str.splits:
            vocab_id = self.vocab.get(split.normalized.normalized)
            if vocab_id is None:
                vocab_id = self.vocab[self.special_tokens.unk_tok]

            split.tokens = [
                Token(
                    id=vocab_id,
                    value=split.normalized.normalized,
                    offsets=(
                        split.normalized.alignments[0][0],
                        split.normalized.alignments[-1][1],
                    ),
                )
            ]

    def detokenize(self, token_ids: List[int]) -> List[str]:
        return [self.inverse_vocab[token_id] for token_id in token_ids]

    def train(
        self,
        corpus: Iterable[PreTokenizedString],  # already in unique word
        progress: bool = True,
        **kwargs,
    ):
        if self.trained:
            default_logger.warning("This Tokenized has been trained before. Return")
            return

        # Add special tokens first
        self.vocab = {
            token: idx for idx, token in enumerate(self.special_tokens.special_tokens)
        }

        self.inverse_vocab = [token for token in self.special_tokens.special_tokens]

        # Add rest of vocab
        all_words = set(
            [
                split.normalized.normalized
                for sentence in corpus
                for split in sentence.splits
            ]
        )

        current_id = len(self.vocab)
        for word in progress_bar(
            sorted(list(all_words)), disable=not progress, desc="Add word to vocabulary"
        ):
            if word not in self.vocab:
                self.vocab[word] = current_id
                self.inverse_vocab.append(word)
                current_id += 1

        self.trained = True

    @classmethod
    def from_config(cls, config):
        # TODO:
        ...
