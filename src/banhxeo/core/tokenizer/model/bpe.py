from __future__ import annotations

import functools
import heapq
import itertools
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple, TypeAlias

import jax

from banhxeo import DEFAULT_SEED
from banhxeo.core.tokenizer.config import SpecialTokens, Token
from banhxeo.core.tokenizer.model import TokenizerModel
from banhxeo.core.tokenizer.pre_tokenizer import PreTokenizedString
from banhxeo.utils import progress_bar
from banhxeo.utils.logging import default_logger

# {"hugs": (15, ["h", "u", "g", "s"])} in which 15 is the frequency
BPEWord: TypeAlias = Dict[str, Tuple[int, List[str]]]


def get_pair_stats(word_freqs: BPEWord):
    pair_stats = defaultdict(int)

    for _, (frequency, split) in word_freqs.items():
        # add each pair to pair stats
        for pair in set(itertools.pairwise(split)):
            pair_stats[pair] += frequency

    return pair_stats


def merge_pair(pair_to_merge: Tuple[str, str], word_freqs: BPEWord):
    new_symbol = "".join(pair_to_merge)
    for word, (frequency, split) in word_freqs.items():
        if pair_to_merge[0] not in split or pair_to_merge[1] not in split:
            continue

        new_split = []
        idx = 0
        while idx < len(split):
            if (idx < len(split) - 1) and (split[idx], split[idx + 1]) == pair_to_merge:
                new_split.append(new_symbol)
                idx += 2
            else:
                new_split.append(split[idx])
                idx += 1

        word_freqs[word] = (frequency, new_split)


class BPEModel(TokenizerModel):
    def __init__(
        self, special_tokens: SpecialTokens, dropout: Optional[float] = None, **kwargs
    ):
        super().__init__(special_tokens=special_tokens)

        self.dropout = dropout
        if self.dropout:
            # add random key from jax
            self.rng = kwargs.get("rng", jax.random.key(DEFAULT_SEED))

        self.merges: Dict[Tuple[str, str], int] = dict()  # merge rules rank

    @functools.lru_cache(maxsize=None)
    def _tokenize_word(self, word: str, is_training: bool = False) -> List[str]:
        def create_pairs(subwords):
            return set(itertools.pairwise(subwords))

        if not self.merges:
            raise ValueError(
                "You haven't build Tokenizer. Please use `BPEModel.train(...)`"
            )

        # create subwords (by characters)
        subwords = list(word)
        if not subwords:
            return []
        subwords[-1] = subwords[-1] + "</w>"

        subwords_pairs = create_pairs(subwords)

        if not subwords_pairs:
            return subwords  # don't merge

        # create heap based on pairs with priority is rank in self.merges
        heap = []
        for i, pair in enumerate(itertools.pairwise(subwords)):
            rank = self.merges.get(pair)
            if rank is not None:
                # Store rank, position, and the pair itself.
                heapq.heappush(heap, (rank, i, pair))

        while heap:
            rank, pos, pair = heapq.heappop(heap)

            if pos >= len(subwords) - 1 or (subwords[pos], subwords[pos + 1]) != pair:
                continue

            # perform dropout
            if self.dropout and is_training:
                if jax.random.uniform(self.rng) < self.dropout:
                    continue  # skip this merge

            # perform merge
            merged_token = "".join(pair)
            subwords[pos : pos + 2] = [merged_token]

            if pos > 0:
                rank = self.merges.get(
                    (previous_merge := (subwords[pos - 1], subwords[pos]))
                )
                if rank is not None:
                    heapq.heappush(heap, (rank, pos - 1, previous_merge))

            if pos < len(subwords) - 1:
                rank = self.merges.get(
                    (after_merge := (subwords[pos], subwords[pos + 1]))
                )
                if rank is not None:
                    heapq.heappush(heap, (rank, pos, after_merge))

        return subwords

    def train(
        self,
        corpus: Iterable[PreTokenizedString],  # already in unique word
        progress: bool = True,
        **kwargs,
    ):
        if self.trained:
            default_logger.warning("This Tokenized has been trained before. Return")
            return

        vocab_size = kwargs.get("vocab_size")
        if vocab_size is None:
            raise ValueError("Cannot have empty vocab size")

        # 1. Preparation
        all_words = [
            split.normalized.normalized
            for sentence in corpus
            for split in sentence.splits
        ]
        word_counts = Counter(all_words)

        initial_vocab_chars = set()
        for word in word_counts:
            initial_vocab_chars.update(list(word))
        initial_vocab_chars.add("</w>")

        self.vocab = {
            token: idx for idx, token in enumerate(self.special_tokens.tokens)
        }

        self.inverse_vocab = [token for token in self.special_tokens.tokens]

        current_id = len(self.vocab)
        for char in sorted(list(initial_vocab_chars)):
            if char not in self.vocab:
                self.vocab[char] = current_id
                self.inverse_vocab.append(char)
                current_id += 1

        # 2. Main Loop
        initial_vocab_size = len(initial_vocab_chars)
        word_freqs = {
            word + "</w>": (count, list(word) + ["</w>"])
            for word, count in word_counts.items()
        }

        rank = 0
        pair_stats = get_pair_stats(word_freqs)

        for _ in progress_bar(
            range(0, (vocab_size - initial_vocab_size)),
            desc="Training BPE",
            disable=not progress,
        ):
            if len(pair_stats) == 0:
                break  # already merge

            most_freq_pair = max(pair_stats, key=pair_stats.get)  # type: ignore
            self.merges[most_freq_pair] = rank
            rank += 1

            merge_pair(most_freq_pair, word_freqs)
            # TODO: Finish BPE algorithm

            merge_token = "".join(most_freq_pair)
            self.vocab[merge_token] = len(self.vocab)
            self.inverse_vocab.append(merge_token)

        self.trained = True

    def detokenize(self, token_ids: List[int]) -> List[str]:
        return [self.inverse_vocab[token_id] for token_id in token_ids]

    def tokenize(
        self, pre_tokenized_str: PreTokenizedString, is_training: bool = False
    ):
        """
        Tokenizes a PreTokenizedString in-place using the trained BPE model.
        """
        for split in pre_tokenized_str.splits:
            word = split.normalized.normalized
            subwords = self._tokenize_word(word, is_training)
            tokens = []
            for subword in subwords:
                # Assign the original
                # word's full offsets to every subword token.
                # e.g., if "tokenization" has offsets (10, 22), then both
                # "token" and "##ization" get offsets (10, 22)
                subword_id = self.vocab.get(
                    subword, self.vocab[self.special_tokens.unk_tok]
                )

                tokens.append(
                    Token(
                        id=subword_id,
                        value=subword,
                        offsets=(
                            split.normalized.alignments[0][0],
                            split.normalized.alignments[-1][1],
                        ),
                    )
                )
            split.tokens = tokens

    @classmethod
    def from_config(cls, config):
        # TODO:
        ...
