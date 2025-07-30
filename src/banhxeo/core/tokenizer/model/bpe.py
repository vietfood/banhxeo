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
# ("h", "ug")
WordPair: TypeAlias = Tuple[str, str]
# {("h", "u"): 15}
PairStats: TypeAlias = defaultdict[WordPair, int]
# Store word and its coresponding frequency
PairHeap: TypeAlias = List[Tuple[int, WordPair]]


def get_pair_stats(word_freqs: BPEWord) -> Tuple[PairStats, PairHeap]:
    pair_stats = defaultdict(int)

    for _, (frequency, split) in word_freqs.items():
        # add each pair to pair stats
        for pair in set(itertools.pairwise(split)):
            pair_stats[pair] += frequency

    # Then create a heap based on pair_stats
    pair_heap = [(-freq, pair) for pair, freq in pair_stats.items()]
    heapq.heapify(pair_heap)

    return pair_stats, pair_heap


def merge_pair(
    pair_to_merge: WordPair,
    word_freqs: BPEWord,
    inverted_word_freqs: defaultdict[WordPair, list],
    pair_stats: PairStats,
    pair_heap: PairHeap,
):
    p1, p2 = pair_to_merge
    new_symbol = p1 + p2

    changes = defaultdict(int)

    for word in inverted_word_freqs[pair_to_merge]:
        freq, split = word_freqs[word]

        i = 0
        new_word_split = []
        while i < len(split):
            if i < len(split) - 1 and (split[i], split[i + 1]) == pair_to_merge:
                # Update pair stats
                # Left side:
                if i > 0:
                    prev = split[i - 1]
                    changes[(prev, p1)] -= freq
                    changes[(prev, new_symbol)] += freq
                # Right side:
                if i < len(split) - 2:
                    nxt = split[i + 2]
                    changes[(p2, nxt)] -= freq
                    changes[(new_symbol, nxt)] += freq

                new_word_split.append(new_symbol)
                i += 2
            else:
                new_word_split.append(split[i])
                i += 1

        word_freqs[word] = (freq, new_word_split)

    for pair, delta in changes.items():
        pair_stats[pair] += delta
        heapq.heappush(pair_heap, (-pair_stats[pair], pair))

    # Remove the merge from stats
    del pair_stats[pair_to_merge]
    # Also remove from inverted word freqs
    del inverted_word_freqs[pair_to_merge]


class BPEModel(TokenizerModel):
    def __init__(
        self,
        special_tokens: SpecialTokens,
        dropout: Optional[float] = None,
        add_boundary_word: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(special_tokens=special_tokens)

        self.dropout = dropout
        if self.dropout:
            # add random key from jax
            self.rng = jax.random.key(kwargs.get("seed", DEFAULT_SEED))

        self.boundary_word = add_boundary_word if add_boundary_word else "</w>"
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
        if self.boundary_word:
            subwords.append(self.boundary_word)

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
            default_logger.warning("This Tokenizer has been trained before. Return")
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

        if self.boundary_word:
            initial_vocab_chars.add(self.boundary_word)

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

        # building word freqs and its inverted version
        word_freqs = {}
        inverted_word_freqs = defaultdict(list)

        for word, count in word_counts.items():
            word_key = word
            split = list(word)
            if self.boundary_word:
                word_key += self.boundary_word
                split += [self.boundary_word]

            word_freqs[word_key] = (count, split)
            for pair in itertools.pairwise(split):
                inverted_word_freqs[pair].append(word_key)

        rank = 0
        pair_stats, pair_heap = get_pair_stats(word_freqs)

        for _ in progress_bar(
            range(0, (vocab_size - initial_vocab_size)),
            desc="Training BPE",
            disable=not progress,
        ):
            if len(pair_stats) == 0:
                break  # already merge

            # pop most frequent pair
            most_freq_pair = None
            while pair_heap:
                nfreq, most_freq_pair = heapq.heappop(pair_heap)
                if (freq := pair_stats.get(most_freq_pair)) is None or freq != -nfreq:
                    continue
                else:
                    break

            if most_freq_pair is None:
                default_logger.warning("There is no pair to merge. Stop process")
                break

            # Add to merges dict
            self.merges[most_freq_pair] = rank
            rank += 1

            # Merge all pairs and update current stats
            merge_pair(
                most_freq_pair, word_freqs, inverted_word_freqs, pair_stats, pair_heap
            )

            # Add new pair with new frequency
            new_pair = "".join(most_freq_pair)

            # Add to vocab and inverse vocab
            merge_token = new_pair
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
