from __future__ import annotations

import functools
import itertools
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple, TypeAlias

from banhxeo.core.tokenizer import SpecialTokens, Token, default_special_tokens
from banhxeo.core.tokenizer.model import TokenizerModel
from banhxeo.core.tokenizer.pre_tokenizer import PreTokenizedString
from banhxeo.utils import progress_bar

BPEWord: TypeAlias = Dict[str, Tuple[int, List[str]]]


def get_pair_stats(word_freqs: BPEWord):
    pair_stats = defaultdict(int)

    for _, (frequency, split) in word_freqs.items():
        # add each pair to pair stats
        for pair in itertools.pairwise(split):
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
    def __init__(self, special_tokens: SpecialTokens):
        self.special_tokens = special_tokens
        self.vocab = defaultdict(int)  # token to id
        self.merges = []

    @functools.lru_cache(maxsize=None)
    def _tokenize_word(self, word: str) -> List[str]:
        if not self.merges:
            raise ValueError(
                "You haven't build Tokenizer. Please use `BPETokenizer.train(...)`"
            )

        subwords = list(word) + ["</w>"]

        for pair in self.merges:
            while True:
                has_merged = False
                next_subwords = []
                idx = 0

                # Scan through the current subwords list
                while idx < len(subwords):
                    if (
                        idx < len(subwords) - 1
                        and (subwords[idx], subwords[idx + 1]) == pair
                    ):
                        next_subwords.append("".join(pair))
                        idx += 2
                        has_merged = True
                    else:
                        next_subwords.append(subwords[idx])
                        idx += 1

                subwords = next_subwords

                if not has_merged:
                    break

        return subwords

    @classmethod
    def train(
        cls,
        corpus: Iterable[PreTokenizedString],  # already in unique word
        vocab_size: int,
        special_tokens: SpecialTokens = default_special_tokens,
        progress: bool = True,
    ):
        bpe = cls(special_tokens)

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

        bpe.vocab = {
            token: idx for idx, token in enumerate(bpe.special_tokens.special_tokens())
        }

        current_id = len(bpe.vocab)
        for char in sorted(list(initial_vocab_chars)):
            if char not in bpe.vocab:
                bpe.vocab[char] = current_id
                current_id += 1

        # 2. Main Loop
        initial_vocab_size = len(initial_vocab_chars)
        word_freqs = {
            word + "</w>": (count, list(word) + ["</w>"])
            for word, count in word_counts.items()
        }

        for _ in progress_bar(
            range(0, (vocab_size - initial_vocab_size)), disable=not progress
        ):
            pair_stats = get_pair_stats(word_freqs)

            if not pair_stats:
                break  # already merge

            most_freq_pair = max(pair_stats, key=pair_stats.get)  # type: ignore
            bpe.merges.append(most_freq_pair)

            merge_pair(most_freq_pair, word_freqs)
            bpe.vocab["".join(most_freq_pair)] = len(bpe.vocab)

        return bpe

    def tokenize(self, pre_tokenized_str: PreTokenizedString):
        """
        Tokenizes a PreTokenizedString in-place using the trained BPE model.
        """
        for split in pre_tokenized_str.splits:
            word = split.normalized.normalized
            subwords = self._tokenize_word(word)
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
