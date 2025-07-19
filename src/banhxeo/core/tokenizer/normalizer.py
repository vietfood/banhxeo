from __future__ import annotations

import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Tuple


@dataclass
class NormalizedString:
    """
    - This is the direct modification from NormalizedString in HuggingFace's Tokenizer library
    - Reference: https://github.com/huggingface/tokenizers/blob/v0.21.3/tokenizers/src/tokenizer/normalizer.rs#L844
    """

    original: str  # Before modification
    normalized: str  # After modification

    # Mapping from normalized string to original one: (start, end) for each
    # index of the normalized string
    alignments: List[Tuple[int, int]]

    # If this NormalizedString is a slice of a bigger one, we keep the track
    # of the missing part, so that we can still give offsets from this original
    # string.
    original_shift: int

    def __eq__(self, value) -> bool:
        return self.normalized == value.normalized

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.slice(index.start, index.stop, index.step)
        else:
            if index >= self.__len__():
                raise IndexError()
            return self.normalized[index]

    def __len__(self):
        return len(self.normalized)

    @property
    def original_offset(self) -> Tuple[int, int]:
        return (self.original_shift, self.original_shift + len(self.original))

    @classmethod
    def from_str(cls, s: str) -> NormalizedString:
        return cls(
            original=s,
            normalized=s,
            alignments=[(v, v + 1) for v in range(len(s))],
            original_shift=0,
        )

    def slice(self, start: int, end: int, step: int = 1) -> NormalizedString:
        if end < start:
            raise ValueError()

        if end - start > len(self.normalized):
            raise ValueError()

        offset = self.alignments[start][0] if start < len(self) else len(self)

        return NormalizedString(
            original=self.original,
            normalized=self.normalized[start:end:step],
            alignments=self.alignments[start:end:step],
            original_shift=offset,
        )

    def filter(self, filter_fn: Callable[[str], bool]) -> NormalizedString:
        """
        Args:
            fileter_fn (Callable[[str], bool]): A function that accepts a character and returns True if the character should be kept, and False if it should be discarded.
        """
        kept_chars = {
            idx: chr for idx, chr in enumerate(self.normalized) if filter_fn(chr)
        }
        result_str = "".join(kept_chars.values())

        return NormalizedString(
            original=self.original,
            normalized=result_str,
            alignments=[self.alignments[idx] for idx in kept_chars.keys()],
            original_shift=self.original_shift,
        )

    def transform(self, change_fn: Callable[[str], str]) -> NormalizedString:
        normalized = self.normalized
        alignments = self.alignments

        result_str = []
        result_alignments = []

        for idx, char in enumerate(normalized):
            changed = change_fn(char)
            align = alignments[idx]
            n = len(changed)
            result_str.extend(changed)
            result_alignments.extend([align] * n)

        return NormalizedString(
            original=self.original,
            normalized="".join(result_str),
            alignments=result_alignments,
            original_shift=self.original_shift,
        )


class Normalizer(ABC):
    @abstractmethod
    def normalize(self, normalized: NormalizedString) -> NormalizedString:
        """
        Applies normalization to a NormalizedString.
        """


class StripNormalizer(Normalizer):
    def __init__(self, strip_left: bool = True, strip_right: bool = True):
        self.left = strip_left
        self.right = strip_right

    def normalize(self, normalized: NormalizedString) -> NormalizedString:
        start_idx = 0
        end_idx = len(normalized)

        if self.left:
            while (
                start_idx < len(normalized)
                and normalized.normalized[start_idx].isspace()
            ):
                start_idx += 1

        if self.right:
            while end_idx > start_idx and normalized.normalized[end_idx - 1].isspace():
                end_idx -= 1

        return normalized[start_idx:end_idx:1]  # type: ignore


class LowercaseNormalizer(Normalizer):
    def normalize(self, normalized: NormalizedString) -> NormalizedString:
        return normalized.transform(change_fn=lambda x: x.lower())


class NFCNormalizer(Normalizer):
    def normalize(self, normalized: NormalizedString) -> NormalizedString:
        def process_cluster(
            cluster_chars, cluster_alignments, final_chars, final_alignments
        ):
            if not cluster_chars:  # on not empty cluster
                return

            norm_chr = unicodedata.normalize("NFC", "".join(cluster_chars))

            final_chars.extend(list(norm_chr))
            final_alignments.extend(
                [(cluster_alignments[0][0], cluster_alignments[-1][1])] * len(norm_chr)
            )

        final_chars = []
        final_alignments = []

        current_cluster_chars = []
        current_cluster_alignments = []

        for char, alignment in zip(normalized.normalized, normalized.alignments):
            if unicodedata.combining(char) == 0:
                process_cluster(
                    current_cluster_chars,
                    current_cluster_alignments,
                    final_chars,
                    final_alignments,
                )

                current_cluster_chars.clear()
                current_cluster_alignments.clear()

            current_cluster_chars.append(char)
            current_cluster_alignments.append(alignment)

        # The loop has finished, but the last cluster is still in list
        process_cluster(
            current_cluster_chars,
            current_cluster_alignments,
            final_chars,
            final_alignments,
        )

        return NormalizedString(
            original=normalized.original,
            normalized="".join(final_chars),
            alignments=final_alignments,
            original_shift=normalized.original_shift,
        )


class SequenceNormalizer(Normalizer):
    def __init__(self, sequences: List[Normalizer] | Normalizer):
        self.sequences = sequences if isinstance(sequences, list) else [sequences]

    def normalize(self, normalized: NormalizedString) -> NormalizedString:
        result = normalized
        for normalizer in self.sequences:
            result = normalizer.normalize(result)
        return result
