from __future__ import annotations

import functools
from dataclasses import asdict, dataclass
from typing import List, Literal, Optional, Tuple, Union


@dataclass
class Token:
    id: int
    value: str
    offsets: Tuple[int, int]


@dataclass
class SpecialTokens:
    pad_tok: str = "<PAD>"
    unk_tok: str = "<UNK>"
    bos_tok: str = "<BOS>"
    eos_tok: str = "<EOS>"

    # BERT/Seq2Seq specific
    cls_tok: str = "<CLS>"
    sep_tok: str = "<SEP>"
    mask_tok: str = "<MASK>"
    resv_tok: str = "<RESV>"

    @functools.cached_property
    def tokens(self) -> List[str]:
        """Returns a list of all configured special tokens in a conventional order.

        0. `pad_tok`
        1. `unk_tok`
        2. `cls_tok`
        3. `eos_tok`
        4. `sep_tok`
        5. `mask_tok`
        6. `bos_tok`
        7. `resv_tok`
        """
        ordered_tokens = [
            self.pad_tok,
            self.unk_tok,
            self.cls_tok,
            self.eos_tok,
            self.sep_tok,
            self.mask_tok,
            self.bos_tok,
            self.resv_tok,
        ]

        # Avoid duplicate (in case user sets, e.g., bos_tok = sep_tok)
        final_tokens = []
        seen = set()
        for token in ordered_tokens:
            if token not in seen:
                final_tokens.append(token)
                seen.add(token)
        return final_tokens

    def special_token_idx(self, token: str) -> int:  # noqa: D102
        try:
            return self.tokens.index(token)
        except ValueError:
            raise ValueError(
                f"Token '{token}' is not a configured special token in this VocabConfig."
            )

    @property
    def pad_id(self):
        return self.special_token_idx(self.pad_tok)

    @property
    def unk_id(self):
        return self.special_token_idx(self.unk_tok)


default_special_tokens = SpecialTokens()


@dataclass
class ProcessConfig:
    max_length: Optional[int] = None
    truncation: bool = False
    padding: Union[bool, Literal["do_not_pad", "max_length", "longest"]] = (
        False  # False = "do_not_pad", True = "longest"
    )
    padding_side: Literal["left", "right"] = "left"
    truncation_side: Literal["left", "right"] = "right"
    add_special_tokens: bool = True

    def dict(self):
        return {k: v for k, v in asdict(self).items()}
