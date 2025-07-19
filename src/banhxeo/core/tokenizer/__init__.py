from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from banhxeo.core.tokenizer.model import TokenizerModel
from banhxeo.core.tokenizer.normalizer import NormalizedString, Normalizer
from banhxeo.core.tokenizer.post_processor import PostProcessor
from banhxeo.core.tokenizer.pre_tokenizer import PreTokenizer


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

    @functools.lru_cache(maxsize=None)
    def special_tokens(self) -> List[str]:
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
            return self.special_tokens().index(token)
        except ValueError:
            raise ValueError(
                f"Token '{token}' is not a configured special token in this VocabConfig."
            )


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


class Tokenizer:
    def __init__(
        self,
        normalizer: Normalizer,
        pre_tokenizer: PreTokenizer,
        model: TokenizerModel,
        post_processor: PostProcessor,
    ):
        self.normalizer = normalizer
        self.pre_tokenizer = pre_tokenizer
        self.model = model
        self.post_processor = post_processor

    def __call__(
        self,
        texts: List[str],
        return_tensors: Optional[Literal["jax", "np"]] = "jax",
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: Union[bool, Literal["do_not_pad", "max_length", "longest"]] = (
            False  # False = "do_not_pad", True = "longest"
        ),
        padding_side: Literal["left", "right"] = "left",
        truncation_side: Literal["left", "right"] = "right",
        add_special_tokens: bool = True,
    ) -> Dict[str, Any]:
        pre_tokenized_strs = []
        for text in texts:
            # Step 1: Normalized string
            normalized_string = NormalizedString.from_str(text)
            normalized_string = self.normalizer.normalize(normalized_string)

            # Step 2: Pre Tokenize normalized string
            pre_tokenized_str = self.pre_tokenizer.pre_tokenize(normalized_string)

            # Step 3: Turn pre tokenized to ID
            self.model.tokenize(pre_tokenized_str)

            pre_tokenized_strs.append(pre_tokenized_str)

        # Step 4: Post process ID tokenized string
        post_process_config = ProcessConfig(
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            padding_side=padding_side,
            truncation_side=truncation_side,
            add_special_tokens=add_special_tokens,
        )

        post_process_result = self.post_processor.process_batch(
            pre_tokenized_strs, config=post_process_config
        )

        match return_tensors:
            case "jax":
                import jax.numpy as jnp

                return {
                    key: jnp.array(value) for key, value in post_process_result.items()
                }
            case "np":
                import numpy as np

                return {
                    key: np.array(value) for key, value in post_process_result.items()
                }
            case _:
                return post_process_result

    def encode(
        self,
        text: str,
        return_tensors: Optional[Literal["jax", "np"]] = "jax",
        **kwargs,
    ) -> Dict[str, Any]:
        results = self.__call__([text], return_tensors, **kwargs)
        return results["input_ids"]

    def batch_encode(
        self,
        texts: List[str],
        return_tensors: Optional[Literal["jax", "np"]] = "jax",
        **kwargs,
    ) -> Dict[str, Any]:
        results = self.__call__(texts, return_tensors, **kwargs)
        return results["input_ids"]
