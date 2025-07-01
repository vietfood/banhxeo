from __future__ import annotations

import json
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Union

import jax
from jax import numpy as jnp
from pydantic import BaseModel, computed_field, field_validator, model_validator
from typing_extensions import Self

from banhxeo.utils import progress_bar, validate_config
from banhxeo.utils.logging import default_logger


class TokenizerConfig(BaseModel):
    min_freq: int = 1

    pad_tok: str = "<PAD>"
    unk_tok: str = "<UNK>"
    bos_tok: str = "<BOS>"
    eos_tok: str = "<EOS>"

    # BERT/Seq2Seq specific
    cls_tok: str = "<CLS>"
    mask_tok: str = "<MASK>"
    resv_tok: str = "<RESV>"

    @field_validator("min_freq", mode="before")
    @classmethod
    def check_min_freq(cls, value: int) -> int:  # noqa: D102
        if value < 1:
            raise ValueError("Minimum frequency (min_freq) must be at least 1.")
        return value

    @computed_field
    @property
    def special_tokens(self) -> List[str]:
        """Returns a list of all configured special tokens in a conventional order.
        0. `pad_tok`
        1. `unk_tok`
        2. `cls_tok`
        3. `eos_tok`
        4. `mask_tok`
        5. `bos_tok`
        6. `resv_tok`
        """
        ordered_tokens = [
            self.pad_tok,
            self.unk_tok,
            self.cls_tok,
            self.eos_tok,
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
            return self.special_tokens.index(token)
        except ValueError:
            raise ValueError(
                f"Token '{token}' is not a configured special token in this VocabConfig."
            )


class EncodeConfig(BaseModel):
    max_length: Optional[int] = None
    truncation: bool = False
    padding: Union[bool, Literal["do_not_pad", "max_length", "longest"]] = (
        False  # False = "do_not_pad", True = "longest"
    )
    padding_side: Literal["left", "right"] = "left"
    truncation_side: Literal["left", "right"] = "right"

    @model_validator(mode="after")
    def check_padding(self) -> Self:  # noqa: D102
        if isinstance(self.padding, str):
            if self.padding == "max_length" and self.max_length is None:
                raise ValueError(
                    "If padding is max_length, you must provide value for max_length parameter"
                )
        elif isinstance(self.padding, bool):
            self.padding = "longest" if self.padding else "do_not_pad"
        return self


default_tokenizer_config = TokenizerConfig()


class Tokenizer(metaclass=ABCMeta):
    type: str = "base"

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config if config else default_tokenizer_config
        self._token_to_idx: Dict[str, int] = {}
        self._idx_to_token: List[str] = []

    @property
    def vocab_size(self) -> int:
        return len(self._idx_to_token)

    @property
    def pad_id(self) -> int:
        return self._token_to_idx[self.config.pad_tok]

    @property
    def unk_id(self) -> int:
        return self._token_to_idx[self.config.unk_tok]

    @property
    def bos_id(self) -> Optional[int]:
        return self._token_to_idx.get(self.config.bos_tok)

    @property
    def eos_id(self) -> Optional[int]:
        return self._token_to_idx.get(self.config.eos_tok)

    @property
    def pad_tok(self) -> str:
        return self.config.pad_tok

    @property
    def bos_tok(self) -> str:
        return self.config.bos_tok

    @property
    def eos_tok(self) -> str:
        return self.config.eos_tok

    @property
    def unk_tok(self) -> str:
        return self.config.unk_tok

    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> List[str]:
        # Default is regrex tokenizer
        import re

        pattern = r"\w+(?:'\w+)?|[^\w\s]"
        return re.findall(pattern, text)

    def build(self, corpus: Iterable[str]):
        if self.type == "hf":
            raise ValueError("Cannot build from HuggingFace Tokenizer.")

        # Add special tokens first, in the order defined by the config's property.
        for token_str in self.config.special_tokens:
            if token_str and token_str not in self._token_to_idx:
                idx = len(self._idx_to_token)
                self._idx_to_token.append(token_str)
                self._token_to_idx[token_str] = idx

        token_counts = defaultdict(int)

        total_samples = 0
        for idx_sample, text_sample in progress_bar(
            enumerate(corpus),
            unit=" sentence",
            unit_scale=True,
            desc="Building vocabulary",
        ):
            try:
                tokens = self.tokenize(text_sample)
                for token in tokens:
                    token_counts[token] += 1
            except Exception as e:
                default_logger.warning(
                    f"Tokenizer failed on sample {idx_sample} ('{text_sample[:50]}...'): {e}. Skipping."
                )
            total_samples += 1

        sorted_tokens_by_freq = sorted(
            token_counts.items(), key=lambda item: item[1], reverse=True
        )

        min_freq = self.config.min_freq
        added_corpus_tokens = 0
        for token, count in sorted_tokens_by_freq:
            if (
                count >= min_freq and token not in self._token_to_idx
            ):  # Check it's not already a special token
                idx = len(self._idx_to_token)
                self._idx_to_token.append(token)
                self._token_to_idx[token] = idx
                added_corpus_tokens += 1

        default_logger.info(
            f"Vocabulary built: {len(self._idx_to_token)} unique tokens "
            f"({len(self.config.special_tokens)} special, {added_corpus_tokens} from corpus) "
            f"from {total_samples} sentences. Min frequency: {min_freq}."
        )

    def __call__(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        return_array: bool = True,
        **kwargs,
    ) -> Union[Dict[str, jax.Array], Dict[str, List[int]]]:
        config = validate_config(
            config_cls=EncodeConfig, add_special_tokens=add_special_tokens, **kwargs
        )

        batch_ids = []

        for text in texts:
            tokens = self.tokenize(text)

            if add_special_tokens:
                tokens = [self.bos_tok] + tokens + [self.eos_tok]

            if (
                config.truncation
                and config.max_length is not None
                and len(tokens) > config.max_length
            ):
                if config.truncation_side == "right":
                    tokens = tokens[: config.max_length]
                else:
                    tokens = tokens[-config.max_length :]

            batch_ids.append(
                [self._token_to_idx.get(tok, self.pad_id) for tok in tokens]
            )

        batch_longest = max([len(ids) for ids in batch_ids]) if batch_ids else 0
        batch_size = len(batch_ids)

        if config.padding == "max_length" and config.max_length is not None:
            max_seq_len = config.max_length
        elif config.padding == "longest":
            # Find the max length from the processed lists. Handle empty input.
            max_seq_len = batch_longest
        else:
            # If no padding then must ensure all sentence has same shape
            if len({len(ids) for ids in batch_ids}) > 1:
                raise ValueError(
                    "Padding is False/'do_not_pad', but sequences have different lengths. "
                    "Enable padding or ensure all sequences are the same length."
                )
            max_seq_len = batch_longest

        ids_arr = jnp.full(
            (batch_size, max_seq_len), fill_value=self.pad_id, dtype=jnp.int64
        )

        for i, id_list in enumerate(batch_ids):
            ids = id_list[:max_seq_len]
            if config.padding_side == "right":
                ids_arr[i, : len(ids)] = ids
            elif config.padding_side == "left":
                ids_arr[i, -len(ids) :] = ids

        attention_mask_arr = (ids_arr != self.pad_id).astype(jnp.int64)

        if return_array:
            outputs = {"input_ids": ids_arr, "attention_mask": attention_mask_arr}
        else:
            outputs = {
                "input_ids": ids_arr.tolist(),
                "attention_mask": attention_mask_arr.tolist(),
            }

        return outputs

    def encode(
        self,
        texts: List[str],
        return_array: bool = True,
        **kwargs,
    ) -> Union[jax.Array, List[int]]:
        results = self.__call__(texts, return_array, **kwargs)
        return results["input_ids"]

    def save(self, save_directory: Union[str, Path]):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(path / "tokenizer_config.json", "w") as f:
            f.write(self.config.model_dump_json(indent=2))

        # Save vocabulary
        with open(path / "vocab.json", "w") as f:
            json.dump(self._token_to_idx, f, indent=2)

        # Save tokenizer class info to know how to load it
        with open(path / "tokenizer.json", "w") as f:
            json.dump({"tokenizer_class": self.__class__.__name__}, f)

    @classmethod
    def load_from_disk(cls, load_directory: Union[str, Path]) -> "Tokenizer":
        if cls.type == "hf":
            raise ValueError(
                "Cannot use load_from_disk for HuggingFace Tokenizer, use load_pretrained instead"
            )

        path = Path(load_directory)

        with open(path / "tokenizer_config.json", "r") as f:
            config = TokenizerConfig.model_validate_json(f.read())

        with open(path / "tokenizer.json", "r") as f:
            data = json.load(f)
            saved_tokenizer_name = data.get("tokenizer_class", "Unknown")
            if cls.__class__.__name__ != saved_tokenizer_name:
                default_logger.warning(
                    f"Tokenizer mismatch: Loaded vocabulary was built with "
                    f"'{saved_tokenizer_name}', but provided tokenizer is "
                    f"'{cls.__class__.__name__}'. Ensure compatibility."
                )
            tokenizer = cls(config=config)

        with open(path / "vocab.json", "r") as f:
            tokenizer._token_to_idx = json.load(f)
            # Rebuild idx_to_token mapping
            tokenizer._idx_to_token = [""] * len(tokenizer._token_to_idx)
            for token, idx in tokenizer._token_to_idx.items():
                tokenizer._idx_to_token[idx] = token

        return tokenizer


class NLTKTokenizer(Tokenizer):
    """A tokenizer that uses NLTK's TreebankWordTokenizer"""

    type: str = "nltk"

    def tokenize(self, text: str, **kwargs) -> List[str]:
        try:
            from nltk.tokenize.treebank import TreebankWordTokenizer

            t = TreebankWordTokenizer()
            return t.tokenize(text)

        except ImportError:
            default_logger.warning(
                "NLTK not found. Falling back to regex-based tokenizer. "
                "Install NLTK for TreebankWordTokenizer: `pip install nltk`"
            )
            return super().tokenize(text, **kwargs)

    def detokenize(self, tokens: List[str], **kwargs) -> str:
        try:
            from nltk.tokenize.treebank import TreebankWordDetokenizer

            detokenizer = TreebankWordDetokenizer()
            return detokenizer.detokenize(tokens)
        except ImportError:
            default_logger.warning(
                "NLTK not found. Falling back to space-joining detokenizer. "
                "Install NLTK for TreebankWordDetokenizer: `pip install nltk`"
            )
            return " ".join(tokens)
