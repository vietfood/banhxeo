import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, computed_field, field_validator

from banhxeo.core.tokenizer import Tokenizer
from banhxeo.utils import progress_bar
from banhxeo.utils.logging import DEFAULT_LOGGER


class VocabConfig(BaseModel):
    min_freq: int = 1
    pad_tok: str = "<PAD>"  # Special token shouldn't be empty
    unk_tok: str = "<UNK>"
    bos_tok: str = "<BOS>"  # begin of sentence
    sep_tok: str = "<SEP>"  # end of sentence or sequence seperate
    mask_tok: Optional[str] = None  # "<MASK>"  or BERT
    cls_tok: Optional[str] = None  # "<CLS>"
    resv_tok: Optional[str] = None  # "<RESERVED>"

    @field_validator("min_freq", mode="before")
    @classmethod
    def check_positive(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Min frequencies cannot be 0")
        return value

    @computed_field  # Cache
    @property
    def special_tokens(self) -> list[str]:
        """Following are list of all of the special tokens with
        their corresponding ids:
        - "[CLS]": 0 (Optional)
        - "[SEP]": 1
        - "[BOS]": 2
        - "[MASK]": 3 (Optional)
        - "[PAD]": 4
        - "[RESERVED]": 5 (Optional)
        - "[UNK]": 6
        - An id (starting at 7) will be assigned to each character.
        """
        result = []
        if self.cls_tok:
            result.append(self.cls_tok)
        result = result + [
            self.sep_tok,
            self.bos_tok,
        ]
        if self.mask_tok:
            result.append(self.mask_tok)
        result.append(self.pad_tok)
        if self.resv_tok:
            result.append(self.resv_tok)
        result.append(self.unk_tok)
        return result

    def special_token_idx(self, token) -> int:
        """Responsiblity from the caller, if token isn't in special token list, this method will raise ValueError"""

        idx = self.special_tokens.index(token)
        return idx


DEFAULT_VOCAB_CONFIG = VocabConfig()


class Vocabulary:
    """Manages token<=>ID mapping, special tokens."""

    def __init__(self, vocab_config: Optional[VocabConfig] = None):
        self.vocab_config = vocab_config if vocab_config else DEFAULT_VOCAB_CONFIG

        self.tokenizer = None
        self._idx_to_token = []
        self._token_to_idx = {}

    @classmethod
    def load(cls, path: Union[Path, str], tokenizer: Tokenizer):
        """Loading pre-trained vocabulary from file"""
        if isinstance(path, str):
            path = Path(path)

        if not path.is_file():
            raise ValueError(f"Current json path={path.name} is not a valid path")

        with open(path, "r") as file:
            d = json.load(file)
            vocab = cls(VocabConfig.model_validate(d["config"]))
            vocab._token_to_idx = d["token_to_idx"]
            vocab._idx_to_token = d["idx_to_token"]

            if tokenizer.__class__.__name__ != d["tokenizer"]:
                raise ValueError(
                    f"Current tokenizer {tokenizer.__class__.__name__} isn't match with tokenizer {d['tokenizer']}"
                )
            vocab.tokenizer = tokenizer

        return vocab

    @classmethod
    def build(
        cls,
        corpus: List[str],  # Expect a list of string
        tokenizer: Tokenizer,
        **kwargs,
    ):
        vocab = cls(
            VocabConfig(
                min_freq=kwargs.get("min_freq", 1),
                pad_tok=kwargs.get("pad_tok", "<PAD>"),
                unk_tok=kwargs.get("unk_tok", "<UNK>"),
                sep_tok=kwargs.get("sep_tok", "<SEP>"),
                bos_tok=kwargs.get("bos_tok", "<BOS>"),
                mask_tok=kwargs.get("mask_tok"),
                cls_tok=kwargs.get("cls_tok"),
                resv_tok=kwargs.get("resv_tok"),
            )
        )

        for token in vocab.vocab_config.special_tokens:
            if token and token not in vocab._token_to_idx:
                idx = len(vocab._idx_to_token)
                vocab._idx_to_token.append(token)
                vocab._token_to_idx[token] = idx

        token_counts = defaultdict(int)
        print("Tokenizing corpus and counting frequencies...")
        for idx, text_sample in progress_bar(
            enumerate(corpus), unit=" sentence", unit_scale=True, total=len(corpus)
        ):
            try:
                tokens = tokenizer(text_sample)
                for token in tokens:
                    token_counts[token] += 1
            except Exception as e:
                DEFAULT_LOGGER.warning(
                    f"Tokenizer failed on sample {idx} with error: {e}"
                )

        # Add tokens from corpus
        sorted_tokens = sorted(
            token_counts.items(), key=lambda item: item[1], reverse=True
        )

        min_freq = vocab.vocab_config.min_freq
        for token, count in sorted_tokens:
            if count >= min_freq and token not in vocab._token_to_idx:
                idx = len(vocab._idx_to_token)
                vocab._idx_to_token.append(token)
                vocab._token_to_idx[token] = idx

        DEFAULT_LOGGER.info(
            f"Vocabulary built: {vocab.vocab_size} unique tokens (including special) from {len(corpus)} sentences."
        )

        vocab.tokenizer = tokenizer

        return vocab

    # [[ Private method ]]
    def _convert_token_to_id(self, token: str) -> int:
        if len(self._token_to_idx) == 0:
            raise ValueError("Vocabulary not built yet.")
        return self._token_to_idx.get(token, self.unk_id)

    def _convert_id_to_token(self, idx: int) -> str:
        if len(self._idx_to_token) == 0:
            raise ValueError("Vocabulary not built yet.")

        if 0 <= idx < len(self._idx_to_token):
            return self._idx_to_token[idx]
        else:
            # Raise error instead of returning None implicitly
            raise IndexError(f"Index {idx} out of vocabulary range.")

    # [[ Public method ]]
    def save(self, path: Union[str, Path]):
        """Save to file

        Args:
            file (_type_): _description_
        """
        if len(self._token_to_idx) == 0 or len(self._idx_to_token) == 0:
            raise ValueError("Vocabulary not built yet.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": self.vocab_config.model_dump(),
            "tokenizer": self.tokenizer.__class__.__name__,
            "token_to_idx": self._token_to_idx,
            "idx_to_token": self._idx_to_token,
        }

        DEFAULT_LOGGER.info(f"Save vocabulary to path {path} with ...")

        with open(path, "w+", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @property
    def vocab_size(self) -> int:
        return len(self.get_vocab())

    def get_vocab(self) -> List[int]:
        if len(self._token_to_idx) == 0 or len(self._idx_to_token) == 0:
            raise ValueError("Vocabulary not built yet.")
        return self.idx_to_token

    @property
    def unk_id(self) -> int:
        return self.vocab_config.special_token_idx(self.vocab_config.unk_tok)

    @property
    def pad_id(self) -> int:
        return self.vocab_config.special_token_idx(self.vocab_config.pad_tok)

    @property
    def bos_id(self) -> int:
        return self.vocab_config.special_token_idx(self.vocab_config.bos_tok)

    @property
    def sep_id(self) -> int:
        return self.vocab_config.special_token_idx(self.vocab_config.sep_tok)

    @property
    def sep(self) -> int:
        return self.vocab_config.special_token_idx(self.vocab_config.sep_tok)

    @property
    def unk_tok(self) -> str:
        return self.vocab_config.unk_tok

    @property
    def bos_toks(self) -> List[str]:
        return [self.vocab_config.bos_tok]

    @property
    def sep_toks(self) -> List[str]:
        return [self.vocab_config.sep_tok]

    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def idx_to_token(self):
        return self._idx_to_token

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self._convert_token_to_id(token) for token in tokens]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self._convert_id_to_token(id) for id in ids]

    # Convenient method
    def __len__(self) -> int:
        return self.vocab_size
