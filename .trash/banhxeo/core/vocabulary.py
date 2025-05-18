import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel, field_validator
from tqdm import tqdm

from banhxeo.core.tokenizer import Tokenizer
from banhxeo.utils.logging import DEFAULT_LOGGER


class VocabConfig(BaseModel):
    min_freq: int = 1
    pad_tok: str = "<PAD>"  # Special token shouldn't be empty
    unk_tok: str = "<UNK>"
    sos_tok: str = "<SOS>"
    eos_tok: str = "<EOS>"

    @field_validator("min_freq", mode="before")
    @classmethod
    def check_positive(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Min frequencies cannot be 0")
        return value

    def special_tokens(self) -> list[str]:
        return [self.pad_tok, self.unk_tok, self.sos_tok, self.eos_tok]


DEFAULT_VOCAB_CONFIG = VocabConfig()


class Vocabulary:
    """Manages token<=>ID mapping, special tokens."""

    def __init__(self, vocab_config: Optional[VocabConfig]):
        self.vocab_config = vocab_config if vocab_config else DEFAULT_VOCAB_CONFIG

        self._vocab = None

        self.idx_to_token = []
        self.token_to_idx = {}

        self.pad_idx: int = -1
        self.unk_idx: int = -1
        self.sos_idx: int = -1
        self.eos_idx: int = -1

    @classmethod
    def load(cls, path: str):
        """Loading pre-trained vocabulary from file

        Args:
            path (_type_): _description_
        """
        # TODO:

    @classmethod
    def build(
        cls,
        corpus: List[str],  # Expect a list of string
        tokenizer: Tokenizer,
        min_freq: int = 1,
        pad_tok: str = "<PAD>",  # Special token shouldn't be empty
        unk_tok: str = "<UNK>",
        sos_tok: str = "<SOS>",
        eos_tok: str = "<EOS>",
    ):
        vocab = cls(
            VocabConfig(
                min_freq=min_freq,
                pad_tok=pad_tok,
                unk_tok=unk_tok,
                sos_tok=sos_tok,
                eos_tok=eos_tok,
            )
        )

        for token in vocab.vocab_config.special_tokens():
            if token and token not in vocab.token_to_idx:
                idx = len(vocab.idx_to_token)
                vocab.idx_to_token.append(token)
                vocab.token_to_idx[token] = idx

        vocab.pad_idx = vocab.token_to_idx.get(vocab.vocab_config.pad_tok, -1)
        vocab.unk_idx = vocab.token_to_idx.get(vocab.vocab_config.unk_tok, -1)
        vocab.sos_idx = vocab.token_to_idx.get(vocab.vocab_config.sos_tok, -1)
        vocab.eos_idx = vocab.token_to_idx.get(vocab.vocab_config.eos_tok, -1)

        token_counts = defaultdict(int)
        print("Tokenizing corpus and counting frequencies...")
        for idx, text_sample in tqdm(
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
            if count >= min_freq and token not in vocab.token_to_idx:
                idx = len(vocab.idx_to_token)
                vocab.idx_to_token.append(token)
                vocab.token_to_idx[token] = idx

        vocab._vocab = {
            "token_to_idx": vocab.token_to_idx,
            "idx_to_token": vocab.idx_to_token,
        }

        DEFAULT_LOGGER.info(
            f"Vocabulary built: {vocab.__vocab_size()} unique tokens (including special) from {len(corpus)} sentences."
        )

        return vocab

    # [[ Private method ]]
    def __token_to_id(self, token: str) -> int:
        if self._vocab is None or "token_to_idx" not in self._vocab:
            raise ValueError("Vocabulary not built yet.")

        return self._vocab["token_to_idx"].get(token, self.unk_idx)

    def __id_to_token(self, idx: int) -> str:
        if self._vocab is None or "idx_to_token" not in self._vocab:
            raise ValueError("Vocabulary not built yet.")

        if 0 <= idx < len(self._vocab["idx_to_token"]):
            return self._vocab["idx_to_token"][idx]
        else:
            # Raise error instead of returning None implicitly
            raise IndexError(f"Index {idx} out of vocabulary range.")

    # [[ Public method ]]
    def save(self, path: Union[str, Path], type: str = "json"):
        """Save to file

        Args:
            file (_type_): _description_
        """
        current_support_type = ["json", "pickle"]
        if (type.strip().lower()) not in ["json", "pickle"]:
            raise ValueError(
                f"Only support save file type in {str(current_support_type)}"
            )

        if self._vocab is None:
            raise ValueError("Vocabulary not built yet.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "specials": self.vocab_config.special_tokens(),
            "token_to_idx": self._vocab["token_to_idx"],
            "idx_to_token": self._vocab["idx_to_token"],
        }

        DEFAULT_LOGGER.info(f"Save vocabulary to path {path} with type {type} ...")

        if type == "json":
            with open(path, "w+", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif type == "pickle":
            with open(path, "wb+", encoding="utf-8") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __vocab_size(self) -> int:
        if self._vocab is None or "idx_to_token" not in self._vocab:
            raise ValueError("Vocabulary is not built yet.")
        return len(self._vocab["idx_to_token"])

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.__token_to_id(token) for token in tokens]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.__id_to_token(id) for id in ids]

    @property
    def tokens_ids(self):
        if self._vocab is None or "token_to_idx" not in self._vocab:
            raise ValueError("Vocabulary is not built yet.")
        return self._vocab["token_to_idx"]

    @property
    def ids_tokens(self):
        if self._vocab is None or "idx_to_token" not in self._vocab:
            raise ValueError("Vocabulary is not built yet.")
        return self._vocab["idx_to_token"]

    @property
    def eos_tok(self) -> str:
        return self.vocab_config.eos_tok

    @property
    def sos_tok(self) -> str:
        return self.vocab_config.sos_tok

    @property
    def pad_id(self) -> int:
        if self.pad_idx == -1:
            raise ValueError(
                "Vocabulary is not built yet or there is something wrong with building/loading vocabulary"
            )
        return self.pad_idx

    @property
    def unk_id(self) -> int:
        if self.unk_idx == -1:
            raise ValueError(
                "Vocabulary is not built yet or there is something wrong with building/loading vocabulary"
            )
        return self.unk_idx

    # Covenient method
    def __len__(self) -> int:
        return self.__vocab_size()

    def __getitem__(self, index):
        return self.__id_to_token(index)
