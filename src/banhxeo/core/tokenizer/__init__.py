from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

from banhxeo.core.tokenizer.config import ProcessConfig
from banhxeo.core.tokenizer.decoder import DECODER_FACTORY, Decoder
from banhxeo.core.tokenizer.model import MODEL_FACTORY, TokenizerModel
from banhxeo.core.tokenizer.normalizers import (
    NORMALIZER_FACTORY,
    NormalizedString,
    Normalizer,
)
from banhxeo.core.tokenizer.post_processor import POST_PROCESSOR_FACTORY, PostProcessor
from banhxeo.core.tokenizer.pre_tokenizer import (
    PRE_TOKENIZER_FACTORY,
    PreTokenizedString,
    PreTokenizer,
    Split,
)
from banhxeo.utils import progress_bar
from banhxeo.utils.file import load_json


class Tokenizer:
    def __init__(
        self,
        normalizer: Normalizer,
        pre_tokenizer: PreTokenizer,
        model: TokenizerModel,
        post_processor: PostProcessor,
        decoder: Decoder,
    ):
        self.normalizer = normalizer
        self.pre_tokenizer = pre_tokenizer
        self.model = model
        self.post_processor = post_processor
        self.decoder = decoder

    @property
    def vocab_size(self) -> int:
        if (vocab := self.model.get_vocab()) is None:
            raise ValueError("Vocabulary is None")
        return len(vocab)

    @property
    def special_tokens(self):
        return self.model.special_tokens

    def __call__(
        self,
        texts: Union[str, List[str]],
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
        if isinstance(texts, str):
            texts = [texts]

        pre_tokenized_strs = []
        for text in texts:
            # Step 1: Normalized string
            normalized_string = NormalizedString.from_str(text)
            normalized_string = self.normalizer.normalize(normalized_string)

            # Step 2: Pre Tokenize normalized string
            pre_tokenized_str = self.pre_tokenizer.pre_tokenize(
                PreTokenizedString(splits=[Split(normalized=normalized_string)])
            )

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
                    key: jnp.array(value, dtype=jnp.int32)
                    for key, value in post_process_result.items()
                }
            case "np":
                import numpy as np

                return {
                    key: np.array(value, dtype=np.int32)
                    for key, value in post_process_result.items()
                }
            case _:
                return post_process_result

    def encode(
        self,
        text: str,
        return_tensors: Optional[Literal["jax", "np"]] = "jax",
        **kwargs,
    ) -> Any:
        results = self.__call__([text], return_tensors, **kwargs)
        return results["input_ids"][0]

    def batch_encode(
        self,
        texts: List[str],
        return_tensors: Optional[Literal["jax", "np"]] = "jax",
        **kwargs,
    ) -> Any:
        results = self.__call__(texts, return_tensors, **kwargs)
        return results["input_ids"]

    def decode(self, token_ids: List[int], **kwargs):
        return self.batch_decode([token_ids])[0]

    def batch_decode(
        self,
        batch_ids: List[List[int]],
        skip_special_tokens: bool = True,  # A common and useful parameter
        **kwargs,
    ):
        batch_tokens = [self.model.detokenize(token_ids) for token_ids in batch_ids]

        cleaned_batch_tokens = []
        special_tokens_to_skip = set(self.model.special_tokens.tokens)

        for tokens in batch_tokens:
            if skip_special_tokens:
                cleaned_tokens = [
                    tok for tok in tokens if tok not in special_tokens_to_skip
                ]
                cleaned_batch_tokens.append(cleaned_tokens)
            else:
                cleaned_batch_tokens.append(tokens)

        batch_str = [
            self.decoder.decode(tokens, **kwargs) for tokens in cleaned_batch_tokens
        ]
        return batch_str

    def train(self, corpus: Iterable[str], progress: bool = True):
        pre_tokenized_corpus = []
        for text in progress_bar(
            corpus, desc="Pre-tokenizing text", disable=not progress
        ):
            normalized_text = self.normalizer.normalize(NormalizedString.from_str(text))
            pre_tokenized_text = self.pre_tokenizer.pre_tokenize(
                PreTokenizedString(splits=[Split(normalized=normalized_text)])
            )
            pre_tokenized_corpus.append(pre_tokenized_text)

        self.model.train(corpus=pre_tokenized_corpus, progress=progress)

    @classmethod
    def from_pretrained(cls, path: Path | str):
        # TODO
        ...
