from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from banhxeo.core.tokenizer import ProcessConfig, SpecialTokens
from banhxeo.core.tokenizer.pre_tokenizer import PreTokenizedString


class PostProcessor(ABC):
    @abstractmethod
    def process(
        self, tokenized_str: PreTokenizedString, config: ProcessConfig
    ) -> Dict[str, Any]:
        """
        Turn PreTokenizedString (with tokens) into dictionary
        {'input_ids': [...], 'attention_mask': [...]}
        """

    @abstractmethod
    def process_batch(
        self, tokenized_strs: List[PreTokenizedString], config: ProcessConfig
    ) -> Dict[str, Any]:
        """
        Batch version of process method
        """


class GeneralPostProcessor(PostProcessor):
    def __init__(self, special_tokens: SpecialTokens):
        self.special_tokens = special_tokens

    def process(
        self, tokenized_str: PreTokenizedString, config: ProcessConfig
    ) -> Dict[str, Any]:
        batch_result = self.process_batch([tokenized_str], config=config)
        return {key: value[0] for key, value in batch_result.items()}

    @abstractmethod
    def add_tokens(self, token_ids) -> List[int]:
        """
        How PostProcessor should implement add specials tokens
        Example:
        - For BERT-style, we add "SEP" and "CLS".
        - For GPT-style, we add "BOS" and "EOS".
        """

    def process_batch(
        self, tokenized_strs: List[PreTokenizedString], config: ProcessConfig
    ) -> Dict[str, Any]:
        batch_ids = []

        for tokenized_str in tokenized_strs:
            token_ids = [
                token.id for split in tokenized_str.splits for token in split.tokens  # type: ignore
            ]

            if (
                config.truncation
                and config.max_length is not None
                and len(token_ids) > config.max_length - 2
            ):
                token_ids = token_ids[: config.max_length - 2]

            if config.add_special_tokens:
                token_ids = self.add_tokens(token_ids)

            batch_ids.append(token_ids)

        batch_longest = max([len(ids) for ids in batch_ids]) if batch_ids else 0

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

        padding_ids = []
        attention_masks = []
        for token_ids in batch_ids:
            pad_length = max_seq_len - len(token_ids)
            attention_mask = [1] * len(token_ids)
            if pad_length > 0:
                new_pad_ids = [
                    self.special_tokens.special_token_idx(self.special_tokens.pad_tok)
                ] * pad_length
                new_pad_mask = [0] * pad_length
                match config.padding_side:
                    case "left":
                        token_ids = new_pad_ids + token_ids
                        attention_mask = new_pad_mask + attention_mask
                    case "right":
                        token_ids += new_pad_ids
                        attention_mask += new_pad_mask
            padding_ids.append(token_ids)
            attention_masks.append(attention_mask)

        return {"input_ids": padding_ids, "attention_masks": attention_masks}


class GPTPostProcessor(GeneralPostProcessor):
    def _add_tokens(self, token_ids) -> List[int]:
        return (
            [self.special_tokens.special_token_idx(self.special_tokens.bos_tok)]
            + token_ids
            + [self.special_tokens.special_token_idx(self.special_tokens.eos_tok)]
        )


class BertPostProcessor(GeneralPostProcessor):
    def _add_tokens(self, token_ids) -> List[int]:
        return (
            [self.special_tokens.special_token_idx(self.special_tokens.cls_tok)]
            + token_ids
            + [self.special_tokens.special_token_idx(self.special_tokens.sep_tok)]
        )
