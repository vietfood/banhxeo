import itertools
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np
from pydantic import model_validator
from typing_extensions import Self

from banhxeo.core.vocabulary import Vocabulary
from banhxeo.model.base import GENERATE_LOOP_UPPER_BOUND, BaseLanguageModel, ModelConfig
from banhxeo.model.config import GenerateConfig
from banhxeo.utils import progress_bar
from banhxeo.utils.logging import DEFAULT_LOGGER


@dataclass
class NgramTrieNode:
    id: int
    children: Dict[int, "NgramTrieNode"] = field(
        default_factory=dict
    )  # To store child nodes: {'word_id': TrieNode()}
    count: int = 0  # To store the count of the n-gram ending at this node
    is_end_of_ngram: bool = False  # Flag to mark the end of a complete n-gram


class NGramConfig(ModelConfig):
    n: int
    smoothing: Union[bool, str] = False
    k: Optional[float] = None

    @model_validator(mode="after")
    def check_smoothing(self) -> Self:
        if isinstance(self.smoothing, str):
            supported = ["add_k", "laplace", "none"]
            if self.smoothing not in supported:
                raise ValueError(f"Smoothing is not one of {supported=}")
            if self.smoothing == "add_k" and self.k is None:
                raise ValueError(
                    "If smoothing is add_k, you must provide value for k parameter"
                )
        elif isinstance(self.smoothing, bool):
            self.smoothing = "laplace" if self.smoothing else "none"
        return self

    @model_validator(mode="after")
    def check_n(self) -> Self:
        if self.n < 1:
            raise ValueError("N must be at least 1 for N-gram model")
        return self


class NGram(BaseLanguageModel):
    ConfigClass = NGramConfig

    def __init__(
        self,
        vocab: Vocabulary,
        n: int = 2,
        smoothing: Union[bool, str] = False,
        k: Optional[int] = None,
    ):
        super().__init__(
            vocab=vocab, model_config=NGramConfig(n=n, smoothing=smoothing, k=k)
        )
        self.config: NGramConfig

        self.vocab = vocab
        self.root = NgramTrieNode(id=-1)
        self.prefix_root = NgramTrieNode(id=-1)
        self.total_tokens = 0  # for unigram

    def _add_tokens(self, root: NgramTrieNode, tokens: list[str], n: int):
        if not tokens:
            return

        input_ids = self.vocab.tokens_to_ids(tokens)

        for i in range(len(input_ids) - n + 1):
            ngram_tokens = input_ids[i : i + n]
            self._add_ngram_ids(root, ngram_tokens)

    def _add_ngram_ids(self, root: NgramTrieNode, input_ids: list[int]):
        current_node = root

        for token in input_ids:
            if token not in current_node.children:
                current_node.children[token] = NgramTrieNode(id=token)
            current_node = current_node.children[token]

        current_node.count += 1
        current_node.is_end_of_ngram = True  # Mark that a full n-gram ends here

    def _get_ngram_count_by_ids(self, root: NgramTrieNode, ngram_ids: list[int]):
        current_node = root
        for token_id in ngram_ids:
            if token_id not in current_node.children:
                return 0
            current_node = current_node.children[token_id]
        return current_node.count  # Assuming count is only for full n-grams ending here

    def _get_probability(self, token: str, context_ids: list[int]):
        if self.vocab is None or self._is_trained_or_fitted:
            raise ValueError("N-gram model hasn't been fitted yet or vocab is missing.")

        n = self.config.n  # type: ignore

        # Convert context and token to IDs
        token_id = self.vocab._convert_token_to_id(token)

        if (
            token_id == self.vocab.unk_id and token != self.vocab.unk_tok
        ):  # Handle Out Of Vocabulary (OOV) token
            # Smoothing should take care of that
            DEFAULT_LOGGER.warning(
                f"Current {token=} isn't appeared in vocabulary, fallback to unknown token {self.vocab.unk_tok}"
            )

        # Adjust context to be n-1
        if len(context_ids) > n - 1:
            context_ids = context_ids[-(n - 1) :]

        # Pad context with BOS if it's shorter than n-1 (for consistency with how they were added in fit)
        num_bos_needed = (n - 1) - len(context_ids)
        if num_bos_needed > 0 and n > 1:
            context_ids = ([self.vocab.bos_id] * num_bos_needed) + context_ids

        # Numerator: count of (context_ids, token_id)
        full_ngram_for_query_ids = context_ids + [token_id]
        if (
            len(full_ngram_for_query_ids) > n
        ):  # Should only happen if initial prompt is too long
            full_ngram_for_query_ids = full_ngram_for_query_ids[-n:]

        numerator = self._get_ngram_count_by_ids(self.root, full_ngram_for_query_ids)

        # Denominator
        if n > 1:
            # The context for the denominator should be exactly n-1 tokens
            context_for_denominator_ids = full_ngram_for_query_ids[
                :-1
            ]  # This is the (n-1)gram prefix
            denominator = self._get_ngram_count_by_ids(
                self.prefix_root, context_for_denominator_ids
            )
        else:  # Unigram
            denominator = self.total_tokens

        smoothing = self.config.smoothing  # type: ignore
        k_smooth = self.config.k  # type: ignore

        vocab_size_for_smoothing = len(self.vocab)  # Use len(vocab) for V

        if smoothing == "add_k" and k_smooth is not None:
            numerator += k_smooth
            denominator += vocab_size_for_smoothing * k_smooth
        elif smoothing == "laplace":
            numerator += 1
            denominator += vocab_size_for_smoothing

        if denominator == 0:
            DEFAULT_LOGGER.warning(
                f"Denominator is zero for token '{token}' with context {str(self.vocab.ids_to_tokens(context_ids))}."
                f"Fallback to small probability. You should use smoothing to avoid this"
            )
            return 1e-9  # Return small probability

        return numerator / denominator

    def fit(self, corpus: list[str]):
        n = self.config.n  # type: ignore

        DEFAULT_LOGGER.debug(f"Counting {n}-grams...")
        for sentence in progress_bar(
            corpus, total=len(corpus), unit="sentence", unit_scale=True
        ):
            # tokenize input
            tokens = self.vocab.tokenizer.tokenize(sentence)  # type: ignore

            # add <s> (or <BOS>) and </s> (or <SEP>)
            num_bos = n - 1 if n > 1 else 1
            tokens = (
                (  # https://stackoverflow.com/questions/24225072/repeating-elements-of-a-list-n-times
                    list(
                        itertools.chain.from_iterable(
                            itertools.repeat(x, num_bos) for x in self.vocab.bos_toks
                        )
                    )
                )
                + tokens
                + self.vocab.sep_toks
            )
            self.total_tokens = self.total_tokens + len(tokens)

            self._add_tokens(self.root, tokens, n)
            if n > 1:  # We don't need prefix for unigram
                self._add_tokens(self.prefix_root, tokens, n - 1)

        self._is_trained_or_fitted = True

    def _get_next_token_probabilities(
        self, current_context_ids: list[int]
    ) -> np.ndarray:
        n = self.config.n  # type: ignore
        probs = np.zeros(len(self.vocab), dtype=np.float32)  # Use float32

        # Find the context node in the prefix_root (for denominator count)
        denominator_context_node_count = 0
        if n > 1:
            denominator_context_node_count = self._get_ngram_count_by_ids(
                self.prefix_root, current_context_ids
            )
        else:
            denominator_context_node_count = self.total_tokens

        # Traverse main Trie with current_context_ids to find possible next tokens
        context_node_in_main_trie = self.root
        if n > 1:
            for token_id in current_context_ids:
                if token_id not in context_node_in_main_trie.children:
                    context_node_in_main_trie = None  # Context not found
                    break
                context_node_in_main_trie = context_node_in_main_trie.children[token_id]

        # Iterate through all vocab tokens to assign probabilities
        for next_token_id in range(len(self.vocab)):
            numerator_ngram_count = 0
            if (
                context_node_in_main_trie
                and next_token_id in context_node_in_main_trie.children
            ):
                numerator_ngram_count = context_node_in_main_trie.children[
                    next_token_id
                ].count

            # Apply smoothing
            current_numerator = numerator_ngram_count
            current_denominator = denominator_context_node_count

            smoothing = self.config.smoothing  # type: ignore
            k_smooth = self.config.k  # type: ignore
            vocab_size_for_smoothing = len(self.vocab)

            if smoothing == "add_k" and k_smooth is not None:
                current_numerator += k_smooth
                current_denominator += vocab_size_for_smoothing * k_smooth
            elif smoothing == "laplace":
                current_numerator += 1
                current_denominator += vocab_size_for_smoothing

            if current_denominator == 0:
                probs[next_token_id] = 1e-9 / len(self.vocab)
            else:
                probs[next_token_id] = current_numerator / current_denominator

        # Normalize probabilities
        sum_probs = np.sum(probs)
        if sum_probs > 0:
            probs /= sum_probs
        else:
            probs = np.ones(len(self.vocab)) / len(self.vocab)
        return probs

    def generate_sequence(
        self,
        prompt: str,
        sampling: str = "greedy",
        max_length: Optional[int] = 20,
        **kwargs,
    ) -> str:
        config = GenerateConfig(
            sampling=sampling,
            max_length=max_length,
            k=kwargs.get("k"),
            p=kwargs.get("p"),
            temp=kwargs.get("temp"),
        )

        if self.vocab is None:
            raise ValueError("N-gram model haven't fitted yet")

        n = self.config.n  # type: ignore

        prompt_tokens = self.vocab.tokenizer.tokenize(prompt)  # type: ignore
        current_context_ids = self.vocab.tokens_to_ids(prompt_tokens)

        if n > 1:
            if len(current_context_ids) >= (n - 1):
                current_context_ids = current_context_ids[-(n - 1) :]
            else:  # Shorter than n-1, pad with BOS
                num_bos_needed = (n - 1) - len(current_context_ids)
                current_context_ids = (
                    [self.vocab.bos_id] * num_bos_needed
                ) + current_context_ids
        else:  # n=1
            current_context_ids = []

        output_ids = []

        for _ in range(config.max_length or GENERATE_LOOP_UPPER_BOUND):
            if output_ids and output_ids[-1] == self.vocab.sep_id:  # Check for EOS/SEP
                break
            if config.max_length is not None and len(output_ids) >= config.max_length:
                break

            token_probs = self._get_next_token_probabilities(current_context_ids)

            if config.sampling == "greedy":
                next_token_id = np.argmax(token_probs).item()
            elif config.sampling == "top_k" and config.k is not None:
                if config.k >= len(token_probs):
                    indices_to_remove = []
                else:
                    indices_to_remove = (
                        token_probs < np.partition(token_probs, -config.k)[-config.k]
                    )
                token_probs[indices_to_remove] = 0
                next_token_id = np.random.choice(len(self.vocab), p=token_probs)
            else:
                raise NotImplementedError(
                    f"Sampling method '{config.sampling}' not implemented."
                )

            output_ids.append(next_token_id)

            # Update context for the next iteration
            if n > 1:
                temp_full_sequence = (
                    self.vocab.tokens_to_ids(prompt_tokens) + output_ids
                )
                current_context_ids = temp_full_sequence[-(n - 1) :]
                # BOS padding if still too short (shouldn't happen after first few steps if prompt was short)
                num_bos_needed_update = (n - 1) - len(current_context_ids)
                if num_bos_needed_update > 0:
                    current_context_ids = (
                        [self.vocab.bos_id] * num_bos_needed_update
                    ) + current_context_ids

        return self.vocab.tokenizer.detokenize(self.vocab.ids_to_tokens(output_ids))  # type: ignore
