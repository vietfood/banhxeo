from typing import Dict

import torch

from jaxtyping import Integer
from torch import nn
from torch.utils.data import Dataset

from banhxeo.core import Tokenizer, Vocabulary
from banhxeo.data.base import BaseTextDataset
from banhxeo.model.neural import NeuralLanguageModel, NeuralModelConfig
from banhxeo.utils import progress_bar


class Word2VecDataset(Dataset):
    def __init__(
        self,
        text_dataset: BaseTextDataset,
        window_size: int,
        k_negative_samples: int,
        alpha: float,
        tokenizer: Tokenizer,
        vocab: Vocabulary,
    ):
        self.text_dataset = text_dataset
        self.window_size = window_size
        self.k_negative_samples = k_negative_samples
        self.alpha = alpha
        self.tokenizer = tokenizer
        self.vocab = vocab

        self.data = self._build_word2vec_data()

    def _build_word2vec_data(self):
        # Build word counts frequency for negative sampling
        # Create a list of word indices for negative sampling based on frequency
        # (P(w) = count(w)^alpha)
        word_freqs = torch.tensor(
            [
                self.vocab.get_word_counts()[word] ** self.alpha
                for word in self.vocab.token_to_idx
            ],
            dtype=torch.float,
            device="cpu",
        )  # Let the tensor on CPU for efficiency
        word_dist = word_freqs / word_freqs.sum()

        training_data = list()
        for sentence in progress_bar(
            self.text_dataset.get_all_texts(),
            desc=f"Building dataset for Word2Vec from dataset {self.text_dataset.config.name}",
            unit="sentence",
            unit_scale=True,
        ):
            tokens = self.tokenizer.tokenize(sentence)
            sent_ids = self.vocab.tokens_to_ids(tokens)
            for i, target_word_id in enumerate(sent_ids):
                for j in range(
                    max(0, i - self.window_size),
                    min(len(sent_ids), i + self.window_size + 1),
                ):
                    if i == j:  # Skip the target word itself
                        continue
                    # Get context word based on window size
                    # This is positive sample so it is 1
                    context_word_idx = sent_ids[j]
                    training_data.append((target_word_id, context_word_idx, 1))

                    # For each context word, we sampling k negative samples
                    neg_samples_count = 0
                    while neg_samples_count < self.k_negative_samples:
                        # Sample using torch.multinomial
                        negative_sample_indices = torch.multinomial(
                            word_dist, 1, replacement=True
                        )
                        negative_sample_id = negative_sample_indices[0].item()

                        if (
                            negative_sample_id != context_word_idx
                            and negative_sample_id != target_word_id
                        ):  # Ensure negative sample is not context word or target word
                            training_data.append(
                                (target_word_id, negative_sample_id, 0)
                            )  # Negative sample
                            neg_samples_count += 1
        return training_data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw_data = self.data[idx]
        return {
            "target_words": torch.tensor(raw_data[0], dtype=torch.long),
            "context_words": torch.tensor(raw_data[1], dtype=torch.long),
            "labels": torch.tensor(raw_data[2], dtype=torch.long),
        }


class Word2Vec(NeuralLanguageModel):
    """An implementation of Word2Vec with SGNS (Skip-gram with negative sampling) variation."""

    def __init__(self, model_config: NeuralModelConfig, vocab: Vocabulary):
        super().__init__(model_config, vocab)
        self.config: NeuralModelConfig

        self.target_embedding = nn.Embedding(
            num_embeddings=self.vocab.vocab_size,
            embedding_dim=self.config.embedding_dim,
            padding_idx=self.vocab.pad_id,
        )

        self.context_embedding = nn.Embedding(
            num_embeddings=self.vocab.vocab_size,
            embedding_dim=self.config.embedding_dim,
            padding_idx=self.vocab.pad_id,
        )

    def forward(
        self,
        target_words: Integer[torch.Tensor, "batch"],  # noqa: F821
        context_words: Integer[torch.Tensor, "batch"],  # noqa: F821
    ) -> Dict[str, torch.Tensor]:
        context_embeds = self.context_embedding(context_words)
        target_embeds = self.target_embedding(target_words)

        # Dot product between two "matrix" with shape (batch, embed_dim)
        # We want to get dot product between embed_dim dimension only
        logits = torch.einsum("be,be->b", target_embeds, context_embeds)

        return {"logits": logits}

    @staticmethod
    def prepare_data(
        data: BaseTextDataset,
        tokenizer: Tokenizer,
        vocab: Vocabulary,
        window_size: int = 2,
        k_negative_samples: int = 3,
        **kwargs,
    ):
        return Word2VecDataset(
            data,
            window_size=window_size,
            k_negative_samples=k_negative_samples,
            alpha=kwargs.get("alpha", 0.75),
            tokenizer=tokenizer,
            vocab=vocab,
        )
