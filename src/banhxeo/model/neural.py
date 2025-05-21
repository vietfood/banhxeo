import json
from abc import abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Integer

from banhxeo import CPU_DEVICE, GPU_DEVICE
from banhxeo.core.vocabulary import Vocabulary
from banhxeo.model.base import BaseLanguageModel
from banhxeo.model.config import GenerateConfig, ModelConfig
from banhxeo.utils.logging import DEFAULT_LOGGER


class NeuralModelConfig(ModelConfig):
    embedding_dim: int


class NeuralLanguageModel(BaseLanguageModel, nn.Module):
    def __init__(self, model_config: NeuralModelConfig, vocab: Vocabulary):
        BaseLanguageModel.__init__(self, model_config, vocab)  # Call LM init first
        nn.Module.__init__(self)  # Then nn.Module init

        self.downstream_heads = nn.ModuleDict()  # Downstream task

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def summary(self):
        super().summary()
        print(self)

    @abstractmethod
    def forward(
        self,
        input_ids: Integer[torch.Tensor, "batch seq"],  # noqa: F722
        attention_mask: Optional[Integer[torch.Tensor, "batch seq"]] = None,  # noqa: F722
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the its own forward pass."
        )

    def generate_sequence(
        self,
        prompt: str,  # Takes raw string prompt
        generate_config: Optional[GenerateConfig] = None,
        **kwargs,  # For tokenizer config during prompt processing
    ) -> str:
        """
        Only applicable to autoregressive models.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the generation loop."
        )

    def attach_downstream_head(self, head_name: str, head_module: nn.Module):
        """Attaches a task-specific head to the model."""
        if head_name in self.downstream_heads:
            DEFAULT_LOGGER.warning(f"Replacing existing downstream head: {head_name}")
        self.downstream_heads[head_name] = head_module
        DEFAULT_LOGGER.info(
            f"Attached downstream head: {head_name} ({head_module.__class__.__name__})"
        )

    def get_downstream_head_output(
        self, head_name: str, base_model_output: Dict[str, torch.Tensor], **head_kwargs
    ) -> torch.Tensor:
        """
        Passes the base model's output through a specified downstream head.
        Args:
            head_name: Name of the head to use.
            base_model_output: Output dictionary from the base model's forward pass.
                               Expected to contain keys like 'last_hidden_state', 'pooled_output', etc.
            head_kwargs: Additional arguments to pass to the head's forward method.
        Returns:
            Output tensor from the downstream head.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the generation loop."
        )

    # --- Saving and Loading ---
    def save_model(self, save_path: Union[str, Path]):
        save_path = Path(save_path)
        save_directory = save_path.parent
        save_directory.mkdir(parents=True, exist_ok=True)

        # 1. Save model state_dict
        torch.save(self.state_dict(), save_path)

        # 2. Save model config
        config_path = save_directory / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.model_dump(), f, indent=2)

        # 3. Save vocabulary (if you want to bundle it)
        #    This assumes vocab has a save method
        if self.vocab and hasattr(self.vocab, "save"):
            self.vocab.save(save_directory / "vocabulary.json")  # Or a subfolder

        DEFAULT_LOGGER.info(f"Model saved to {save_directory}")

    @classmethod
    def load_model(cls, load_directory: Union[str, Path]):
        raise NotImplementedError()

    def to_gpu(self):
        if GPU_DEVICE is None:
            DEFAULT_LOGGER.warning(
                "No GPU detected (CUDA or MPS). Model remains on CPU."
            )
            return self
        self.to(GPU_DEVICE)
        DEFAULT_LOGGER.info(f"Model moved to {GPU_DEVICE}.")
        return self  # for chaining

    def to_cpu(self):
        self.to(CPU_DEVICE)
        DEFAULT_LOGGER.info(f"Model moved to {CPU_DEVICE}.")
        return self
