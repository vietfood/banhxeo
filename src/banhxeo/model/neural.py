from __future__ import annotations

import json

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import torch
import torch.nn as nn

from banhxeo import CPU_DEVICE, GPU_DEVICE
from banhxeo.core.tokenizer import Tokenizer, TokenizerConfig
from banhxeo.core.vocabulary import Vocabulary
from banhxeo.model.base import BaseLanguageModel
from banhxeo.model.config import GenerateConfig, ModelConfig
from banhxeo.utils.logging import DEFAULT_LOGGER


class NeuralModelConfig(ModelConfig):  # noqa: D101
    embedding_dim: int


class NeuralLanguageModel(BaseLanguageModel, nn.Module):
    """Abstract base class for neural network-based language models.

    Extends `BaseLanguageModel` and `torch.nn.Module`. It provides common
    functionalities for neural models, such as device management (CPU/GPU),
    model saving/loading (weights and config), attaching downstream heads,
    and a more detailed summary.

    Subclasses must implement the `forward` method and typically override
    `generate_sequence` if applicable.

    Attributes:
        config (NeuralModelConfig): Configuration specific to neural models,
            inheriting from `ModelConfig` and often adding `embedding_dim`.
        vocab (Vocabulary): The vocabulary used by the model.
        downstream_heads (nn.ModuleDict): A dictionary to hold task-specific
            output layers (heads) that can be attached to the base model.
    """

    def __init__(self, model_config: NeuralModelConfig, vocab: Vocabulary):
        """Initializes the NeuralLanguageModel.

        Args:
            model_config: The configuration object, instance of `NeuralModelConfig`
                or its subclass.
            vocab: The `Vocabulary` instance.
        """
        BaseLanguageModel.__init__(self, model_config, vocab)  # Call LM init first
        nn.Module.__init__(self)  # Then nn.Module init

        self.downstream_heads = nn.ModuleDict()  # Downstream task

    def freeze(self):
        """Freezes all parameters of the model.

        Sets `requires_grad = False` for all parameters, making them
        non-trainable. Useful for feature extraction or fine-tuning only
        a part of the model (e.g., a downstream head).
        """
        for param in self.parameters():
            param.requires_grad = False
        DEFAULT_LOGGER.info(
            f"All parameters in {self.__class__.__name__} have been frozen."
        )

    def unfreeze(self) -> None:
        """Unfreezes all parameters of the model.

        Sets `requires_grad = True` for all parameters, making them trainable.
        """
        for param in self.parameters():
            param.requires_grad = True
        DEFAULT_LOGGER.info(
            f"All parameters in {self.__class__.__name__} have been unfrozen."
        )

    def summary(self) -> None:
        """Prints an enhanced summary of the neural model.

        Includes the summary from `BaseLanguageModel` and also prints the
        PyTorch model structure (layers and parameters).
        """
        super().summary()
        print("\n--- PyTorch Model Structure ---")
        print(self)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(
            f"  Current Device: {next(self.parameters()).device if list(self.parameters()) else 'N/A (No parameters)'}"
        )

    @abstractmethod
    def forward(
        self,
        *args: Any,  # More generic for diverse model inputs
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Defines the computation performed at every call.

        Subclasses must implement this method. It should take tensors as input
        (e.g., `input_ids`, `attention_mask`) and return a dictionary of
        output tensors (e.g., `logits`, `hidden_states`, `loss` if computed).

        Args:
            *args: Variable length argument list for model inputs.
            **kwargs: Arbitrary keyword arguments for model inputs.

        Returns:
            A dictionary where keys are string names of outputs (e.g., "logits",
            "last_hidden_state") and values are the corresponding `torch.Tensor`s.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement its own forward pass."
        )

    def generate_sequence(
        self,
        prompt: str,
        generate_config: Optional[GenerateConfig] = None,
        tokenizer_config: Optional[TokenizerConfig] = None,  # For processing prompt
        **kwargs: Any,  # For tokenizer.encode during prompt processing
    ) -> str:
        """Generates a sequence of text starting from a given prompt.

        This method is typically applicable to autoregressive models (e.g., GPT-like LMs,
        sequence-to-sequence models in generation mode). Non-autoregressive models
        (like MLP classifiers or Word2Vec) should raise `NotImplementedError`.

        The implementation should handle:
        1. Tokenizing the input `prompt` using `self.vocab.tokenizer`.
        2. Iteratively predicting the next token.
        3. Applying sampling strategies specified in `generate_config` (e.g., greedy, top-k).
        4. Stopping generation based on `max_length` or an end-of-sequence token.
        5. Detokenizing the generated token IDs back into a string.

        Args:
            prompt: The initial text string to start generation from.
            generate_config: A `GenerateConfig` object specifying generation
                parameters like `max_length`, `sampling` strategy, `top_k`, etc.
                If None, default generation parameters should be used.
            tokenizer_config: An optional `TokenizerConfig` for encoding the prompt.
                If None, a default sensible configuration should be used (e.g.,
                no padding, no truncation initially unless prompt is too long for model).
            **kwargs: Additional keyword arguments that might be passed to the
                tokenizer's `encode` method when processing the prompt.

        Returns:
            The generated text string (excluding the initial prompt, or including it,
            based on implementation choice).

        Raises:
            NotImplementedError: If the model does not support sequence generation.
            ValueError: If prerequisites for generation (like a tokenizer in vocab) are missing.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support sequence generation, "
            "or it has not been implemented yet."
        )

    def attach_downstream_head(self, head_name: str, head_module: nn.Module) -> None:
        """Attaches a task-specific head to the base model.

        This allows reusing the base model's learned representations for different
        downstream tasks (e.g., classification, token classification) by adding
        a new final layer or set of layers.

        Args:
            head_name: A unique string name for the head. If a head with this
                name already exists, it will be replaced.
            head_module: The `torch.nn.Module` instance representing the head.
                This head will be registered in `self.downstream_heads`.
        """
        if head_name in self.downstream_heads:
            DEFAULT_LOGGER.warning(f"Replacing existing downstream head: '{head_name}'")
        self.downstream_heads[head_name] = head_module
        # Ensure the new head is moved to the same device as the model
        model_device = (
            next(self.parameters()).device if list(self.parameters()) else CPU_DEVICE
        )
        head_module.to(model_device)
        DEFAULT_LOGGER.info(
            f"Attached downstream head: '{head_name}' ({head_module.__class__.__name__}) "
            f"to device {model_device}."
        )

    def get_downstream_head_output(
        self, head_name: str, base_model_output: Dict[str, torch.Tensor], **head_kwargs
    ) -> torch.Tensor:
        """Passes features from the base model's output through a specified downstream head.

        Args:
            head_name: The name of the downstream head to use (must have been
                previously attached via `attach_downstream_head`).
            base_model_output: The dictionary output from the base model's
                `forward()` method. The head might expect specific keys from
                this dictionary (e.g., "last_hidden_state", "pooled_output").
            **head_kwargs: Additional keyword arguments to pass to the
                downstream head's `forward` method.

        Returns:
            The output tensor from the specified downstream head.

        Raises:
            KeyError: If `head_name` is not found in `self.downstream_heads`.
            ValueError: If the `head_module` (from `self.downstream_heads[head_name]`)
                is not a callable `nn.Module` or if its `expected_input_key` (if defined)
                is not in `base_model_output`.
        """
        if head_name not in self.downstream_heads:
            raise KeyError(
                f"Downstream head '{head_name}' not found. "
                f"Available heads: {list(self.downstream_heads.keys())}"
            )

        head_module = self.downstream_heads[head_name]
        if not isinstance(
            head_module, nn.Module
        ):  # Should always be true due to ModuleDict
            raise ValueError(f"Head '{head_name}' is not a valid nn.Module.")

        # Determine which feature from base_model_output to pass to the head
        # Some heads might have an 'expected_input_key' attribute
        feature_to_pass: Optional[torch.Tensor] = None
        if (
            hasattr(head_module, "expected_input_key")
            and head_module.expected_input_key
        ):
            expected_key = head_module.expected_input_key
            if expected_key not in base_model_output:
                raise ValueError(
                    f"Downstream head '{head_name}' expects key '{expected_key}' "
                    f"from base model output, but found keys: {list(base_model_output.keys())}"
                )
            feature_to_pass = base_model_output[expected_key]  # type: ignore
        else:
            raise ValueError(
                "Head Module don't have expected_input_key, cannot feed to downstream head."
            )

        return head_module(feature_to_pass, **head_kwargs)

    # --- Saving and Loading ---
    def save_model(
        self, save_directory: Union[str, Path]
    ) -> None:  # Changed from save_path to save_directory
        """Saves the neural model's state_dict, configuration, and vocabulary.

        The model's `state_dict` is saved to `pytorch_model.bin` (or similar),
        the configuration (`self.config`) to `config.json`, and the vocabulary
        (`self.vocab`) to `vocabulary.json` within the specified `save_directory`.

        Args:
            save_directory: The directory path where the model components will be saved.
                The directory will be created if it doesn't exist.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save model state_dict
        model_save_path = save_dir / f"{self.__class__.__name__}.bin"
        torch.save(self.state_dict(), model_save_path)
        DEFAULT_LOGGER.info(f"Model state_dict saved to {model_save_path}")

        # 2. Save model config (self.config should be a Pydantic model)
        config_path = save_dir / "config.json"
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                # Use model_dump for Pydantic models
                json.dump(self.config.model_dump(mode="json"), f, indent=2)
            DEFAULT_LOGGER.info(f"Model configuration saved to {config_path}")
        except Exception as e:
            DEFAULT_LOGGER.error(
                f"An unexpected error occurred while saving model config: {e}"
            )

        # 3. Save vocabulary
        if self.vocab:
            vocab_path = save_dir / "vocabulary.json"
            try:
                self.vocab.save(vocab_path)
            except Exception as e:
                DEFAULT_LOGGER.error(f"Failed to save vocabulary to {vocab_path}: {e}")

        DEFAULT_LOGGER.info(
            f"Model {self.__class__.__name__} successfully saved to {save_dir}"
        )

    @classmethod
    def load_model(
        cls: Type[NeuralLanguageModel],
        load_directory: Union[str, Path],
        vocab: Optional[Vocabulary] = None,
        tokenizer_for_vocab_load: Optional[
            Tokenizer
        ] = None,  # Needed if vocab is loaded and needs tokenizer
        **model_kwargs: Any,  # For overriding config params or passing to __init__
    ) -> NeuralLanguageModel:
        """Loads a neural model from a saved directory.

        This method reconstructs the model by:
        1. Loading the configuration from `config.json`.
        2. Loading the vocabulary from `vocabulary.json` (if it exists and `vocab` is not provided).
        3. Instantiating the model with the loaded config and vocab.
        4. Loading the saved weights from `pytorch_model.bin` into the model.

        Args:
            cls: The specific `NeuralLanguageModel` subclass to instantiate.
            load_directory: The directory path from which to load the model components.
            vocab: An optional pre-loaded `Vocabulary` instance. If provided,
                loading `vocabulary.json` from the directory is skipped.
            tokenizer_for_vocab_load: A `Tokenizer` instance, required if `vocab`
                is None and `vocabulary.json` needs to be loaded (as `Vocabulary.load`
                requires a tokenizer).
            **model_kwargs: Additional keyword arguments to pass to the model's
                `__init__` method, potentially overriding loaded configuration values.

        Returns:
            An instance of the neural model class, loaded with configuration and weights.

        Raises:
            FileNotFoundError: If `config.json` or `pytorch_model.bin` is missing.
            ValueError: If vocabulary cannot be loaded/provided and is essential.
        """
        load_dir = Path(load_directory)

        # 1. Load model config
        config_path = load_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Configuration file 'config.json' not found in {load_dir}"
            )
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        # Allow overriding loaded config with model_kwargs
        config_dict.update(model_kwargs)

        # Determine which config class to use (subclass might override ConfigClass)
        config_cls = cls.ConfigClass  # type: ignore

        # Forgivingly remove keys from config_dict not in config_cls for Pydantic
        valid_config_keys = config_cls.model_fields.keys()
        filtered_config_dict = {
            k: v for k, v in config_dict.items() if k in valid_config_keys
        }
        unknown_keys = set(config_dict.keys()) - valid_config_keys
        if unknown_keys:
            DEFAULT_LOGGER.warning(
                f"Ignoring unknown keys from loaded config for {cls.__name__}: {unknown_keys}"
            )

        loaded_config = config_cls(**filtered_config_dict)

        # 2. Load or use provided vocabulary
        loaded_vocab_instance = vocab
        if loaded_vocab_instance is None:
            vocab_path = load_dir / "vocabulary.json"
            if vocab_path.is_file():
                if tokenizer_for_vocab_load is None:
                    # Attempt to infer tokenizer type from config if possible, or use a basic one
                    # This part is tricky without knowing the original tokenizer
                    DEFAULT_LOGGER.warning(
                        "tokenizer_for_vocab_load not provided for loading vocabulary.json. "
                        "Attempting to proceed, but ensure compatibility or provide a Tokenizer."
                    )
                    # Fallback: Create a very basic tokenizer for Vocabulary.load
                    # This might not be the original tokenizer used when vocab was saved.
                    from banhxeo.core.tokenizer import Tokenizer

                    tokenizer_for_vocab_load = Tokenizer()

                loaded_vocab_instance = Vocabulary.load(
                    vocab_path, tokenizer=tokenizer_for_vocab_load
                )
                DEFAULT_LOGGER.info(f"Vocabulary loaded from {vocab_path}")
            else:
                # If vocab_size is in config, we might proceed without a full vocab object
                # if model architecture only needs vocab_size.
                if loaded_config.vocab_size is None:
                    raise ValueError(
                        f"Vocabulary not provided and 'vocabulary.json' not found in {load_dir}, "
                        "and vocab_size not in config. Cannot initialize model."
                    )
                DEFAULT_LOGGER.warning(
                    f"'vocabulary.json' not found in {load_dir}. "
                    "Model will be initialized without a Vocabulary object, "
                    f"relying on vocab_size={loaded_config.vocab_size} from config."
                )

        if loaded_vocab_instance is None and loaded_config.vocab_size is None:
            raise ValueError(
                "Vocabulary is required to load the model, but was not provided or found."
            )

        init_args = loaded_config.model_dump()
        model = cls(vocab=loaded_vocab_instance, **init_args)  # type: ignore

        # 4. Load model state_dict
        model_path = load_dir / "pytorch_model.bin"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model weights file 'pytorch_model.bin' not found in {load_dir}"
            )

        state_dict = torch.load(
            model_path, map_location=torch.device("cpu")
        )  # Load to CPU first

        # Handle potential mismatches (e.g. from older versions, missing keys)
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys:
            DEFAULT_LOGGER.warning(
                f"Missing keys when loading state_dict: {load_result.missing_keys}"
            )
        if load_result.unexpected_keys:
            DEFAULT_LOGGER.warning(
                f"Unexpected keys when loading state_dict: {load_result.unexpected_keys}"
            )

        model._is_trained_or_fitted = True  # Assume loaded model was trained
        DEFAULT_LOGGER.info(f"Model {cls.__name__} loaded successfully from {load_dir}")
        return model

    def to_gpu(self):
        """Moves the model and its parameters to the GPU if available.

        Checks for CUDA or MPS (Apple Silicon GPU) availability.
        If no GPU is found, a warning is logged, and the model remains on CPU.

        Returns:
            The model itself, now on the GPU (or CPU if no GPU).
        """
        if GPU_DEVICE is None:
            DEFAULT_LOGGER.warning(
                "No GPU detected (CUDA or MPS). Model remains on CPU."
            )
            return self
        self.to(GPU_DEVICE)
        DEFAULT_LOGGER.info(f"Model {self.__class__.__name__} moved to {GPU_DEVICE}.")
        return self

    def to_cpu(self):
        """Moves the model and its parameters to the CPU.

        Returns:
            The model itself, now on the CPU.
        """
        self.to(CPU_DEVICE)
        DEFAULT_LOGGER.info(f"Model {self.__class__.__name__} moved to {CPU_DEVICE}.")
        return self
