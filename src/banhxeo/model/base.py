from __future__ import annotations

from abc import ABCMeta
from pathlib import Path
from typing import Optional, Type, Union

from banhxeo.core.vocabulary import Vocabulary
from banhxeo.model.config import ModelConfig
from banhxeo.utils.logging import DEFAULT_LOGGER


GENERATE_LOOP_UPPER_BOUND = 100_000


class BaseLanguageModel(metaclass=ABCMeta):
    """Abstract base class for all language models in the Banhxeo library.

    This class defines a common interface for language models, including
    configuration management, saving/loading, and basic summary functionalities.
    It is intended to be subclassed by specific model types (e.g., N-gram,
    neural models).

    Attributes:
        config (ModelConfig): The configuration object for the model.
        vocab (Vocabulary): The vocabulary associated with the model.
        _is_trained_or_fitted (bool): A flag indicating whether the model
            has been trained (for neural models) or fitted (for statistical models).
    """

    # Should be override by subclasses's config
    # ConfigClass: Type[ModelConfig] = ModelConfig

    def __init__(self, model_config: ModelConfig, vocab: Vocabulary):
        """Initializes the BaseLanguageModel.

        Args:
            model_config: The configuration object for the model.
                It should be an instance of `ModelConfig` or its subclass.
            vocab: The `Vocabulary` instance to be used by the model.
        """
        self.config = model_config
        self.vocab = vocab
        self._is_trained_or_fitted: bool = False

        if self.config.vocab_size is None and self.vocab:
            self.config.vocab_size = len(self.vocab)

        DEFAULT_LOGGER.info(
            f"Initializing {self.__class__.__name__} "
            f"with config: {type(self.config).__name__} "
            f"(vocab size from vocab: {self.config.vocab_size if self.vocab else 'N/A'})"
        )

    def get_config(self) -> ModelConfig:
        """Returns the configuration object of the model.

        Returns:
            The `ModelConfig` (or subclass) instance associated with this model.
        """
        return self.config

    def summary(self) -> None:
        """Prints a human-readable summary of the model.

        This typically includes the model class name, its configuration,
        and its training/fitted status. Subclasses (like `NeuralLanguageModel`)
        may override this to provide more detailed summaries (e.g., layer structure).
        """
        print(f"\n--- Model Summary: {self.__class__.__name__} ---")
        print(f"  Configuration: {str(self.config)}")
        if self.vocab:
            print(f"  Vocabulary Size: {len(self.vocab)}")
        print(f"  Trained/Fitted: {self._is_trained_or_fitted}")

    def save_model(self, save_directory: Union[Path, str]) -> None:
        """Saves the model's state and configuration.

        The specific format and content depend on the model type. For neural
        models, this typically involves saving weights and architecture config.
        For statistical models, it might involve saving learned parameters or data structures.

        Args:
            save_path: The file path or directory path where the model should be saved.
                Conventionally, for neural models, this might be a directory. For
                simpler models, it could be a single file.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement save_model."
        )

    @classmethod
    def load_model(
        cls: Type[BaseLanguageModel],
        load_directory: Union[Path, str],
        vocab: Optional[
            Vocabulary
        ] = None,  # Vocab might be loaded alongside or provided
        **kwargs,
    ):
        """Loads a model from a saved state.

        Args:
            cls: The class itself.
            load_path: The file path or directory path from which to load the model.
            # vocab: Optionally, a Vocabulary instance if not saved with the model.
            **kwargs: Additional arguments that might be needed for loading,
                      such as a Vocabulary instance if not bundled.

        Returns:
            An instance of the language model class.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(f"{cls.__name__} must implement load_model.")
