from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Union

from banhxeo.core.vocabulary import Vocabulary
from banhxeo.model.config import ModelConfig
from banhxeo.utils.logging import DEFAULT_LOGGER

GENERATE_LOOP_UPPER_BOUND = 100_000


class BaseLanguageModel(metaclass=ABCMeta):
    def __init__(self, model_config: ModelConfig, vocab: Vocabulary):
        self.config = model_config
        self.vocab = vocab
        self._is_trained_or_fitted = False

        if self.config.vocab_size is None and self.vocab:
            self.config.vocab_size = len(self.vocab)

        DEFAULT_LOGGER.info(
            f"Initializing {self.__class__.__name__} with config: {self.config}"
        )

    def get_config(self) -> ModelConfig:
        """Returns the configuration dictionary of the model."""
        return self.config

    def summary(self):
        """
        Prints a human-readable summary of the model.
        """
        print(f"\n--- Model Summary: {self.__class__.__name__} ---")
        print(f"  Configuration: {self.get_config()}")
        print(f"  Trained/Fitted: {self._is_trained_or_fitted}")

    def save_model(self, save_path: Union[Path, str]):  # Changed to directory
        """Saves model state and configuration to a directory."""
        raise NotImplementedError()

    @classmethod
    def load_model(cls, load_directory: Union[Path, str]):  # Changed to directory
        """Loads a model from a directory."""
        raise NotImplementedError()

    def explain_model_type(self) -> None:
        """
        Prints a brief, simple explanation of what this type of model is,
        its typical use cases, and its basic principles.
        """
        # Example:
        # print(f"{self.__class__.__name__} is a type of X model. "
        #       "It's typically used for Y and works by Z.")
        raise NotImplementedError("Subclasses should provide a model type explanation.")

    def explain_what_is_learned(self) -> None:
        """
        Prints an explanation of what the model learns during its
        training or fitting process (e.g., word probabilities, vector representations,
        network weights).
        """
        # Example:
        # print(f"During training, {self.__class__.__name__} learns: ...")
        raise NotImplementedError("Subclasses should explain what they learn.")
