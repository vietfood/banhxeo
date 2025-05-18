from abc import ABCMeta
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, model_validator
from typing_extensions import Self

from banhxeo.utils.logging import DEFAULT_LOGGER

GENERATE_LOOP_UPPER_BOUND = 100_000


class ModelConfig(BaseModel):
    def __str__(self):
        raise NotImplementedError()


class GenerateConfig(BaseModel):
    sampling: str = "greedy"
    max_length: Optional[int] = None  # generate until sep/end token
    k: Optional[int] = None  # for top K
    p: Optional[float] = None  # for top P
    temp: Optional[int] = None  # for temperature


class LanguageModel(metaclass=ABCMeta):
    def __init__(self, model_config: ModelConfig):
        """
        Initializes the model with a configuration.
        The config dictionary should store all hyperparameters and settings.
        """
        self.config = model_config
        self._is_trained_or_fitted = False
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

    def save_model(self, filepath: Union[Path, str]):
        """
        Saves the model's learned state (parameters, counts, etc.) and configuration.
        """
        raise NotImplementedError("Subclasses must implement model saving.")

    @classmethod
    def load_model(cls, filepath: Union[Path, str]):
        """
        Loads a model from a file.
        """
        raise NotImplementedError("Subclasses must implement model loading.")

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

    def get_hyperparameters_table(self) -> str:
        """
        Returns a string formatted as a table or list of
        key hyperparameters and their current values.
        """
        table = f"Hyperparameters for {self.__class__.__name__}:\n"
        if not self.config:
            return (
                table + "  (No configuration provided or model not fully initialized)"
            )
        for key, value in self.config.model_dump().items():
            table += f"  - {key}: {value}\n"
        return table

    def generate_sequence(
        self,
        prompt: str,
        sampling: str = "greedy",
        max_length: Optional[int] = 20,
        **kwargs,
    ) -> str:
        """
        Generates a sequence of tokens based on the learned probabilities.
        """
        raise NotImplementedError
