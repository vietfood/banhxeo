from banhxeo.core.tokenizer import Tokenizer
from banhxeo.core.vocabulary import Vocabulary
from banhxeo.models import LanguageModel, ModelConfig


class ProbLanguageModel(LanguageModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

    def fit(self, corpus: list[str]):
        raise NotImplementedError("Probabilistic models must implement 'fit' logic.")
        # self._set_trained_status(True) # Call this at the end of fitting

    def _get_probability(self, *args, **kwargs):
        """
        Calculates the probability of a sequence, token, or event.
        The signature will be highly specific to the model (e.g., for an N-gram,
        it might be `get_probability(context_tokens: tuple, target_token: str)`).
        """
        raise NotImplementedError
