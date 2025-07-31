from typing import Any, Dict, Optional

import flax.linen as nn
from flax.training import train_state

from banhxeo import DEFAULT_SEED


class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    # If a model has no batch statistics, it is None
    batch_stats: Any = None

    # rng to keep for init, dropout, etc.
    rng: Any = None


class TrainerModule:
    model_class: nn.Module
    model_hparams: Dict[str, Any]
    optimizer_hparams: Dict[str, Any]
    logger_params: Optional[Dict[str, Any]] = None
    seed: int = DEFAULT_SEED
    debug: bool = False

    # TODO
