from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict


@dataclass(frozen=True)
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 12
    num_embeds: int = 768
    dropout_rate: float = 0.1
    use_bias: bool = True
    dtype: Optional[str] = None


# TODO
