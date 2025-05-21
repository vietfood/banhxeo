from dataclasses import dataclass
from typing import List, Optional, Union

from torch import Size

from banhxeo.model.config import ModelConfig


@dataclass
class LayerNormConfig:
    eps: float = 1e-5
    elementwise_affine: bool = True
    bias: bool = True


@dataclass
class MLPConfig:
    dim: int = 3072


@dataclass
class MHAConfig:
    dim: int = 64
    n_heads: int = 12


class GPT2Config(ModelConfig):
    d_model: int = 768
    debug: bool = True
    init_range: float = 0.02
    n_ctx: int = 1024
    n_layers: int = 12
    mha: MHAConfig
    ln: LayerNormConfig
    mlp: MLPConfig
