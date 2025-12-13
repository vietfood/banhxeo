from typing import ClassVar, List, Optional, Tuple, Union

import numpy as np
import torch

from banhxeo.core.buffer import LazyBuffer


class Tensor:
    __slots__ = "lazydata", "requires_grad", "grad", "_ctx"
    __deletable__ = ("_ctx",)
    training: ClassVar[bool] = False

    def __init__(
        self,
        data: Optional[Union[List, LazyBuffer, Tuple, np.ndarray, torch.Tensor]] = None,
        device: Optional[Union[str, tuple, list]] = None,
    ):
        pass
