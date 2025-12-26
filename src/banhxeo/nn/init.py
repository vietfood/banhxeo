import math
from typing import Tuple

from banhxeo import Tensor


# https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
def kaiming_uniform(shape: Tuple[int, ...], a: float = 0.01, **kwargs):
    bound = (
        math.sqrt(3.0) * math.sqrt(2.0 / (1 + a**2)) / math.sqrt(math.prod(shape[1:]))
    )
    return Tensor.uniform(shape, low=-bound, high=bound, **kwargs)
