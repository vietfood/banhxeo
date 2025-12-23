import numpy as np
import torch

from banhxeo import Tensor


def test_impossible_reshape():
    # 1. Create (4, 4)
    t = Tensor(np.arange(16).reshape(4, 4))

    # 2. Transpose -> (4, 4) but with stride (1, 4)
    t = t.permute((1, 0))

    # 3. Reshape -> (16,)
    # This should trigger your contiguous barrier logic!
    flat = t.reshape((16,))
    flat.realize()

    assert flat.lazydata.realized is not None
    print(flat.lazydata.realized.data)
    # Should print: [0, 4, 8, 12, 1, 5, 9, 13, ...] (Column major order)


test_impossible_reshape()
