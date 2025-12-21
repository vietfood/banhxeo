import numpy as np
import torch

from src.banhxeo import Tensor
from src.banhxeo.buffer import MovementOps
from src.banhxeo.view import View


def test_slice_transpose():
    print("=== Testing Slice on Transposed Tensor ===")

    # 1. Setup Data: 3x4 Matrix
    # [[ 0,  1,  2,  3],
    #  [ 4,  5,  6,  7],
    #  [ 8,  9, 10, 11]]
    data = np.arange(12, dtype=np.float32)
    t = Tensor(data)
    # Force view to (3, 4)
    t.lazydata.view = View.create((3, 4))

    # 2. Permute (Transpose) -> (4, 3)
    # [[ 0, 4, 8],
    #  [ 1, 5, 9],
    #  [ 2, 6, 10],
    #  [ 3, 7, 11]]
    t_T = t.permute((1, 0))

    # 3. Slice top-left (2, 2)
    # Expected:
    # [[ 0, 4],
    #  [ 1, 5]]
    #
    # Internally:
    # Original Strides: (4, 1)
    # Transposed Strides: (1, 4)
    # Sliced Offset:
    #   dim0 start=0 -> + 0 * 1 = 0
    #   dim1 start=0 -> + 0 * 4 = 0
    #   Total offset = 0
    # Sliced Shape: (2, 2)
    # Strides: (1, 4) -> Still non-contiguous relative to row-major!

    # Pass slice args as tuple of tuples: ((start, end), (start, end))
    # We need to expose a way to call slice.
    # Since Tensor doesn't have slice method yet, we can hack it or add it.

    # Adding a helper to Tensor class for testing:
    # def slice(self, arg): return Tensor(self.lazydata.movement_ops(MovementOps.SLICE, arg))

    out = Tensor(t_T.lazydata.movement_ops(MovementOps.SLICE, ((0, 2), (0, 2))))

    result = out.realize().view(out.lazydata.view.shape)

    # PyTorch verification
    pt = torch.tensor(data).reshape(3, 4)
    pt_out = pt.permute(1, 0)[0:2, 0:2].contiguous()

    print("My Result:\n", result)
    print("PyTorch Result:\n", pt_out)

    assert torch.allclose(result, pt_out)
    print("✅ PASS: Slice+Permute composition works.")


if __name__ == "__main__":
    test_slice_transpose()
