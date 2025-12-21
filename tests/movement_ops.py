"""
Thanks Claude and Gemini 3.0
"""

import numpy as np
import pytest
import torch

from banhxeo import Tensor
from banhxeo.buffer import MovementOps
from banhxeo.view import View


@pytest.fixture
def setup_2d_tensor():
    """Fixture to create a Tensor with a specific 2D shape view."""

    def _setup(data, shape):
        t = Tensor(data)
        # HACK: Force view to desired shape because Tensor init only supports 1D inference correctly
        t.lazydata.view = View.create(shape)
        return t

    return _setup


class TestMovementOps:
    """Test suite for tensor movement operations."""

    def test_permute(self, setup_2d_tensor):
        """Test 2D permute (transpose) operation."""
        # Create a flattened tensor [0, 1, 2, 3, 4, 5]
        data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        t = setup_2d_tensor(data, (2, 3))

        # Perform Permute (1, 0) -> Transpose
        out = t.permute((1, 0))

        # Realize
        result = out.realize()

        # Verify against PyTorch
        torch_t = torch.tensor(data, device="cuda").reshape(2, 3)
        torch_out = torch_t.permute(1, 0).contiguous()

        assert torch.allclose(result, torch_out), "Permute logic is wrong."

    def test_slice(self, setup_2d_tensor):
        """Test slice on transposed tensor."""
        # Setup Data: 3x4 Matrix
        data = np.arange(12, dtype=np.float32)
        t = setup_2d_tensor(data, (3, 4))

        # Permute (Transpose) -> (4, 3)
        t_T = t.permute((1, 0))

        # Slice top-left (2, 2)
        slice_args = ((0, 2), (0, 2))
        out = t_T.slice(slice_args)

        result = out.realize()

        # PyTorch verification
        pt = torch.tensor(data, device="cuda").reshape(3, 4)
        pt_out = pt.permute(1, 0)[0:2, 0:2].contiguous()

        assert torch.allclose(result, pt_out), "Slice+Permute composition failed."

    def test_expand(self, setup_2d_tensor):
        """Test expand (broadcast) operation."""
        # Setup Data: (3, 1)
        data = [0.0, 1.0, 2.0]
        t = setup_2d_tensor(data, (3, 1))

        # Expand to (3, 4)
        out = t.expand((3, 4))

        # Realize
        result = out.realize()

        # PyTorch verification
        pt = torch.tensor(data, device="cuda").reshape(3, 1)
        pt_out = pt.expand(3, 4).contiguous()

        assert torch.allclose(result, pt_out), "Expand logic failed."

    def test_reshape_noop(self):
        """Verify that RESHAPE is currently a no-op (identity) as per implementation."""
        data = [0.0, 1.0, 2.0, 3.0]
        t = Tensor(data)  # Shape (4,)

        # Call RESHAPE op directly
        out_lazy = t.lazydata.movement_ops(MovementOps.RESHAPE, (2, 2))
        out = Tensor(out_lazy)

        # Expectation: Currently implementation returns self.view for RESHAPE
        # So shape remains (4,)
        assert out.lazydata.view.shape == (4,)
        # If it were implemented, it should be (2, 2)

    def test_pad_noop(self):
        """Verify that PAD is currently a no-op (identity) as per implementation."""
        data = [1.0]
        t = Tensor(data)  # Shape (1,)

        # Call PAD op directly
        out_lazy = t.lazydata.movement_ops(MovementOps.PAD, ((0, 1),))
        out = Tensor(out_lazy)

        # Expectation: Currently implementation returns self.view for PAD
        assert out.lazydata.view.shape == (1,)
