"""
Thanks Claude and Gemini 3.0
"""

import math

import pytest
import torch

from banhxeo import Tensor


@pytest.fixture
def tolerance():
    """Default tolerance values for tensor comparisons."""
    return {"atol": 1e-5, "rtol": 1e-5}


def assert_tensors_close(banhxeo_tensor, torch_tensor, atol=1e-5, rtol=1e-5):
    """Helper for comparing banhxeo tensors with PyTorch tensors."""
    result = banhxeo_tensor.realize()
    assert torch.allclose(result, torch_tensor, atol=atol, rtol=rtol), (
        f"\nGot: {result}\nExpected: {torch_tensor}"
    )


class TestUnaryOps:
    """Test suite for unary operations on tensors."""

    def test_log2(self):
        """Test log2 operation."""
        data = [1.0, 2.0, 4.0, 8.0, 0.5]
        t = Tensor(data)
        out = t.log2()

        pt = torch.tensor(data)
        assert_tensors_close(out, torch.log2(pt))

    def test_exp2(self):
        """Test exp2 operation. Note: banhxeo.Tensor.exp() maps to UnaryOps.EXP2."""
        data = [0.0, 1.0, 2.0, -1.0]
        t = Tensor(data)
        out = t.exp2()

        pt = torch.tensor(data)
        assert_tensors_close(out, torch.exp2(pt))

    def test_sin(self):
        """Test sine operation."""
        data = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
        t = Tensor(data)
        out = t.sin()

        pt = torch.tensor(data)
        assert_tensors_close(out, torch.sin(pt))

    def test_sqrt(self):
        """Test square root operation."""
        data = [4.0, 9.0, 16.0, 0.0]
        t = Tensor(data)
        out = t.sqrt()

        pt = torch.tensor(data)
        assert_tensors_close(out, torch.sqrt(pt))


class TestBinaryOps:
    """Test suite for binary operations on tensors."""

    def test_add_simple(self):
        """Test simple element-wise addition."""
        data1 = [1.0, 2.0, 3.0]
        data2 = [4.0, 5.0, 6.0]
        t1, t2 = Tensor(data1), Tensor(data2)

        out = t1 + t2
        pt = torch.tensor(data1) + torch.tensor(data2)
        assert_tensors_close(out, pt)

    def test_sub_simple(self):
        """Test simple element-wise subtraction."""
        t1 = Tensor([10.0, 20.0])
        t2 = Tensor([1.0, 2.0])
        assert_tensors_close(t1 - t2, torch.tensor([9.0, 18.0]))

    def test_mul_simple(self):
        """Test simple element-wise multiplication."""
        t1 = Tensor([2.0, 3.0])
        t2 = Tensor([4.0, 5.0])
        assert_tensors_close(t1 * t2, torch.tensor([8.0, 15.0]))

    def test_scalar_ops(self):
        """Test implicit conversion of int/float to Tensor."""
        t = Tensor([1.0, 2.0, 3.0])

        # Test __add__ with float
        out_add = t + 5.0
        assert_tensors_close(out_add, torch.tensor([6.0, 7.0, 8.0]))

        # Test __mul__ with int
        out_mul = t * 2
        assert_tensors_close(out_mul, torch.tensor([2.0, 4.0, 6.0]))


class TestBroadcasting:
    """
    Critical system test verifying that _broadcasted calls expand()
    correctly before binary operations.
    """

    def test_broadcast_vector_to_matrix(self):
        """Test broadcasting a vector to a matrix."""
        # Shape (2, 3)
        data1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        # Shape (3,) -> should broadcast to (1, 3) -> (2, 3)
        data2 = [10.0, 20.0, 30.0]

        t1 = Tensor(data1)
        t2 = Tensor(data2)

        out = t1 + t2

        pt1 = torch.tensor(data1)
        pt1 = torch.tensor(data1)
        pt2 = torch.tensor(data2)
        assert_tensors_close(out, pt1 + pt2)

    def test_broadcast_column_to_matrix(self):
        """Test broadcasting a column vector to a matrix."""
        # Shape (2, 3)
        data1 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        # Shape (2, 1) -> should broadcast to (2, 3)
        data2 = [[10.0], [20.0]]

        t1 = Tensor(data1)
        t2 = Tensor(data2)

        out = t1 + t2

        pt1 = torch.tensor(data1)
        pt2 = torch.tensor(data2)
        assert_tensors_close(out, pt1 + pt2)


class TestChainedOps:
    """Test that the LazyBuffer graph builds correctly over multiple steps."""

    def test_complex_expression(self):
        """Test a complex chained expression: sin(x + y) * sqrt(x)."""
        # Formula: sin(x + y) * sqrt(x)
        x_data = [4.0, 16.0]
        y_data = [0.0, 0.0]

        x = Tensor(x_data)
        y = Tensor(y_data)

        # (4+0).sin() * sqrt(4) = sin(4) * 2
        out = (x + y).sin() * x.sqrt()

        pt_x = torch.tensor(x_data)
        pt_y = torch.tensor(y_data)
        pt_out = torch.sin(pt_x + pt_y) * torch.sqrt(pt_x)

        assert_tensors_close(out, pt_out)
