import numpy as np
import pytest
import torch

from banhxeo.tensor import Tensor

device = "cuda"


# --- Helper Function ---
def check(bx_tensor, th_tensor, atol=1e-5, rtol=1e-5):
    """
    Compares a banhxeo Tensor with a PyTorch Tensor.
    """
    # Force realization and conversion to numpy
    bx_np = bx_tensor.numpy()
    th_np = th_tensor.detach().numpy()

    # Check shapes
    assert bx_np.shape == th_np.shape, f"Shape mismatch: {bx_np.shape} vs {th_np.shape}"

    # Check values
    np.testing.assert_allclose(bx_np, th_np, atol=atol, rtol=rtol)


# --- Basic Tests ---


def test_sum_simple_1d():
    data = np.random.randn(1024).astype(np.float32)
    t = Tensor(data, device=device)
    th = torch.from_numpy(data).to(device)

    check(t.sum(), th.sum())


def test_sum_2d_axis_0():
    # This tests the PERMUTE logic in your frontend
    # (128, 256) -> sum(0) -> (256,)
    shape = (128, 256)
    data = np.random.randn(*shape).astype(np.float32)
    t = Tensor(data, device=device)
    th = torch.from_numpy(data).to(device)

    check(t.sum(axis=0), th.sum(dim=0))


def test_sum_2d_axis_1():
    # This tests the Direct Backend Mapping (last dim)
    shape = (128, 256)
    data = np.random.randn(*shape).astype(np.float32)
    t = Tensor(data, device=device)
    th = torch.from_numpy(data).to(device)

    check(t.sum(axis=1), th.sum(dim=1))


def test_sum_keepdim():
    shape = (32, 32)
    data = np.random.randn(*shape).astype(np.float32)
    t = Tensor(data, device=device)
    th = torch.from_numpy(data).to(device)

    # Check shapes explicitly
    out = t.sum(axis=1, keepdim=True)
    assert out.shape == (32, 1)
    check(out, th.sum(dim=1, keepdim=True))


# --- Edge Cases (The Important Ones) ---


def test_sum_non_contiguous():
    # Create a non-contiguous view using slicing
    # x shape (10, 10), we slice x[:, ::2] -> shape (10, 5) with stride (10, 2)
    data = np.random.randn(10, 10).astype(np.float32)
    t = Tensor(data, device=device)
    th = torch.from_numpy(data).to(device)

    t_view = t[:, ::2]
    th_view = th[:, ::2]

    # This triggers the "if not src.view.is_contiguous()" check in your backend
    check(t_view.sum(), th_view.sum())
    check(t_view.sum(axis=1), th_view.sum(dim=1))


def test_max_negative_values():
    # Initialize with values < 0 to ensure your -inf init works
    data = np.random.uniform(-100, -1, (10, 10)).astype(np.float32)
    t = Tensor(data, device=device)
    th = torch.from_numpy(data).to(device)

    # Note: torch.max returns (values, indices) tuple when dim is specified
    check(t.max(axis=0), th.max(dim=0).values)
    check(t.max(), th.max())


# --- Composite Ops Tests ---


def test_mean():
    data = np.random.randn(50, 50).astype(np.float32)
    t = Tensor(data, device=device)
    th = torch.from_numpy(data).to(device)

    check(t.mean(), th.mean())
    check(t.mean(axis=0), th.mean(dim=0))


def test_min():
    # Tests the -max(-x) trick
    data = np.random.randn(50, 50).astype(np.float32)
    t = Tensor(data, device=device)
    th = torch.from_numpy(data).to(device)

    check(t.min(), th.min())
    check(t.min(axis=1), th.min(dim=1).values)


# --- Large Scale System Test ---


@pytest.mark.slow
def test_large_reduction():
    # Reduce 10 million elements
    # 10,000 rows * 1,000 columns
    # This verifies your grid calculation (M,) vs (M/BLOCK,) logic
    M, N = 10000, 1000
    t = Tensor.rand((M, N), device=device)

    # Just check it runs without crashing and shape is right
    res = t.sum(axis=1)
    res.realize()
    assert res.shape == (M,)
