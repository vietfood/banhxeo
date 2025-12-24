"""
Thanks Gemini 3.0 Pro
"""

import numpy as np
import torch

import banhxeo

# Set seed for reproducibility (so you can debug specific values)
np.random.seed(1337)
torch.manual_seed(1337)


def test_mlp_forward():
    print("=== 🧪 Testing MLP Forward Pass (x @ W + b) ===")

    # 1. Setup Dimensions
    # Keep them small enough to inspect, but non-square to catch transpose bugs
    B = 4  # Batch size
    Din = 8  # Input dim
    Dout = 16  # Output dim

    device = "CUDA"

    # 2. Prepare Data (Numpy source)
    # We use float32.
    raw_x = np.random.randn(B, Din).astype(np.float32)
    raw_w = np.random.randn(Din, Dout).astype(np.float32)
    raw_b = np.random.randn(Dout).astype(np.float32)

    # 3. Ground Truth (PyTorch)
    t_x = torch.tensor(raw_x)
    t_w = torch.tensor(raw_w)
    t_b = torch.tensor(raw_b)

    # Run PyTorch execution
    t_matmul = t_x @ t_w
    t_out = t_matmul + t_b

    print(f"Shapes: Input{t_x.shape} @ Weights{t_w.shape} + Bias{t_b.shape}")
    print(f"Expected Output Shape: {t_out.shape}")

    # 4. Banhxeo Execution
    bx_x = banhxeo.Tensor(raw_x, device=device)
    bx_w = banhxeo.Tensor(raw_w, device=device)
    bx_b = banhxeo.Tensor(raw_b, device=device)

    # Step A: Just Matmul
    print("\n[1/2] Testing Matmul only (x @ W)...")
    bx_matmul = bx_x @ bx_w
    bx_matmul.realize()

    # Pull data back (assuming .numpy() or accessing realized buffer directly)
    # Using internal buffer access to be safe if .numpy() isn't implemented on Tensor yet
    res_matmul = bx_matmul.lazydata.realized.to_cpu().numpy()  # type: ignore

    if np.allclose(res_matmul, t_matmul.numpy(), atol=1e-3, rtol=1e-3):
        print("✅ Matmul matched PyTorch!")
    else:
        print("❌ Matmul Mismatch!")
        print("Expected first row:\n", t_matmul.numpy()[0])
        print("Got first row:\n", res_matmul[0])
        print("Max diff:", np.abs(res_matmul - t_matmul.numpy()).max())
        return  # Stop here if matmul is broken

    # Step B: Add Bias (Implicit Broadcasting)
    # This tests if your View logic handles (B, Dout) + (Dout,) correctly
    print("\n[2/2] Testing Add Bias (x @ W + b)...")

    # NOTE: If your Tensor.add doesn't auto-reshape, you might fail here.
    # This is part of the test!
    bx_out = bx_matmul.add(bx_b)
    bx_out.realize()

    res_out = bx_out.lazydata.realized.to_cpu().numpy()  # type: ignore

    if np.allclose(res_out, t_out.numpy(), atol=1e-3, rtol=1e-3):
        print("✅ MLP Forward Pass matched PyTorch!")
        print("\nSample Output:\n", res_out[0, :5])
    else:
        print("❌ Bias Add Mismatch! (Check your broadcasting/view logic)")
        print(f"Shape of result: {res_out.shape}")
        print("Expected first row:\n", t_out.numpy()[0])
        print("Got first row:\n", res_out[0])
        print("Max diff:", np.abs(res_out - t_out.numpy()).max())


if __name__ == "__main__":
    test_mlp_forward()
