"""
Thanks Gemini 3.0 Pro
"""

import numpy as np
import torch

import banhxeo

# Set seed for reproducibility (so you can debug specific values)
np.random.seed(1337)
torch.manual_seed(1337)


def test_activations():
    print("=== 🧪 Testing Activations (ReLU & Where) ===")

    # 1. Setup
    device = "CUDA"
    # Create a vector with mix of positive and negative values
    data = [-2.0, -1.0, 0.0, 1.0, 2.0]
    t_in = banhxeo.Tensor(data, device=device)

    # 2. Test Less (<)
    print("\n[1/3] Testing Less (< 0)...")
    t_mask = t_in < 0.0
    t_mask.realize()
    res_mask = t_mask.numpy()

    print(f"Input: {data}")
    print(f"Mask (<0): {res_mask}")

    # Expect: [1.0, 1.0, 0.0, 0.0, 0.0] (assuming float mask)
    expected_mask = [1.0, 1.0, 0.0, 0.0, 0.0]
    if np.allclose(list(res_mask), expected_mask):  # type: ignore
        print("✅ Less Op works!")
    else:
        print("❌ Less Op Failed!")
        return

    # 3. Test Where
    print("\n[2/3] Testing Where...")
    # Replace negatives with -100
    t_where = banhxeo.Tensor.where(t_mask, -100.0, t_in)
    t_where.realize()
    res_where = t_where.numpy()

    print(f"Where(mask, -100, x): {res_where}")
    expected_where = [-100.0, -100.0, 0.0, 1.0, 2.0]

    if np.allclose(list(res_where), expected_where):  # type: ignore
        print("✅ Where Op works!")
    else:
        print("❌ Where Op Failed!")
        return

    # 4. Test ReLU
    print("\n[3/3] Testing ReLU...")
    t_relu = t_in.relu()
    t_relu.realize()
    res_relu = t_relu.numpy()

    print(f"ReLU result: {res_relu}")
    expected_relu = [0.0, 0.0, 0.0, 1.0, 2.0]

    if np.allclose(list(res_relu), expected_relu):  # type: ignore
        print("✅ ReLU works!")
    else:
        print("❌ ReLU Failed!")


if __name__ == "__main__":
    test_activations()
