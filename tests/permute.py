import torch

from src.banhxeo import Tensor
from src.banhxeo.view import View


def test_permute_2d():
    print("=== Testing 2D Permute (Transpose) ===")

    # 1. Create a flattened tensor [0, 1, 2, 3, 4, 5]
    data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    t = Tensor(data)

    # 2. HACK: Manually force a (2, 3) view so we can test permute
    # Shape: (2, 3) -> Strides: (3, 1)
    # [[0, 1, 2],
    #  [3, 4, 5]]
    t.lazydata.view = View.create((2, 3))
    print(
        f"Original View: Shape={t.lazydata.view.shape}, Strides={t.lazydata.view.strides}"
    )

    # 3. Perform Permute (1, 0) -> Transpose
    # Expected Shape: (3, 2)
    # Expected Strides: (1, 3)  <-- Strides are swapped!
    # [[0, 3],
    #  [1, 4],
    #  [2, 5]]
    out = t.permute((1, 0))
    print(
        f"Permuted View: Shape={out.lazydata.view.shape}, Strides={out.lazydata.view.strides}"
    )

    # 4. Realize (Run the kernel)
    # This checks if your Triton code correctly uses the swapped strides
    result = out.realize()

    # 5. Verify against PyTorch
    torch_t = torch.tensor(data, device="cuda").reshape(2, 3)
    torch_out = torch_t.permute(
        1, 0
    ).contiguous()  # contiguous matches physical memory output

    print("\nMy Result:\n", result)
    print("PyTorch Result:\n", torch_out)

    assert torch.allclose(result, torch_out), "MISMATCH! Kernel logic is wrong."
    print("\n✅ PASS: Permute logic holds.")


if __name__ == "__main__":
    test_permute_2d()
