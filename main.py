import numpy as np

import banhxeo


def test_impossible_reshape(device="cpu"):
    # 1. Create (4, 4)
    t = banhxeo.Tensor(np.arange(16).reshape(4, 4), device=device)

    # 2. Transpose -> (4, 4) but with stride (1, 4)
    t = t.permute((1, 0))

    # 3. Reshape -> (16,)
    # This should trigger your contiguous barrier logic!
    flat = t.reshape((16,))
    flat.realize()

    assert flat.lazydata.realized is not None
    print(flat.lazydata.realized.data)
    # Should print: [0, 4, 8, 12, 1, 5, 9, 13, ...] (Column major order)


def test_mixed_execution(device="cpu"):
    # 1. Elementwise Ops (Should Fuse)
    a = banhxeo.Tensor([1, 2], device=device)
    b = banhxeo.Tensor([3, 4], device=device)
    c = (a + b) * 2  # This should be ONE kernel call

    # 2. Reshape (Should trigger Contiguous Barrier)
    # create non-contiguous view
    d = c.expand((2, 2)).permute((1, 0))
    e = d.reshape((4,))  # Barrier 1: Contiguous

    # 3. MatMul (Should trigger MatMul Barrier)
    # (4,1) @ (1,4) -> (4,4)
    f = e.reshape((4, 1)) @ e.reshape((1, 4))  # Barrier 2: MatMul

    # 4. Final Realize
    print(f.realize())


# test_impossible_reshape()
test_mixed_execution("cuda")
