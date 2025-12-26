import triton
import triton.language as tl


def reduce_sum_kernel(
    X,
    Y,
    stride_x_row,
    stride_x_col,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Map program to the row we are reducing
    row_idx = tl.program_id(0)

    # 2. Calculate the base pointer for this specific row
    # We move X to the start of the row
    row_start_ptr = X + row_idx * stride_x_row

    # 3. Accumulate
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        # CRITICAL: Multiply col index by col stride to handle non-contiguous views
        # e.g., x[:, ::2] has stride_x_col = 2
        offsets = cols * stride_x_col

        val = tl.load(row_start_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        _sum += val

    # 4. Final reduction
    result = tl.sum(_sum, axis=0)
    tl.store(Y + row_idx, result)


def reduce_max_kernel(
    X,  # pointer to input data
    Y,  # pointer to output data
    stride_x_row,  # Stride to jump to next row (view.strides[0])
    stride_x_col,  # Stride to jump pixels in row (view.strides[1])
    N,  # Number of columns (reduction dimension size)
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = X + row_idx * stride_x_row

    _max = tl.full([BLOCK_SIZE], float("-inf"), dtype=tl.float32)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        offsets = cols * stride_x_col
        val = tl.load(row_start_ptr + offsets, mask=mask, other=float("-inf")).to(
            tl.float32
        )
        _max = tl.maximum(_max, val)

    result = tl.max(_max, axis=0)
    tl.store(Y + row_idx, result)
