## Module 6: Specialized Kernels

Improve matmul and reduce implementations.

## Why This Module Is Later

Matmul and reductions are important, but they are not the foundation of the
compiler.

They use different parallel patterns from elementwise kernels, so they should be
treated as explicit scheduling boundaries until the rest of the compiler is
inspectable. This module is where you study GPU programming: tiling,
coalescing, accumulation, tree reduction, and batching.

Do not use this module to hide framework bugs. If a matmul result is wrong
because views or broadcasting are wrong, go back to Module 1 or Module 2.

### Assignment 6.1: Understand the Matmul Kernel ⭐⭐

**Task:** Annotate every line of `kernels/matmul.py` with comments explaining:
1. What each variable represents
2. Why that particular optimization is used
3. What would happen without it

**Key concepts to understand:**
- Tiling: Why BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K?
- Memory coalescing: Why the specific access pattern?
- L2 cache optimization: What does GROUP_SIZE_M do?
- Accumulator: Why accumulate in registers?

**Resources:**
- [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [NVIDIA matmul optimization](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)

**Why this assignment:** Annotating an existing kernel forces you to separate
Triton syntax from GPU-performance ideas. You should know which lines are about
correctness and which lines are about speed.

---

### Assignment 6.2: Implement Batched Matmul ⭐⭐⭐

**Current limitation:** Only 2D matmul at `backend/__init__.py:226-260`

**Task:** Support these cases:
1. `[B, M, K] @ [B, K, N]` → `[B, M, N]` (batched)
2. `[B, M, K] @ [K, N]` → `[B, M, N]` (broadcast right)
3. `[M, K] @ [B, K, N]` → `[B, M, N]` (broadcast left)

**Implementation steps:**

1. **Update `buffer.py:320-334`** - Handle batch dimensions in `matmul()`:
```python
def matmul(self, other: "LazyBuffer"):
    # Determine output shape based on input shapes
    # Handle 2D, 3D, and broadcasting cases
    # YOUR CODE HERE
```

2. **Create batched kernel** - `kernels/matmul.py`:
```python
@triton.jit
def batched_matmul_kernel(
    A, B, C,
    batch_stride_a, batch_stride_b, batch_stride_c,  # New!
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, ...
):
    # Get batch index from program_id
    batch_id = tl.program_id(2)  # New dimension!

    # Offset pointers by batch
    A = A + batch_id * batch_stride_a
    B = B + batch_id * batch_stride_b
    C = C + batch_id * batch_stride_c

    # Rest is same as 2D kernel
```

3. **Update `exec_matmul()`** - Handle batch dimension in grid:
```python
grid = lambda META: (
    triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    1,
    B,  # Batch dimension!
)
```

**Hint for broadcast:** If one input doesn't have batch dim, its `batch_stride = 0` (same as expand with stride 0).

**Test cases:**
```python
# Standard batched
a = Tensor.rand(4, 10, 20, device="cuda")
b = Tensor.rand(4, 20, 30, device="cuda")
c = a @ b
assert c.shape == (4, 10, 30)

# Broadcast right operand
a = Tensor.rand(4, 10, 20, device="cuda")
b = Tensor.rand(20, 30, device="cuda")
c = a @ b
assert c.shape == (4, 10, 30)
```

**tinygrad reference:**
- tinygrad handles batched matmul by reshaping/permuting to 2D, doing 2D matmul, then reshaping back
- Alternative: explicit batched kernel (what we're doing)

**Why this assignment:** Batched matmul connects three earlier ideas: batch
broadcasting, stride-0 reuse, and specialized kernel launch geometry.

---

### Assignment 6.3: Implement Multi-Axis Reduce ⭐⭐⭐

**Current limitation:** `_reduce()` in `tensor.py:234-280` only handles single axis.

**Task:** Support `x.sum(axis=(0, 2))` reducing multiple axes at once.

**Current approach:**
```python
# For x.sum(axis=(0, 2)) with shape (2, 3, 4):
# 1. Permute to move reduce axes to end: (3, 2, 4) with axes (1, 0, 2)
# 2. Reshape to 2D: (3, 8)
# 3. Reduce last axis: (3, 1)
# 4. Reshape to output: (3,) or (1, 3, 1) depending on keepdim
```

**Your task:** Implement this properly.

```python
def _reduce(self, op: ReduceOp, axis=None, keepdim=False):
    if axis is None:
        # Reduce all dimensions
        flat = self.reshape((math.prod(self.shape),))
        result = flat._reduce(op, axis=0)
        return result if not keepdim else result.reshape((1,) * len(self.shape))

    if isinstance(axis, int):
        axis = (axis,)

    # Normalize negative axes
    axis = tuple(a if a >= 0 else len(self.shape) + a for a in axis)

    # YOUR IMPLEMENTATION:
    # 1. Permute reduce axes to end
    # 2. Merge reduce axes into one
    # 3. Do single-axis reduce
    # 4. Handle keepdim
```

**Test cases:**
```python
x = Tensor.rand(2, 3, 4, device="cuda")

# Single axis
assert x.sum(axis=1).shape == (2, 4)

# Multiple axes
assert x.sum(axis=(0, 2)).shape == (3,)

# All axes
assert x.sum().shape == ()

# With keepdim
assert x.sum(axis=1, keepdim=True).shape == (2, 1, 4)
```

**tinygrad reference:**
- `tinygrad/tensor.py` → `Tensor._reduce()`
- Note how tinygrad handles the axis parameter and keepdim

**Why this assignment:** Multi-axis reduce teaches lowering. The user asks for
one high-level op, but the framework may implement it as permute, reshape,
single-axis reduce, and reshape back.

---

### Assignment 6.4: Reduce Kernel with Tree Reduction ⭐⭐⭐

**Problem:** Current reduce kernel (`kernels/reduce.py:5-37`) uses simple loop.

For very large N (reduction dimension), this is slow because:
1. One thread does all the work for each row
2. No parallelism within the reduction

**Better approach:** Tree reduction with multiple threads cooperating.

```
Level 0: [a, b, c, d, e, f, g, h]  (8 elements, 4 threads)
Level 1: [a+b, c+d, e+f, g+h]      (4 elements, 2 threads)
Level 2: [a+b+c+d, e+f+g+h]        (2 elements, 1 thread)
Level 3: [total]                    (1 element)
```

**This is advanced.** Start by understanding the current kernel, then study:
- [Triton reduce tutorial](https://triton-lang.org/main/getting-started/tutorials/04-reduction.html)
- [NVIDIA parallel reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

**Why this assignment:** Tree reduction is a performance lesson, not a framework
architecture lesson. It belongs here only after the simple reduce path is correct.

---
