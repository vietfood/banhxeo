## Module 2: Forward Correctness

Make banhxeo prove its forward behavior against PyTorch.

## Why This Module Comes Before Autograd

Autograd depends on forward semantics.

If `expand`, `slice`, `permute`, `sum`, or broadcasting are wrong, gradient tests
will fail in noisy ways. You will waste time debugging backward formulas when
the forward value was already broken.

This module is not about making banhxeo production-correct. It is about building
a small harness that catches semantic changes before scheduling, IR, and kernel
work make bugs harder to localize.

## Assignment 2.1: Build A Forward Comparison Helper ⭐⭐

**Task:** Add a helper that compares a banhxeo expression against PyTorch.

The helper should:

- create identical inputs for banhxeo and PyTorch
- run the same operation
- realize the banhxeo output
- compare shape, dtype where relevant, and values
- use small tolerances for floating point ops

Start with CUDA for the real compiler path. Keep CPU support if it helps you
debug faster, but CUDA correctness is the checkpoint that matters.

**Why this assignment:** Reusable comparison helpers prevent each test from
inventing its own correctness standard.

## Assignment 2.2: Test Elementwise And Unary Ops ⭐⭐

Test groups:

- elementwise: add, sub, mul, div, max, less, where
- unary: neg, exp, log, sin, sqrt
- scalar interactions: tensor + scalar, scalar - tensor, tensor / scalar

Use asymmetric shapes:

```python
(2, 3)
(3, 1)
(1, 4)
(2, 3, 4)
```

Avoid only testing `(2, 2)`. Symmetric shapes hide bugs.

**Why this assignment:** Elementwise ops are the baseline for fusion and IR. If
these are not trusted, later generated kernels have no stable target behavior.

## Assignment 2.3: Test Movement Ops ⭐⭐⭐

Test:

- reshape
- permute
- transpose
- expand
- slice
- chained views like `x[1:][:, 1:].T`

For every test, record the expected `shape`, `strides`, and `offset` when the
answer is not obvious.

**Why this assignment:** Movement ops are where tensor frameworks stop being
"arrays with operators" and become storage interpretation systems.

## Assignment 2.4: Test Reductions And Matmul ⭐⭐

Test:

- `sum(axis=...)`
- `max(axis=...)`
- `mean(axis=...)`
- 2D matmul

Keep multi-axis reduce as an expected failure until Module 6 if it is not
implemented yet.

**Why this assignment:** Reductions and matmul are scheduling boundaries later.
Before treating them as boundaries, prove their standalone behavior.

## Checkpoint

At the end of this module, you should know:

1. Which forward ops currently match PyTorch.
2. Which failures are view/indexing bugs.
3. Which failures are unsupported features.
4. Which failures need to become explicit `xfail` tests.

Do not start autograd cleanup until this list is written down.
