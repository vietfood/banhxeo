## Module 9: Compiler Optimizations

This module is advanced. Do it only after the IR is stable enough to preserve
correctness.

## Why This Module Exists

Optimization is not random cleverness. It is semantics-preserving program
rewriting under constraints.

Before this module, banhxeo should already have:

- forward correctness tests
- gradient correctness tests
- explicit scheduling rules
- a basic IR
- at least one renderer

Now you can start asking whether the compiler can make the program better
without changing what it means.

## Assignment 9.1: Constant Folding ⭐⭐

Fold operations whose inputs are compile-time constants:

```text
v0 = const 2
v1 = const 3
v2 = add v0, v1
```

becomes:

```text
v2 = const 5
```

**Why this assignment:** Constant folding is the smallest useful optimization.
It teaches rewrite mechanics without requiring a cost model.

## Assignment 9.2: Dead Code Elimination ⭐⭐

Remove IR values that do not affect the final store.

Questions:

1. Which ops have side effects?
2. Is `STORE` always live?
3. Can `LOAD` be deleted if its result is unused?

**Why this assignment:** DCE forces you to model use-def chains and side effects.

## Assignment 9.3: Common Subexpression Elimination ⭐⭐⭐

Detect duplicate pure expressions:

```text
v0 = add a, b
v1 = add a, b
v2 = mul v0, v1
```

and reuse the first result:

```text
v0 = add a, b
v2 = mul v0, v0
```

**Why this assignment:** CSE teaches structural equality, purity, and the cost
of pretending all ops are interchangeable.

## Assignment 9.4: Algebraic Simplification ⭐⭐⭐

Implement only boring identities:

- `x + 0 -> x`
- `x * 1 -> x`
- `x * 0 -> 0`
- `x - 0 -> x`

Avoid unsafe floating-point rewrites unless you can explain their numerical
tradeoff.

**Why this assignment:** Algebraic rewrites look easy and become wrong fast.
Floating point is not symbolic math.

## Assignment 9.5: Layout-Aware Rewrites ⭐⭐⭐

Use view metadata to decide whether a rewrite is legal or useful.

Examples:

- remove redundant contiguous calls
- fold adjacent reshapes when legal
- avoid materializing a view when indexing can represent it

**Why this assignment:** Tensor compiler optimization is not just scalar
algebra. Layout and memory access are part of the semantics.

## Assignment 9.6: Tiny Cost Model ⭐⭐⭐

Write a rough estimator for:

- memory reads
- memory writes
- approximate flops
- number of kernels

Use it only for logging at first.

**Why this assignment:** A compiler needs a reason to choose one lowering over
another. A bad visible cost model is more educational than hidden vibes.
