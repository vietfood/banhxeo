## Module 8: Advanced IR And Multi-Backend Lowering

This module is advanced. Do it only after the basic IR path works.

## Why This Module Exists

Module 5 gives you a tiny IR for one fused elementwise kernel. That is useful,
but it is not enough for a production compiler.

As banhxeo grows, one IR will start carrying too many jobs:

- graph-level tensor semantics
- kernel-level load/store behavior
- shape and dtype inference
- device capabilities
- memory effects
- lowering to different backend languages

This is the pressure that leads production systems toward multi-level IRs like
TVM's Relax/TensorIR split or MLIR dialects. Do not copy MLIR yet. Use this
module to understand why MLIR-shaped ideas exist.

## Assignment 8.1: Separate Graph IR From Kernel IR ⭐⭐⭐

Write a design note that separates:

- graph IR: tensor ops, shapes, fusion candidates, high-level semantics
- kernel IR: scalar/vector ops, loads, stores, masks, offsets

Questions:

1. Which current `LazyBuffer` concepts belong to graph IR?
2. Which concepts belong to kernel IR?
3. Where should shape inference live?
4. Where should device-specific lowering begin?

**Why this assignment:** A single IR is simple until it becomes a junk drawer.
This assignment teaches where the split should happen.

## Assignment 8.2: Add Typed IR Values ⭐⭐

Extend the basic IR design with explicit value metadata:

- name
- dtype
- shape or scalar/vector rank
- producer op
- optional memory effect

Do not implement a full type system. Add only what a renderer or verifier
actually needs.

**Why this assignment:** Untyped IR is easy to build and easy to break. Typed
values let passes reject impossible rewrites before codegen.

## Assignment 8.3: Add A Target Capability Object ⭐⭐

Design a small target description:

```python
Target(
    name="triton",
    supports_float16=True,
    supports_bfloat16=True,
    max_block_size=1024,
    address_space_model="cuda",
)
```

Use it to answer:

1. Can this op lower to this backend?
2. Does this dtype exist on the target?
3. Does this renderer need masks, bounds checks, or special intrinsics?

**Why this assignment:** Multi-backend support is not just another renderer. The
compiler must know what each backend can legally express.

## Assignment 8.4: Write A Toy Second Renderer ⭐⭐⭐

Pick one small target:

- Python pseudo-code
- NumPy loop code
- Metal-like pseudo-code
- C-like scalar loop

Render only a tiny subset: load, const, add, mul, store.

**Why this assignment:** A second renderer exposes whether your IR is genuinely
backend-neutral or secretly Triton-shaped.

## Assignment 8.5: MLIR Reading Checkpoint ⭐⭐⭐

Read about MLIR dialects, operations, types, and lowering.

Write down:

1. What problem dialects solve.
2. What banhxeo would gain from MLIR.
3. What banhxeo would lose by adopting MLIR too early.
4. Which banhxeo IR concepts map naturally to MLIR.

**Why this assignment:** MLIR is production-grade infrastructure, but it is also
heavy. You should migrate to it only after banhxeo has enough compiler pressure
to justify the complexity.
