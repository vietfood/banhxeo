# The Banhxeo Performance Dream 🚀

> "Make it work, then make it right, then make it fast." — We are here.

This document outlines the advanced features and architectural changes needed to transform **banhxeo** from an educational toy into a performant inference backend for `nanovLLM`.

## 1. Advanced Kernel Fusion

Currently, we fuse elementwise operations. To reach vLLM speeds, we need to go deeper.

- [ ] **Symbolic Shapes**: Stop recompiling kernels for every new batch size. Implement symbolic shape tracking (e.g., `Shape((Variable("B"), 784))`) so one kernel handles all matching shapes.
- [ ] **Aggressive Fusion**: Fuse `ReduceOp` with elementwise successors. (e.g., `(x.sum() + 1).exp()` should be one kernel).
- [ ] **Pattern Matching**: Detect common subgraphs (like `x * sigmoid(x)` -> `swish`) and emit specialized optimized Triton assembly.

## 2. Memory Management

Allocating `torch.Tensor` (our current underlying buffer) has overhead.

- [ ] **Bypass PyTorch Allocator**: Use `cudaMalloc` directly via `ctypes` or `triton.driver` for raw memory control.
- [ ] **Buffer Reuse / Arena Allocator**: Implement a memory pool. When `z = x + y` is computed and `x` is never used again, reuse `x`'s memory for `z` if shapes match.
- [ ] **In-Place Ops**: Support true in-place mutations in the graph to save VRAM.

## 3. Triton Tuning & Specialization

Our current Triton generation is "one size fits all".

- [ ] **Autotuning**: Use `triton.autotune` to find the best `BLOCK_SIZE`, `num_warps`, and `num_stages` for specific hardware at runtime.
- [ ] **Hardware-Specific Heuristics**: Hand-tuned configs for A100 vs 4090.
- [ ] **Flash Attention**: Implement a fused `FlashAttention` kernel. This is mandatory for efficient LLM inference.
- [ ] **Paged Attention**: The "vLLM" secret sauce. Implement KV-cache blocking in Triton.

## 4. Graph Optimizations

Before generating code, optimize the `LazyBuffer` graph.

- [ ] **Common Subexpression Elimination (CSE)**: If `a + b` is computed twice, compute it once.
- [ ] **Constant Folding**: Pre-compute operations on `LoadOp.CONST` during graph build time.
- [ ] **Dead Code Elimination**: Prune branches of the graph that don't contribute to the realized result.

## 5. The "vLLM" Specifics

To support `nanovLLM`:

- [ ] **FP16 / BF16 Support**: Essential for LLM weights. Ensure dtype promotion rules are rock solid.
- [ ] **Quantization Kernels**: Implement 4-bit / 8-bit weight unpacking kernels (GPTQ/AWQ style).
- [ ] **KV Cache Manager**: specialized buffer management for the autoregressive loop.

## 6. Distributed / Multi-GPU

The ultimate performance frontier.

- [ ] **Sharded Tensors**: `Tensor.shard(devices=[0, 1], axis=0)`.
- [ ] **Ring Reduce**: Implement distributed communication primitives.

---

*This is the path from "Crispy Pancake" to "Jet Engine".*
