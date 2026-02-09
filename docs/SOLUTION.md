This is a solution (by me) for a course on banhxeo pipeline, you can use this as a reference when diving into banhxeo (Thanks Claude and Gemini for helping me checking the solution).

---

## Module 0: Understand the Current Codebase

### Assigment 0.1: Trace a Simple Operation ⭐

1. Every Tensor operation (from creation to arithmetic to view manipulation) all create **LazyBuffer** objects. In the example code, we have three LazyBuffer object:

```python
a = LazyBuffer(LoadOp.FROM_PYTHON, src=(), ...) 
b = LazyBuffer(LoadOp.FROM_PYTHON, src=(), ...) 
c = LazyBuffer(BinaryOp.ADD, src=(c, a, b), ...)
```

2. From the [Architecture Note](./ARCHITECTURE.md), when we call `Tensor.realize()`, a pipeline will be executed, from Backend dispatcher and "setup" to `TritonCodegen.generate()` for final Triton code generation.

3. The Triton generated code (when running with `DEBUG=2`):
```python
# --- [DEBUG] GENERATED TRITON KERNEL (BINARYOP.ADD) ---
@triton.heuristics(values={'BLOCK_SIZE': lambda args: min(triton.next_power_of_2(args['N']), 1024)})
@triton.jit
def generated_kernel(in_0_ptr, in_1_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    temp_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    linear_offsets = temp_idx
    linear_mask = linear_offsets < N
    in_0 = tl.load(in_0_ptr + linear_offsets, mask=linear_mask)
    in_1 = tl.load(in_1_ptr + linear_offsets, mask=linear_mask)
    temp_2 = in_0 + in_1
    tl.store(out_ptr + linear_offsets, temp_2, mask=linear_mask)
```

4. The Triton generated code is compiled with `compile_triton_src` function in `CUDABackend` class. This is a hack to "JIT" a Jitted function (we cache the kernel in string first then compile it later). After compilation, generated Triton code will become a valid function and therfore it can be used for computation.

### Assignment 0.2: Trace a View Operation ⭐⭐