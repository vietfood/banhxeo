# banhxeo Architecture Reference

Quick reference for the compilation pipeline. For learning, see `COURSE.md`.

---

## banhxeo Pipeline

```
Tensor.realize()
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                      Backend.exec()                          │
├──────────────────────────────────────────────────────────────┤
│  1. schedule()     ──►  Topological sort of LazyBuffer DAG   │
│  2. get_barriers() ──►  Find ops that need separate kernels  │
│  3. exec_*()       ──►  Generate and run kernel              │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                   TritonCodegen.generate()                    │
├──────────────────────────────────────────────────────────────┤
│  1. Visit each LazyBuffer in schedule                        │
│  2. Generate variable names and input arguments              │
│  3. Emit Triton code for each op                             │
│  4. Handle strided access via render_indexing()              │
│  5. Return kernel source string                              │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                   Kernel Execution                            │
├──────────────────────────────────────────────────────────────┤
│  - compile_triton_src() ──► JIT compile via linecache trick  │
│  - kernel_cache        ──► Hash-based caching                │
│  - Specialized kernels ──► matmul.py, reduce.py              │
└──────────────────────────────────────────────────────────────┘
```

---

## tinygrad Pipeline (Reference)

```
Tensor.realize()
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                   Schedule Creation                           │
├──────────────────────────────────────────────────────────────┤
│  1. Rangeify       ──►  Convert movements to loop indices    │
│  2. Split Kernels  ──►  Identify fusion boundaries           │
│  3. Memory Plan    ──►  Buffer allocation/reuse              │
│  4. Toposort       ──►  Execution order                      │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                   Linearization                               │
├──────────────────────────────────────────────────────────────┤
│  - Convert UOp DAG to linear sequence                        │
│  - Insert RANGE/END for loops                                │
│  - Insert IF/ENDIF for conditionals                          │
│  - Add DEFINE_REG for accumulators                           │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                   Optimization Passes                         │
├──────────────────────────────────────────────────────────────┤
│  - Algebraic simplification                                  │
│  - Load/store folding                                        │
│  - Range optimization                                        │
│  - Pattern matching rewrites                                 │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                   Rendering                                   │
├──────────────────────────────────────────────────────────────┤
│  - Device-specific code generation                           │
│  - CUDA/PTX/Metal/OpenCL backends                            │
│  - PatternMatcher-based string rewriting                     │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Concepts from tinygrad

| Concept | Description |
|---------|-------------|
| **UOp** | Universal Operation - single IR for everything, immutable and deduplicated |
| **ShapeTracker** | Tracks tensor memory layout through View objects, enables zero-copy movements |
| **Rangeify** | Converts movement ops (reshape, permute, expand) into index expressions with explicit loops |
| **PatternMatcher** | Graph rewriting engine for optimization passes |

### Kernel Boundaries in tinygrad
- Operations within same "outer range" are fusible
- `COPY`, `BUFFER_VIEW` force kernel boundaries
- All buffers in a kernel must be on same device

---

## File Map

| Component | File | Key Classes/Functions |
|-----------|------|----------------------|
| Tensor API | `tensor.py` | `Tensor`, `backward()`, `realize()` |
| Lazy Evaluation | `core/buffer.py` | `LazyBuffer`, `RawBuffer`, Op enums |
| View System | `core/view.py` | `View` |
| Autograd | `core/function.py` | `Function`, op-specific classes |
| CPU Backend | `backend/torch.py` | `TorchInterpreter` |
| CUDA Backend | `backend/__init__.py` | `CUDABackend`, `exec()` |
| Codegen | `backend/triton.py` | `TritonCodegen`, `render_indexing()` |
| Matmul Kernel | `backend/kernels/matmul.py` | `matmul_kernel` |
| Reduce Kernel | `backend/kernels/reduce.py` | `reduce_sum_kernel`, `reduce_max_kernel` |

---

## Debug Levels

```bash
DEBUG=1 python script.py  # Print generated kernel source
DEBUG=2 python script.py  # + Schedule visualization
DEBUG=3 python script.py  # + Kernel execution time
DEBUG=4 python script.py  # + Intermediate buffer values
```

---

## tinygrad Files to Study

| Priority | File | Purpose |
|----------|------|---------|
| ⭐⭐⭐ | `CLAUDE.md` | Architecture overview |
| ⭐⭐⭐ | `tinygrad/tensor.py` | User API, `realize()` |
| ⭐⭐⭐ | `tinygrad/uop/ops.py` | UOp IR definition |
| ⭐⭐ | `tinygrad/shape/view.py` | View implementation |
| ⭐⭐ | `tinygrad/engine/schedule.py` | Scheduling |
| ⭐⭐ | `tinygrad/schedule/rangeify.py` | Fusion logic |
| ⭐⭐ | `tinygrad/renderer/cstyle.py` | Code generation |
| ⭐ | `tinygrad/gradient.py` | Autograd rules |
