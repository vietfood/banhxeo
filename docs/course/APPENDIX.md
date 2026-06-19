## Appendix A: tinygrad Code Map

### Essential Files

| File | Purpose | Read When |
|------|---------|-----------|
| `AGENTS.md` | Architecture overview | First! |
| `tinygrad/tensor.py` | User API | Understanding API design |
| `tinygrad/uop/ops.py` | UOp IR | Building your IR |
| `tinygrad/uop/__init__.py` | Op definitions | Adding new ops |
| `tinygrad/shape/view.py` | View implementation | Module 1 |
| `tinygrad/shape/shapetracker.py` | View composition | Module 1 |
| `tinygrad/schedule/indexing.py` | Movement lowering | Module 1 |
| `tinygrad/gradient.py` | Autograd rules | Module 3 |
| `tinygrad/engine/schedule.py` | Kernel creation | Module 4 |
| `tinygrad/schedule/rangeify.py` | Fusion logic | Module 4 |
| `tinygrad/renderer/cstyle.py` | Code generation | Module 5 |
| `tinygrad/uop/ops.py` | Advanced UOp IR | Module 8 |

### Study Order

1. `CLAUDE.md` - high-level architecture
2. Trace `Tensor.realize()` through the codebase
3. Study `shape/view.py` and `shape/shapetracker.py`
4. Understand `create_schedule()` in `engine/schedule.py`
5. Read `uop/ops.py` only after banhxeo needs an IR
6. Read `renderer/cstyle.py` after your first renderer works

Why this order:

Do not start with UOps. UOps are tinygrad's answer after many design pressures
have accumulated. Start with views and scheduling so you know what pressure your
own code is under.

---

## Appendix B: Debugging Tips

**"Cannot reshape non-contiguous view"**
- Solution: Add `.contiguous()` before reshape
- Root cause: Non-contiguous strides can't always be reshaped in-place

**Kernel produces wrong values**
- Check indexing math in generated code
- Verify stride computation for non-contiguous inputs
- Add `DEBUG=1` to see generated kernel

**Kernel fails to compile**
- Check generated code syntax
- Ensure all variables are defined before use
- Look for type mismatches

---

## Appendix C: Checklist by Module

### Module 1: View & Indexing
- [ ] 1.1 Implement `View.to_index_expr()`
- [ ] 1.2 Optimize for contiguous case
- [ ] 1.3 Handle view composition correctly
- [ ] 1.4 Implement `View.pad()` with mask
- [ ] 1.5 Implement `View.shrink()`
- [ ] 1.6 Implement `View.flip()` with negative strides

### Module 2: Forward Correctness
- [ ] 2.1 Build a forward comparison helper
- [ ] 2.2 Test elementwise and unary ops
- [ ] 2.3 Test movement ops
- [ ] 2.4 Test reductions and matmul

### Module 3: Autograd
- [ ] 3.1 Trace the current backward pass
- [ ] 3.2 Implement gradient comparison tests
- [ ] 3.3 Implement movement op gradients
- [ ] 3.4 Verify broadcasting gradient accumulation

### Module 4: Scheduling & Kernel Fusion
- [ ] 4.1 Understand banhxeo's current scheduling
- [ ] 4.2 Formalize barrier rules
- [ ] 4.3 Understand tinygrad's scheduling
- [ ] 4.4 Implement fusion analysis
- [ ] 4.5 Add fusion logging
- [ ] 4.6 Research advanced fusion (optional)

### Module 5: Basic Kernel IR And Codegen
- [ ] 5.1 Analyze current codegen problems
- [ ] 5.2 Understand tinygrad's UOp design
- [ ] 5.3 Design your KernelOp IR
- [ ] 5.4 Implement IRBuilder
- [ ] 5.5 Implement Triton Renderer
- [ ] 5.6 Implement optimization passes (constant folding, DCE)

### Module 6: Specialized Kernels
- [ ] 6.1 Annotate matmul kernel
- [ ] 6.2 Implement batched matmul
- [ ] 6.3 Implement multi-axis reduce
- [ ] 6.4 Tree reduction (optional)

### Module 8: Advanced IR And Multi-Backend Lowering
- [ ] 8.1 Separate graph IR from kernel IR
- [ ] 8.2 Add typed IR values
- [ ] 8.3 Add a target capability object
- [ ] 8.4 Write a toy second renderer
- [ ] 8.5 Complete the MLIR reading checkpoint

### Module 9: Compiler Optimizations
- [ ] 9.1 Constant folding
- [ ] 9.2 Dead code elimination
- [ ] 9.3 Common subexpression elimination
- [ ] 9.4 Algebraic simplification
- [ ] 9.5 Layout-aware rewrites
- [ ] 9.6 Tiny cost model

---

Good luck! Remember: the best way to learn is to implement, get stuck, then discover why the proper solution works better.
