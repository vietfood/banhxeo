## Appendix A: tinygrad Code Map

### Essential Files

| File | Purpose | Read When |
|------|---------|-----------|
| `CLAUDE.md` | Architecture overview | First! |
| `tinygrad/tensor.py` | User API | Understanding API design |
| `tinygrad/uop/ops.py` | UOp IR | Building your IR |
| `tinygrad/uop/__init__.py` | Op definitions | Adding new ops |
| `tinygrad/shape/view.py` | View implementation | Module 1 |
| `tinygrad/shape/shapetracker.py` | View composition | Module 1 |
| `tinygrad/schedule/indexing.py` | Movement lowering | Module 1 |
| `tinygrad/engine/schedule.py` | Kernel creation | Module 3 |
| `tinygrad/schedule/rangeify.py` | Fusion logic | Module 3 |
| `tinygrad/renderer/cstyle.py` | Code generation | Module 2 |
| `tinygrad/gradient.py` | Autograd rules | Module 5 |

### Study Order

1. `CLAUDE.md` - High-level architecture
2. Trace `Tensor.realize()` through the codebase
3. Study `UOp` class in `uop/ops.py`
4. Understand `create_schedule()` in `engine/schedule.py`
5. Read `CStyleLanguage.render()` in `renderer/cstyle.py`

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

### Module 2: Intermediate Representation & Codegen
- [ ] 2.1 Analyze current codegen problems
- [ ] 2.2 Understand tinygrad's UOp design
- [ ] 2.3 Design your KernelOp IR
- [ ] 2.4 Implement IRBuilder
- [ ] 2.5 Implement Triton Renderer
- [ ] 2.6 Implement optimization passes (constant folding, DCE)

### Module 3: Scheduling & Kernel Fusion
- [ ] 3.1 Understand banhxeo's current scheduling
- [ ] 3.2 Formalize barrier rules
- [ ] 3.3 Understand tinygrad's scheduling
- [ ] 3.4 Implement fusion analysis
- [ ] 3.5 Add fusion logging
- [ ] 3.6 Research advanced fusion (optional)

### Module 4: Specialized Kernels
- [ ] 4.1 Annotate matmul kernel
- [ ] 4.2 Implement batched matmul
- [ ] 4.3 Implement multi-axis reduce
- [ ] 4.4 Tree reduction (optional)

### Module 5: Autograd
- [ ] 5.1 Gradient verification tests
- [ ] 5.2 Movement op gradients
- [ ] 5.3 Broadcasting gradient accumulation

---

Good luck! Remember: the best way to learn is to implement, get stuck, then discover why the proper solution works better.
