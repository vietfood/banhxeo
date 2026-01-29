# banhxeo Development Roadmap

## Current Status

**Version:** v0.3 (Autograd Engine - In Progress)

### What Works

- [x] Basic ops: `+`, `-`, `*`, `/`, `log`, `exp`, `sin`, `sqrt`, `neg`
- [x] Comparison ops: `<`, `>`, `<=`, `>=`
- [x] Ternary: `where`
- [x] Lazy evaluation with computation graphs
- [x] View operations: `permute`, `slice`, `expand` (zero-copy!)
- [x] Broadcasting (naive but works)
- [x] Triton codegen for GPU execution
- [x] PyTorch interpreter for CPU execution
- [x] Basic autograd with `backward()`
- [x] Reduce ops: `sum`, `max`, `min`, `mean`
- [x] Matmul via Triton kernel

### Known Issues (See CHECKLIST.md for details)

- `Tensor.maximum()` ignores `self` - signature bug
- `__radd__`/`__rsub__` fail with scalars
- `Tensor.T` assertion is broken
- TorchInterpreter missing ReduceOp handling
- `nn/optimizer.py` is empty (SGD is in tests/)
- Some tests call non-existent methods (`log2`, `exp2`, `relu`)

---

## Roadmap

### v0.2: First Matmul (COMPLETED)

- [x] Solve the Reshape Boss: Implement the `contiguous()` check and the reshape logic
- [x] Naive Matmul: Simple Triton kernel (block size 32, no fancy pipelining)
- [x] The MLP Forward Pass: `x @ W + b` verified against PyTorch

### v0.3: Autograd Engine (IN PROGRESS)

**Backward Ops:**
- [x] Unary ops backward
- [ ] Binary ops backward (partial - need to verify broadcasting)
- [ ] Movement ops backward (partial - need to verify all)
- [x] Ternary ops backward

**Other:**
- [x] Reduce ops: `sum` and `max`
- [ ] Proper broadcasting for gradient flow
- [x] Creation methods (`LoadOp.RAND`)

### v0.4: Bug Fixes & Stabilization (NEW)

**Critical Fixes:**
- [ ] Fix `Tensor.maximum()` signature
- [ ] Fix `__radd__`/`__rsub__` for scalar operands
- [ ] Fix `Tensor.T` assertion (`len(self.shape) == 2`)
- [ ] Add ReduceOp handling to TorchInterpreter

**Missing Methods:**
- [ ] `Tensor.relu()` - wrapper around `maximum(0)`
- [ ] `Tensor.sigmoid()` - `1 / (1 + exp(-x))`
- [ ] `Tensor.tanh()` - `2 * sigmoid(2x) - 1`
- [ ] `Tensor.clamp(min, max)` - using `maximum`/`minimum`
- [ ] `Tensor.squeeze()` / `Tensor.unsqueeze()`

**Test Infrastructure:**
- [ ] Fix broken tests (`log2`, `exp2` calls)
- [ ] Add `conftest.py` with device fixtures
- [ ] Standardize device strings (lowercase)

### v0.5: Neural Network Module

**Layers:**
- [ ] `nn.Linear(in_features, out_features, bias=True)`
- [ ] `nn.ReLU`
- [ ] `nn.Sigmoid`
- [ ] `nn.Softmax(dim)`
- [ ] `nn.Flatten`

**Optimizers:**
- [ ] Move SGD from `tests/` to `nn/optimizer.py`
- [ ] Clean up SGD implementation
- [ ] Add momentum support to SGD
- [ ] Adam optimizer (stretch goal)

**Loss Functions:**
- [ ] Verify `mse_loss` works with gradients
- [ ] Verify `cross_entropy` works (one-hot hack acceptable for now)
- [ ] Add `nll_loss`

### v0.6: MNIST MLP (THE FIRST MILESTONE)

- [ ] Simple MNIST data loader (numpy/torchvision)
- [ ] Build MLP: `Linear(784, 128) -> ReLU -> Linear(128, 10)`
- [ ] Training loop with SGD
- [ ] Verify loss decreases
- [ ] **Goal: >90% accuracy on MNIST**
- [ ] Create `examples/mnist_mlp.py`

### v0.7: CUDA Hardening

- [ ] Fix Triton heuristics syntax error
- [ ] Verify reduce kernels have `@triton.jit` decorators
- [ ] Run all tests on both CPU and CUDA
- [ ] Test kernel fusion (`(a+b)*c` = 1 kernel)
- [ ] Add `Tensor.cuda()` / `Tensor.cpu()` methods

### v0.8: Convolution (STRETCH GOAL)

- [ ] Implement `nn.functional.unfold` (im2col)
- [ ] `nn.Conv2d(in_channels, out_channels, kernel_size)`
- [ ] `nn.MaxPool2d`
- [ ] `nn.AvgPool2d`
- [ ] Train CNN on MNIST
- [ ] **Goal: >98% accuracy on MNIST**

### v1.0: The Dream (nanovLLM Foundation)

- [ ] All ops stable on CPU and CUDA
- [ ] Clean, PyTorch-like API
- [ ] Working MNIST example(s)
- [ ] CI/CD with GitHub Actions
- [ ] Documentation and docstrings
- [ ] Ready to explore LLM inference

---

## Known Limitations (Acceptable for v1.0)

These are intentionally deferred to keep the codebase simple:

| Limitation | Reason |
|------------|--------|
| No negative strides (`x[::-1]`) | Requires flip kernel |
| No advanced indexing | Boolean/integer array indexing is complex |
| No slice assignment (`x[1:3] = y`) | Requires scatter kernel |
| No GATHER kernel | Using one-hot workaround for cross_entropy |
| No PAD/SHRINK ops | Enum exists but not implemented |

---

## Development Notes

### Weight Updates

If you need to *update weights*, update the whole tensor via `assign`:
```python
weight.assign(weight - lr * weight.grad)
```

If you need to *manipulate specific values*, use masks:
```python
x = x * mask + val * (1 - mask)
```

### Debugging

Set `DEBUG=1` to see generated Triton kernels:
```bash
DEBUG=1 python your_script.py
```

### Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/compute_ops.py::TestBinaryOps::test_add
```

---

## Architecture Reference

```
User API          Tensor (tensor.py)
                     │
                     ▼
Autograd          Function (function.py)
                     │
                     ▼
Lazy Graph        LazyBuffer (buffer.py) ◄── View (view.py)
                     │
                     ▼
Backend           CPUBackend ──► TorchInterpreter (torch.py)
                  CUDABackend ──► TritonCodegen (triton.py)
                                      │
                                      ▼
Kernels           matmul.py, reduce.py (specialized)
```
