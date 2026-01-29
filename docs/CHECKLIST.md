# banhxeo v1.0 Stabilization Checklist

**Goal:** A working, trainable MNIST MLP (then CNN) with a PyTorch-like API

**Philosophy:** CPU-first (easier debugging), then CUDA; interleave bugfixes with features

---

## Quick Win Checklist (Do First!)

These are the bugs that would unblock most work:

1. [ ] `tensor.py:177` - Fix `maximum()` signature
2. [ ] `tensor.py:417,426` - Fix `__radd__`/`__rsub__`
3. [ ] `tensor.py:339` - Fix `.T` assertion
4. [ ] `backend/torch.py` - Add ReduceOp handling
5. [ ] Move SGD from test file to `nn/optimizer.py`
6. [ ] Add `Tensor.relu()` method

---

## Phase 0: Foundation Fixes (Critical Bugs)

*These bugs will block almost everything else*

### 0.1 Core Bugs in tensor.py

- [ ] **Fix `Tensor.maximum()` signature** - `tensor.py:177-180`
  - Currently: `Maximum.apply(x)` - ignores `self`!
  - Should be: `Maximum.apply(self, x)`
- [ ] **Fix `__radd__` and `__rsub__`** - `tensor.py:417,426`
  - Currently: `other.add(self)` - fails when `other` is a scalar
  - Should be: `self.add(other)` / `(-self).add(other)`
- [ ] **Fix `Tensor.T` assertion** - `tensor.py:339`
  - Currently: `self.shape == 2` (compares tuple to int)
  - Should be: `len(self.shape) == 2`

### 0.2 TorchInterpreter (CPU Backend) Fixes

- [ ] **Add ReduceOp handling** to `backend/torch.py`
  - Currently missing, will fail silently on CPU reduce ops
  - Implement `ReduceOp.SUM` and `ReduceOp.MAX`
- [ ] **Add MovementOp handling** if missing

### 0.3 Test Infrastructure

- [ ] **Add `conftest.py`** with device fixture that auto-skips CUDA tests on CPU-only
- [ ] **Fix device string inconsistency** - standardize to lowercase `"cuda"` or `"cpu"`
- [ ] **Remove/fix broken tests** (`log2`, `exp2`, `relu` method calls that don't exist)

---

## Phase 1: Core Tensor Operations

*Make tensor ops robust and PyTorch-compatible*

### 1.1 Missing Tensor Methods

- [ ] **Add `Tensor.relu()`** method (wrapper around `maximum(0)` or `where`)
- [ ] **Add `Tensor.sigmoid()`** method
- [ ] **Add `Tensor.tanh()`** method (can derive: `2 * sigmoid(2x) - 1`)
- [ ] **Add `Tensor.log2()` and `Tensor.exp2()`** (or remove from tests)
- [ ] **Add `Tensor.pow(n)`** for integer/float exponents
- [ ] **Add `Tensor.clamp(min, max)`** using `maximum/minimum`

### 1.2 Implement `Tensor.slice()` Method

- [ ] **Complete empty slice method** - `tensor.py:328-331`
  - Currently just has a docstring with TODO
  - Implement using View.slice() and LazyBuffer operations

### 1.3 Division Operation

- [ ] **Verify `Tensor.div()` works** - add tests
- [ ] **Add `Tensor.reciprocal()`** (`1/x`) if useful

### 1.4 Comparison Operations

- [ ] **Add `Tensor.__eq__()` and `Tensor.eq()`**
- [ ] **Add `Tensor.__ne__()` and `Tensor.ne()`**
- [ ] **Verify all comparison ops work:** `<`, `>`, `<=`, `>=`, `==`, `!=`

---

## Phase 2: Shape Operations (Movement Ops)

*Critical for building neural network layers*

### 2.1 Reshape & Contiguous

- [ ] **Fix reshape implementation** - currently documented as NO-OP
  - `view.py` `reshape()` needs to handle non-contiguous case
  - Should call `contiguous()` when stride-based reshape fails
- [ ] **Verify `contiguous()` forces physical copy**
- [ ] **Test reshape chains:** `reshape -> op -> reshape`

### 2.2 Tensor Creation from Nested Lists

- [ ] **Fix Tensor init for multi-dimensional data**
  - Currently noted as buggy in test HACK comment
  - `Tensor([[1,2],[3,4]])` should infer `shape=(2,2)` automatically

### 2.3 Indexing Improvements

- [ ] **Support boolean indexing** (lower priority, but nice-to-have)
- [ ] **Handle ellipsis (`...`)** - `tensor.py:350` TODO

### 2.4 Squeeze/Unsqueeze

- [ ] **Add `Tensor.squeeze(dim=None)`** - remove dims of size 1
- [ ] **Add `Tensor.unsqueeze(dim)`** - add dim of size 1

---

## Phase 3: Autograd Completion

*Backward passes needed for training*

### 3.1 Binary Op Backwards (marked partial in DEVELOPMENT.md)

- [ ] **Verify `Add.backward`** handles broadcasting gradient correctly
- [ ] **Verify `Sub.backward`**
- [ ] **Verify `Mul.backward`** with broadcasting
- [ ] **Verify `Div.backward`**
- [ ] **Add `Matmul.backward`** - critical for training!

### 3.2 Movement Op Backwards (marked partial)

- [ ] **Verify `Reshape.backward`** reverses shape
- [ ] **Verify `Permute.backward`** inverses permutation
- [ ] **Verify `Expand.backward`** sums along expanded dims

### 3.3 Reduce Op Backwards

- [ ] **Verify `Sum.backward`** broadcasts gradient
- [ ] **Add `Mean.backward`** (Sum / count)
- [ ] **Verify `Max.backward`** with argmax gradient routing

### 3.4 Gradient Accumulation

- [ ] **Test gradient accumulation** when tensor used multiple times
- [ ] **Test `Tensor.detach()`** stops gradients

---

## Phase 4: Neural Network Module (`nn/`)

*Building blocks for MNIST*

### 4.1 Move SGD to Proper Location

- [ ] **Move `SGD` from `tests/small_tests/mnist.py` to `nn/optimizer.py`**
- [ ] **Clean up SGD implementation** - currently manipulates `.lazydata.realized.data` directly

### 4.2 Add More Optimizers (optional, but useful)

- [ ] **Add momentum to SGD**
- [ ] **Add Adam optimizer** (stretch goal)

### 4.3 Linear Layer

- [ ] **Add `nn.Linear(in_features, out_features, bias=True)`**
  - Move from `tests/small_tests/mnist.py` to `nn/__init__.py` or `nn/linear.py`
- [ ] **Proper weight initialization** with kaiming_uniform

### 4.4 Activation Layers

- [ ] **Add `nn.ReLU`** module
- [ ] **Add `nn.Sigmoid`** module
- [ ] **Add `nn.Softmax(dim)`** module

### 4.5 Loss Functions

- [ ] **Test `nn.functional.mse_loss`** - add gradient test
- [ ] **Test `nn.functional.cross_entropy`**
  - Currently uses HACK one-hot due to missing GATHER
  - Keep hack for now, document limitation
- [ ] **Add `nn.functional.nll_loss`** (for use with log_softmax)

### 4.6 Module Base Class

- [ ] **Test `Module.parameters()`** returns all nested params
- [ ] **Test `Module.zero_grad()`** clears all gradients
- [ ] **Add `Module.to(device)`** for device transfer

---

## Phase 5: MNIST MLP Training

*The first milestone!*

### 5.1 Data Loading

- [ ] **Create simple MNIST loader** (use torchvision or manual numpy)
  - Load as numpy, convert to banhxeo Tensor
  - Batch iterator
- [ ] **Add `Tensor.from_numpy()`** if not working
- [ ] **Add `Tensor.numpy()`** verification

### 5.2 Build MLP Model

```python
# Target architecture
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

- [ ] **Test forward pass matches PyTorch**
- [ ] **Test backward pass matches PyTorch**

### 5.3 Training Loop

- [ ] **Implement basic training loop:**
  ```python
  for batch in dataloader:
      optimizer.zero_grad()
      pred = model(x)
      loss = cross_entropy(pred, y)
      loss.backward()
      optimizer.step()
  ```
- [ ] **Verify loss decreases over epochs**
- [ ] **Reach >90% accuracy on MNIST** (easy with MLP)

### 5.4 Examples

- [ ] **Create `examples/mnist_mlp.py`** - complete training script
- [ ] **Document expected output/accuracy**

---

## Phase 6: CUDA Backend Hardening

*After CPU works, make Triton robust*

### 6.1 Triton Syntax Fixes

- [ ] **Fix heuristics syntax error** - `triton.py:283`
  - Missing closing parenthesis in lambda
- [ ] **Add `@triton.jit` decorators** to reduce kernels if missing
  - `backend/kernels/reduce.py` lines 5 and 39

### 6.2 Kernel Testing

- [ ] **Run all CPU tests on CUDA** - compare outputs
- [ ] **Test kernel fusion works** - `(a+b)*c` should be 1 kernel
- [ ] **Add benchmark script** comparing CPU vs CUDA

### 6.3 Memory Management

- [ ] **Verify no memory leaks** in realize loop
- [ ] **Add `Tensor.cuda()` / `Tensor.cpu()`** convenience methods

---

## Phase 7: Convolution (Stretch Goal for CNN MNIST)

*Only after MLP works perfectly*

### 7.1 Conv2d via im2col

- [ ] **Implement `nn.functional.unfold`** (im2col)
  - Turns convolution into matmul
- [ ] **Implement `nn.Conv2d(in_channels, out_channels, kernel_size)`**
- [ ] **Test against PyTorch Conv2d output**

### 7.2 Pooling

- [ ] **Add `nn.MaxPool2d`** using strided view + max reduce
- [ ] **Add `nn.AvgPool2d`** using strided view + mean reduce

### 7.3 CNN MNIST

```python
# Target architecture
model = nn.Sequential(
    nn.Conv2d(1, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64*5*5, 10),
)
```

- [ ] **Train CNN on MNIST**
- [ ] **Reach >98% accuracy**

---

## Phase 8: Polish & Documentation

### 8.1 Error Messages

- [ ] **Improve assertion messages** with helpful hints
- [ ] **Add shape mismatch details** to broadcasting errors

### 8.2 Debug Tools

- [ ] **Improve `DEBUG=1` output** - cleaner kernel display
- [ ] **Add computation graph visualization** (graphviz)

### 8.3 Documentation

- [ ] **Update README** with working examples
- [ ] **Update DEVELOPMENT.md** roadmap
- [ ] **Add docstrings** to public API

### 8.4 CI/CD

- [ ] **Add GitHub Actions** for running tests
- [ ] **Add linting (ruff)** to CI
- [ ] **Add type checking (pyright)** to CI

---

## Summary Priority Order

| Priority | Phase | Estimated Effort | Dependency |
|----------|-------|------------------|------------|
| P0 | Phase 0: Foundation Fixes | 1-2 days | None |
| P1 | Phase 1: Core Tensor Ops | 2-3 days | Phase 0 |
| P1 | Phase 2: Shape Operations | 2-3 days | Phase 0 |
| P1 | Phase 3: Autograd | 3-4 days | Phase 0, 1 |
| P2 | Phase 4: NN Module | 2-3 days | Phase 3 |
| P2 | Phase 5: MNIST MLP | 2-3 days | Phase 4 |
| P3 | Phase 6: CUDA Hardening | 2-3 days | Phase 5 |
| P4 | Phase 7: Convolution | 3-5 days | Phase 6 |
| P4 | Phase 8: Polish | Ongoing | All |

---

## Known Limitations (Acceptable for v1.0)

These are intentionally deferred:

- **No negative strides** (`x[::-1]` not supported)
- **No advanced indexing** (boolean/integer array indexing)
- **No slice assignment** (`x[1:3] = y`)
- **GATHER kernel missing** - cross_entropy uses one-hot workaround
- **PAD and SHRINK ops** - enum exists but unimplemented
