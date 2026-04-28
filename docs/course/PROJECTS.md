## Final Projects

### Project A: MNIST MLP ⭐⭐

**Goal:** Train a 2-layer MLP on MNIST to >90% accuracy.

**Requirements:**
- `nn.Linear` layer
- `SGD` optimizer
- Cross-entropy loss (already in `functional.py`, but uses numpy hack)

**Steps:**
1. Implement `nn.Linear` (Assignment in Module N1)
2. Implement `SGD` optimizer
3. Load MNIST data
4. Training loop
5. Verify convergence

---

### Project B: MNIST CNN ⭐⭐⭐

**Goal:** Train a CNN on MNIST to >98% accuracy.

**Requirements (builds on Project A):**
- `View.pad()` and `View.shrink()` working
- `nn.Conv2d` via im2col
- `nn.MaxPool2d`

**This is significantly harder** because Conv2d requires:
- PAD for same-padding
- Sliding window extraction (im2col)
- Batched matmul for the actual convolution
- SHRINK for output sizing

---

### Project C: Kernel Visualization ⭐⭐

**Goal:** Build a visualization tool for generated kernels.

Show:
- The LazyBuffer DAG
- Fusion boundaries
- Generated Triton code with annotations
- Memory access patterns

Extend `utils/viz.py` with richer visualization.

---
