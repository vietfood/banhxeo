## Final Projects

Final projects should prove that the compiler path works end to end. Choose one
only after Modules 0, 1, 2, 3, and 4 are stable. Module 5 is strongly recommended
if the project depends on generated Triton kernels being understandable.

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

**Why this project:** MNIST MLP is the smallest proof that forward ops, autograd,
optimizers, and data movement are coherent enough to train.

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

**Why this project:** CNNs force the view system to grow up. If pad, shrink,
reshape, and batched matmul are shaky, this project will expose it fast.

---

### Project C: Kernel Visualization ⭐⭐

**Goal:** Build a visualization tool for generated kernels.

Show:
- The LazyBuffer DAG
- Fusion boundaries
- Generated Triton code with annotations
- Memory access patterns

Extend `utils/viz.py` with richer visualization.

**Why this project:** Visualization is a compiler-learning project. If you can
show the graph, barriers, generated code, and memory access pattern, you probably
understand the pipeline.

---
