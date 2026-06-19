## Module 0: Understanding the Current Codebase

Before implementing anything, you need to deeply understand what exists.

## Why This Module Exists

This module prevents fake progress.

banhxeo already has a lazy graph, a scheduler, Triton string codegen, specialized
kernels, and an autograd path. Some of it is messy. That is useful only if you
can trace the mess precisely.

Do not fix anything in this module. Your job is to answer:

```text
When does a symbolic tensor become real memory?
```

If you cannot answer that, every later refactor will be guessing.

### Assignment 0.1: Trace a Simple Operation ⭐

**Task:** Trace what happens when you run:
```python
a = Tensor([1, 2, 3], device="cuda")
b = Tensor([4, 5, 6], device="cuda")
c = (a + b).realize()
```

**Questions to answer:**
1. What `LazyBuffer` objects are created?
2. When does `TritonCodegen.generate()` get called?
3. What Triton kernel source is generated?
4. How does the kernel get compiled and executed?

**Hint:** Add `DEBUG=3` environment variable and trace through:
- `tensor.py` → `__add__` → `Add.apply()` 
- `function.py:137-144` → `Add.forward()`
- `buffer.py:184-193` → `compute_ops()`
- `tensor.py:595` → `realize()`
- `backend/__init__.py:299-319` → `exec()`

**Deliverable:** Write a document explaining the full flow with line numbers.

**Why this assignment:** Elementwise add is the smallest example that still
passes through the whole compiler path: Tensor API, Function, LazyBuffer,
schedule, Triton source, compile, launch.

---

### Assignment 0.2: Trace a View Operation ⭐⭐

**Task:** Trace what happens with:
```python
x = Tensor.rand(shape=(2, 3), device="cuda")
y = x.T  # transpose
z = (y + 1).realize()
```

**Questions to answer:**
1. Does `x.T` create a new buffer or just a new view?
2. How does `TritonCodegen` handle the transposed strides?
3. What indexing code is generated in the kernel?

**Key files to trace:**
- `tensor.py:311-313` → `_transpose()`
- `buffer.py:303-304` → `permute()`
- `view.py:81-97` → `View.permute()`
- `triton.py:59-95` → `render_indexing()`

**Insight to discover:** Movement ops don't move data - they change how we *index* into data.

**Why this assignment:** A transpose is the cheapest test of whether you
understand tensor frameworks. If you think it copies data by default, you have
not internalized views yet.

---

### Assignment 0.3: Understand Barriers ⭐⭐

**Task:** Explain why this needs 2 kernels:
```python
x = Tensor.rand(10, 10, device="cuda")
y = x.sum()      # Kernel 1: reduction
z = y + 1        # Kernel 2: elementwise
z.realize()
```

But this needs only 1 kernel:
```python
x = Tensor.rand(10, 10, device="cuda")
y = x + 1        # Fused into 1 kernel
z = y * 2        # 
z.realize()
```

**Key code:** `backend/__init__.py:105-121` (`is_barrier`) and `123-148` (`get_barriers`)

**Question:** Why can't reduction be fused with the following elementwise op?

**Hint:** Think about parallelism. Elementwise ops are embarrassingly parallel (each output independent). Reductions require coordination across threads.

**Why this assignment:** Scheduling is not just topological sort. This example
forces you to see the first real kernel boundary.

---
