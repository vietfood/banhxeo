## Module 0: Understanding the Current Codebase

Before implementing anything, you need to deeply understand what exists.

## Why This Module Exists

This module prevents fake progress.

banhxeo already has a lazy graph, a scheduler, Triton string codegen, specialized
kernels, and an autograd path. Some of it is messy. That is useful only if you
can trace the mess precisely.

Do not do broad fixes in this module. Your job is to answer:

```text
When does a symbolic tensor become real memory?
```

If you cannot answer that, every later refactor will be guessing.

Small fixes are allowed only when they unblock tracing. Everything else goes
into the bug ledger in Assignment 0.4.

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

### Assignment 0.4: Debug Killer Bug Ledger ⭐⭐⭐

**Task:** Run banhxeo until it breaks, then turn each breakage into a small,
classified bug report.

The goal is not to fix everything in Module 0. The goal is to rebuild your
mental model by forcing the code to explain where it is fragile.

Run at least these probes:

```python
# elementwise
(Tensor.rand((2, 3), device="cuda") + 1).realize()

# chained elementwise
((Tensor.rand((2, 3), device="cuda") + 1) * 2 - 3).realize()

# view + elementwise
(Tensor.rand((2, 3), device="cuda").T + 1).realize()

# slice/view
Tensor.rand((4, 5), device="cuda")[1:4, 2:5].realize()

# reduce
(Tensor.rand((4, 5), device="cuda").sum(axis=1) + 1).realize()

# matmul
(Tensor.rand((2, 3), device="cuda") @ Tensor.rand((3, 4), device="cuda")).realize()

# simple backward
x = Tensor.rand((2, 3), device="cuda", requires_grad=True)
(x * 2).sum().backward()
```

Use a small oracle for each probe.

Module 0 does not need the full correctness harness from Module 2, but it does
need enough expected behavior to tell whether a result is suspicious. For each
probe, record one of these oracle types:

```text
Oracle:
  - manual expected value
  - one-off PyTorch comparison
  - one-off NumPy comparison
  - traceback only
```

Examples:

```python
# manual oracle
out = (Tensor([1, 2, 3], device="cuda") + 1).realize().numpy()
expected = [2, 3, 4]

# one-off PyTorch oracle
data = torch.arange(6, device="cuda", dtype=torch.float32).reshape(2, 3)
expected = (data.T + 1).cpu().numpy()
actual = (Tensor(data, device="cuda").T + 1).realize().numpy()
```

Do not build a reusable `check_forward()` helper here. That belongs to Module 2.
In Module 0, the oracle is just enough evidence to classify the failure.

For each failure, write a bug entry:

```text
Title:
Minimal repro:
Oracle:
Expected:
Actual:
Traceback or wrong output:
First suspicious file/function:
Concept owner:
  - Module 1 views/indexing
  - Module 2 forward correctness
  - Module 3 autograd
  - Module 4 scheduling
  - Module 5 IR/codegen
  - Module 6 specialized kernels
Severity:
  - blocks tracing
  - wrong result
  - unsupported feature
  - confusing design
Next action:
  - fix now because it blocks Module 0
  - convert to failing test in later module
  - document as unsupported
```

Rules:

- Reduce every failure to the smallest repro you can.
- Do not fix more than one bug per patch.
- Do not mix a bug fix with cleanup.
- If a failure is really an unsupported feature, say so. Do not pretend it is a
  bug.
- If you cannot explain why the failure belongs to a module, keep tracing.

Deliverable:

Create or update a bug ledger such as `docs/course/BUG_LEDGER.md`.

**Why this assignment:** You forgot parts of the code because the code grew
faster than your understanding. Debugging is how you buy that understanding
back. The bug ledger turns vague fear into a queue of specific, owned problems.
