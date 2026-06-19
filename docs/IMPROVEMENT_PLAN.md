# banhxeo Improvement Plan

This plan is meant to help you learn by building. The goal is not for an
assistant to rewrite banhxeo into tinygrad, Magnetron, or PyTorch. The goal is
to make each next step force one real framework concept into your hands.

## Assumptions

- banhxeo should stay a lazy Python framework with Triton codegen for now.
- The current "Triton string interpreter" is valuable: keep it until it becomes
  the thing that blocks learning.
- Magnetron is worth studying, but mainly for runtime discipline. It should not
  pull banhxeo into a C++ rewrite yet.
- Every implementation step should add tests or a trace note. If the behavior is
  not tested or documented, you probably did not really learn it.

## North Star

Build toward this pipeline:

```text
Tensor API
  -> LazyBuffer graph
  -> view/indexing semantics
  -> forward correctness harness
  -> autograd correctness harness
  -> schedule/fusion groups
  -> simple elementwise kernel IR
  -> Triton renderer
  -> specialized kernels
  -> memory planning
  -> advanced IR
  -> optimization passes
  -> backend execution
```

Do not jump straight to the full tinygrad UOp universe. Start with a tiny IR that
only represents one fused elementwise kernel. Make it boring, inspectable, and
easy to delete if you learn a better shape later.

## Phase 0: Make The Current System Observable

Before changing architecture, make yourself explain the one you already built.

Deliverables:

- Finish `docs/SOLUTION.md` for COURSE Module 0.
- Add one small trace script or notebook that runs:
  - `a + b`
  - `x.T + 1`
  - `x.sum(axis=1) + 1`
- For each case, record:
  - LazyBuffer DAG
  - schedule order
  - barriers
  - generated Triton source

Study questions:

- Which nodes are user-visible tensors?
- Which nodes are just movement metadata?
- Where does a lazy graph become real memory?

Rule: no architecture changes in this phase.

## Phase 1: Fix Views And Indexing First

COURSE Module 1 is the right first implementation target. Almost every serious
tensor framework lives or dies by shape, stride, offset, masks, and view
composition.

Build in this order:

1. `View.to_index_expr(linear_idx, var_prefix)`
2. replace duplicated indexing in `backend/triton.py`
3. tests for contiguous, transpose, broadcast, slice, and negative stride
4. `View.shrink()`
5. `View.flip()`
6. only then consider `View.pad()`

Why this order:

- `to_index_expr` removes real duplication immediately.
- `shrink` and `flip` deepen the same model without introducing masks.
- `pad` forces a design choice: view mask vs lowering to `where`.

Recommended choice for padding:

- For learning tinygrad-style lowering, make `pad` become a masked load or
  `where` during lowering.
- For minimal code today, storing a mask on `View` is okay, but write down that
  it couples view semantics to codegen.

Checkpoint:

```python
x = Tensor([[1, 2], [3, 4]], device="cuda")
assert x.T.numpy().tolist() == [[1, 3], [2, 4]]
assert x[1:].numpy().tolist() == [[3, 4]]
assert x.flip(1).numpy().tolist() == [[2, 1], [4, 3]]
```

## Phase 2: Build A Forward Correctness Harness

Do this before autograd and IR. A compiler that preserves wrong behavior is
still wrong.

Deliverables:

- forward tests against PyTorch for elementwise, movement, reductions, and matmul
- one helper that can run the same comparison on CPU and CUDA
- small asymmetric shapes that expose broadcasting and stride bugs

Priority bugs to investigate:

- `Tensor.__getitem__` appears to duplicate view logic that belongs in `View`
- `visit_UnaryOp` references `op` instead of `buf.op`
- multi-axis reduce is not implemented

Do not fix all of these in one patch. Turn each into a failing test first.

## Phase 3: Build And Verify Autograd

Do this after forward behavior has a reference. Otherwise gradient failures will
mix backward bugs with ordinary tensor bugs.

Deliverables:

- trace notes for one scalar loss backward pass
- gradient comparison tests against PyTorch
- movement-op gradient tests for reshape, permute, expand, shrink, flip, and pad
- explicit expected failures for unsupported gradient cases

Priority bugs to investigate:

- broadcast gradient needs careful verification
- gradient accumulation needs shape checks
- movement gradients should be inverse view operations where possible

## Phase 4: Make Scheduling Explicit

The current barrier rules are hidden inside `CUDABackend`. Pull them into a small
module only after you can explain the current recursive execution.

Deliverables:

- `src/banhxeo/backend/scheduling.py`
- documented `is_barrier(buf)`
- `analyze_fusion(output)` for debug visibility
- debug output at `DEBUG >= 2`

Important lesson:

Scheduling is not just topological sort. It is the policy that decides which
values stay in registers and which values become real buffers.

Keep the first rule set simple:

- realized buffers are boundaries
- random/data-producing loads are boundaries
- reduce is a boundary
- matmul is a boundary
- view/contiguous of an unrealized compute node is a boundary until you have a
  better lowering model

Checkpoint examples:

- `(a + b) * 2 - 1` should be one elementwise kernel.
- `x.sum(axis=1) + 1` should be two kernels.
- `(a @ b) * 2` should be two kernels.

## Phase 5: Add A Tiny Kernel IR

Do this after Phase 1, not before. Otherwise the IR will encode indexing bugs.
Also do it after scheduling, because the first IR should represent one fused
elementwise kernel body, not the whole LazyBuffer graph.

Start deliberately small:

- list-based IR, not DAG-based IR
- elementwise kernels only
- no reductions
- no matmul
- no full tinygrad-style pattern matcher

Minimum useful IR ops:

- `LOAD`
- `STORE`
- `CONST`
- `ADD`, `SUB`, `MUL`, `DIV`, `MAX`, `CMPLT`
- `NEG`, `EXP`, `LOG`, `SIN`, `SQRT`
- `WHERE`
- `CAST`

Deliverables:

- `src/banhxeo/backend/ir.py`
- `src/banhxeo/backend/renderer.py`
- tests that render `(a + b) * 2`
- a flag or small experimental path that compares old codegen and IR codegen

Do not delete the current `TritonCodegen` yet. Run both paths side by side until
the IR path is less confusing than the old path.

Good first optimization passes:

- constant folding
- dead code elimination

Avoid for now:

- common subexpression elimination
- algebraic simplification
- backend-independent indexing IR
- UOp deduplication

Those are good ideas, but not yet.

## Phase 6: Specialized Kernels

Only after correctness tests exist:

1. annotate current matmul and reduce kernels
2. implement multi-axis reduce
3. implement batched matmul
4. consider tree reduction

This phase teaches GPU programming rather than framework architecture. Keep that
distinction clear so the project does not sprawl.

## Phase 7: Memory Planning

Study memory reuse only after scheduling and correctness are stable.

Deliverables:

- lifetime notes for a few scheduled graphs
- a tiny buffer-pool experiment
- a clear rule for when a realized temporary can be released or reused

The point is not to beat PyTorch's allocator. The point is to understand that a
compiler is also responsible for resource planning.

## Phase 8: Advanced IR And Multi-Backend Lowering

Do this only after the basic IR path is less confusing than the old string
codegen path.

Deliverables:

- design note separating graph IR from kernel IR
- typed IR values
- target capability object
- one toy second renderer
- MLIR reading checkpoint

This is where production compiler ideas become relevant. MLIR is worth studying
here, but adopting it before this phase would likely bury the learning under
infrastructure.

## Phase 9: Compiler Optimizations

Do this after the IR has enough structure to preserve semantics.

Deliverables:

- constant folding
- dead code elimination
- common subexpression elimination
- algebraic simplification
- layout-aware rewrites
- tiny cost model for logging

Optimization is program rewriting under constraints. If you cannot state the
constraint, do not implement the rewrite.

## What To Learn From Magnetron

Magnetron is not a better version of banhxeo. It is a different answer to the
question "what is a framework?"

Ideas worth borrowing now:

- explicit runtime context instead of hidden globals
- tensor header vs storage separation
- first-class shape/stride/storage-offset model
- operator metadata table as the compact spec of supported ops
- thin Python layer over a small core
- backend boundary with clear ownership rules

Ideas to postpone:

- shared-library backend ABI
- native snapshot format
- CPU threadpool and architecture-specific kernels
- full C or C++ rewrite

Use Magnetron as a reading exercise like this:

1. Read `mag_tensor.c` and compare it to `LazyBuffer` plus `View`.
2. Read `mag_op_stubs.c` and compare eager dispatch to banhxeo scheduling.
3. Read `mag_operator.h` and design a tiny operator table for banhxeo.
4. Read `mag_autodiff.c` and compare tensor-attached autodiff state with
   banhxeo's `Function` contexts.

## Should banhxeo Become C++?

Not yet.

A C++ rewrite would teach useful systems lessons, but right now it would also
hide the compiler lessons under build systems, bindings, memory ownership, ABI
design, and debugging friction. The better path is:

1. keep banhxeo Python-first until the compiler architecture is clear
2. write one tiny native experiment later, not a rewrite
3. choose the native boundary after you know what needs to be fast

A good later C++ experiment:

```text
cpp_backend/
  TensorStorage
  Shape/Stride helpers
  CPU eager elementwise kernels
  Python bindings
```

This would teach Magnetron-style runtime design without throwing away the lazy
Triton compiler path.

## How To Work With An Assistant

Use the assistant as a reviewer and debugger, not as the main author.

Good prompts:

- "Ask me questions about this generated kernel until I can explain it."
- "Review my `View.to_index_expr` implementation for edge cases."
- "Write tests for this behavior, but do not implement the fix."
- "I tried implementing `shrink`; point out the smallest next correction."
- "Compare my scheduler rules to tinygrad and tell me what I am missing."

Avoid prompts like:

- "Implement Module 5."
- "Rewrite the backend."
- "Make it production-grade."

Those will skip the part where your brain earns the framework.

## Suggested 6-Week Path

Week 1:

- finish Module 0 traces
- write missing notes in `docs/SOLUTION.md`
- add minimal tests directory

Week 2:

- implement `View.to_index_expr`
- remove duplicated indexing code
- test views hard

Week 3:

- implement `shrink` and `flip`
- decide pad design in a short note
- fix any view bugs found by tests

Week 4:

- add PyTorch comparison tests for forward behavior
- trace one backward pass by hand
- add the first gradient comparison tests

Week 5:

- fix exactly one correctness or autograd bug
- formalize scheduling rules
- add fusion logging

Week 6:

- create tiny list-based IR
- render one elementwise kernel through the IR path
- keep old codegen alive

The pain is signal. Follow it, but keep the patches small.
