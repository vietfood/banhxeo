# Magnetron Study Guide

This note is a map for studying the `magnetron/` subtree inside this repository.

The short version:

- `banhxeo` is a lazy tensor compiler with Triton codegen.
- `magnetron` is a mostly eager runtime with a native C core, handwritten operator kernels, dynamic reverse-mode autodiff, and loadable hardware backends.

If you study it with that contrast in mind, the code becomes much easier to place.

---

## What Magnetron Is

Magnetron is built as a small ML runtime with four clear layers:

1. Python user API and training utilities.
2. Nanobind bindings that expose the native runtime.
3. A C core that owns tensors, views, operators, autodiff, snapshots, and backend dispatch.
4. Backend shared libraries such as CPU and CUDA.

The center of gravity is not a compiler IR. It is the runtime itself.

That is the first big mental shift from `banhxeo`.

---

## Architecture In One Diagram

```text
Python examples / nn / optim
        |
        v
python/magnetron/__init__.py
        |
        v
magnetron/bindings/_module.cpp + tensor bindings
        |
        v
include/magnetron/magnetron.h
        |
        v
core runtime
  - mag_context
  - mag_tensor
  - mag_op_stubs
  - mag_autodiff
  - mag_snapshot
  - mag_backend registry
        |
        v
active device submit(...)
        |
        +--> magnetron_cpu
        |
        +--> magnetron_cuda
```

The important point is that Python is thin, the core is explicit, and the backend boundary is a real ABI boundary.

---

## The Core Ideas

### 1. `mag_context_t` owns the runtime

Start with `mag_context_t` in `magnetron/magnetron/core/mag_context.h`.

The context owns:

- machine information
- telemetry counters
- slab allocators for tensor headers, storages, view metadata, and autodiff state
- the backend registry
- the active backend and active device

The implementation in `magnetron/magnetron/core/mag_context.c` makes the model explicit:

- one context is tied to one thread
- gradient recording is on by default
- backends are discovered dynamically at startup
- the requested device string like `cpu:0` is resolved through the registry

This is not a global hidden runtime. It is a concrete object that owns almost everything.

### 2. `mag_tensor_t` is a header plus storage

Study `magnetron/magnetron/core/mag_tensor.h` and `magnetron/magnetron/core/mag_tensor.c` next.

`mag_tensor_t` stores:

- context pointer
- shape and strides via `mag_coords_t`
- dtype
- storage buffer
- element count
- storage offset
- view metadata
- autodiff state
- a version counter

This is a very runtime-oriented tensor design. A tensor is not a symbolic node first. It is a concrete runtime object with storage, view state, and optional gradient metadata.

Two details matter:

- Views are implemented with `mag_as_strided(...)` and share storage.
- The tensor header is reference counted separately from storage.

That separation is a major part of how Magnetron stays explicit while still supporting slicing, reshaping, and aliasing.

### 3. Views are first-class and solved in the runtime

The view path lives mostly in:

- `magnetron/magnetron/core/mag_tensor.c`
- `magnetron/magnetron/core/mag_op_stubs.c`
- `magnetron/magnetron/core/mag_coords.h`

The most important functions are:

- `mag_as_strided(...)`
- `mag_view(...)`
- `mag_reshape(...)`
- `mag_solve_view_strides(...)`

The design is practical:

- if a view is obvious, reuse strides directly
- if the tensor is contiguous, synthesize row-major strides
- otherwise, call the generic stride solver
- if a reshape cannot be a view, materialize with `contiguous()` and rebuild shape

This is a very different learning target from `banhxeo`'s lazy `View` objects. Here the question is not “how do we preserve symbolic movement ops for later fusion?” but “can this runtime object legally alias the same storage?”

### 4. Operators are eager, validated in core, and dispatched to the active device

The heart of execution is `magnetron/magnetron/core/mag_op_stubs.c`.

The typical pattern is:

1. Validate dtype and shape semantics in core.
2. Allocate the result tensor or create a view.
3. Record autodiff metadata if gradient recording is enabled.
4. Build a `mag_command_t`.
5. Call `ctx->active_device->submit(...)`.

This means Magnetron is not centered on kernel fusion or deferred scheduling. The core decides semantics now, then asks the backend to run now.

Good functions to inspect:

- `mag_dispatch(...)`
- `mag_view(...)`
- `mag_reshape(...)`
- `mag_where(...)`
- `mag_matmul(...)`

`mag_dispatch(...)` is especially worth studying because it shows the entire policy boundary:

- core handles correctness and graph recording
- backend handles actual execution

### 5. Autodiff is dynamic reverse-mode built from recorded inputs

The relevant files are:

- `magnetron/magnetron/core/mag_autodiff.h`
- `magnetron/magnetron/core/mag_autodiff.c`
- `magnetron/magnetron/core/mag_gradients.c`
- `magnetron/magnetron/core/mag_toposort.c`
- `magnetron/magnetron/core/mag_operator.h`

The design is direct:

- each output tensor can hold a `mag_au_state_t`
- that state stores the opcode, input tensor references, op attributes, and accumulated gradient
- backward starts from a scalar root
- the graph is traversed with a topo sort over recorded parents
- gradient rules are handwritten per operator in `mag_gradients.c`

This is a classic dynamic autograd tape, closer to PyTorch-style eager autograd than to a lazy compiled graph.

The detail to pay attention to is where graph edges live:

- not in a separate graph object
- not in a global tape
- on each tensor's autodiff state

That keeps the system small and inspectable.

### 6. The operator table is the control plane

`magnetron/magnetron/core/mag_operator.h` is more important than it looks.

It defines:

- the opcode enum
- input/output arity
- supported dtype mask
- operator flags
- backward function pointer

This file is the compact specification of the runtime's operator surface.

If you want to understand “what is the framework, really?”, this is one of the best files to read early.

### 7. Backends are shared libraries behind a small ABI

Study:

- `magnetron/magnetron/core/mag_backend.h`
- `magnetron/magnetron/core/mag_backend.c`
- `magnetron/magnetron/CMakeLists.txt`

The runtime loads backend modules dynamically from shared libraries such as:

- `magnetron_cpu`
- `magnetron_cuda`

The core checks:

- ABI cookie
- runtime version compatibility
- required vtable functions

This is a strong architectural choice. The backend interface is not an internal convenience layer. It is a real plugin boundary.

That makes Magnetron feel more like a small runtime platform than a single monolithic library.

### 8. The CPU backend is already the serious backend

The CPU backend is the mature backend right now.

Start with:

- `magnetron/magnetron/cpu/CMakeLists.txt`
- `magnetron/magnetron/cpu/mag_cpu.c`

Then notice the rest of the directory:

- architecture-specific configuration under `amd64/` and `arm64/`
- operator implementations in `mag_cpu_impl_*.inl`
- threadpool and phase-fence infrastructure
- specialization and autotuning helpers

Important design points:

- the backend picks optimized CPU kernels based on detected hardware
- work can run single-threaded or through the CPU threadpool
- storage allocation is device-owned even on CPU

This is where Magnetron differs sharply from educational runtimes that treat CPU as a toy fallback. Here CPU is the primary systems target.

### 9. CUDA exists, but the architecture still reads as CPU-first

Look at:

- `magnetron/magnetron/cuda/CMakeLists.txt`
- `magnetron/magnetron/cuda/mag_cuda.cu`

The CUDA layer is real, but in this repo snapshot it is not yet the dominant execution story. Even the README describes CUDA as still being completed and stabilized.

So if your goal is to learn the architecture, read CPU first. The CPU backend explains the runtime more clearly.

### 10. Snapshots are part of the runtime, not an afterthought

Study `magnetron/magnetron/core/mag_snapshot.c`.

This file defines the native `.mag` format with sections for:

- file header
- string pool
- metadata map
- tensor descriptors
- tensor buffers

The important idea is not just serialization. It is zero-copy or memory-mapped loading as a first-class systems feature.

That is a major clue about Magnetron's ambition: it wants to be useful for real inference workloads, not only for toy training loops.

### 11. Python `nn` and `optim` are intentionally thin

Read:

- `magnetron/python/magnetron/__init__.py`
- `magnetron/python/magnetron/nn/module.py`
- `magnetron/python/magnetron/nn/layers.py`
- `magnetron/python/magnetron/nn/loss.py`
- `magnetron/python/magnetron/optim.py`

The pattern is simple:

- `Tensor` comes from the native bindings
- `Module`, `Parameter`, layers, losses, and optimizers are written in Python
- these higher-level pieces mostly compose tensor ops rather than hiding them

This is useful to study because it shows a clean split:

- native code for the runtime-critical path
- Python for ergonomics and model composition

---

## The Best Reading Order

If you try to read Magnetron front to back, you will drown in details. Use this order instead.

### Pass 1: Understand the product

Read these first:

- `magnetron/README.md`
- `magnetron/docs/Magnetron-Cheatsheet.md`
- `magnetron/examples/xor/main.py`
- `magnetron/examples/qwen3/README.md`

Goal:

- understand what the framework promises to users
- see both the tiny training case and the real inference case

### Pass 2: Find the Python/native boundary

Read:

- `magnetron/python/magnetron/__init__.py`
- `magnetron/magnetron/bindings/_module.cpp`
- `magnetron/magnetron/bindings/context.cpp`
- `magnetron/magnetron/bindings/tensor_class.cpp`
- `magnetron/magnetron/bindings/tensor_operators.cpp`

Goal:

- see that Python is mostly a thin skin over the native runtime
- identify what is implemented in bindings versus what is implemented in core

### Pass 3: Learn the runtime data model

Read:

- `magnetron/include/magnetron/magnetron.h`
- `magnetron/magnetron/core/mag_context.h`
- `magnetron/magnetron/core/mag_context.c`
- `magnetron/magnetron/core/mag_tensor.h`
- `magnetron/magnetron/core/mag_tensor.c`
- `magnetron/magnetron/core/mag_coords.h`

Goal:

- understand ownership, context lifetime, storage, shape, strides, views, and tensor metadata

### Pass 4: Learn how ops really happen

Read:

- `magnetron/magnetron/core/mag_operator.h`
- `magnetron/magnetron/core/mag_op_stubs.c`
- `magnetron/magnetron/core/mag_reduce_plan.h`

Goal:

- see where argument validation, broadcasting, output allocation, and dispatch happen

This is the single most important pass if you want to borrow ideas for `banhxeo`.

### Pass 5: Learn backward

Read:

- `magnetron/magnetron/core/mag_autodiff.h`
- `magnetron/magnetron/core/mag_autodiff.c`
- `magnetron/magnetron/core/mag_toposort.c`
- `magnetron/magnetron/core/mag_gradients.c`

Goal:

- understand how Magnetron records parents during forward
- understand how it reconstructs execution order for reverse-mode autodiff

### Pass 6: Learn the backend boundary

Read:

- `magnetron/magnetron/core/mag_backend.h`
- `magnetron/magnetron/core/mag_backend.c`
- `magnetron/magnetron/cpu/CMakeLists.txt`
- `magnetron/magnetron/cpu/mag_cpu.c`
- `magnetron/magnetron/cuda/CMakeLists.txt`

Goal:

- understand how the runtime treats hardware support as pluggable modules
- see why CPU is the best backend to learn first

### Pass 7: Learn persistence and real workloads

Read:

- `magnetron/magnetron/core/mag_snapshot.c`
- `magnetron/examples/ae/main.py`
- `magnetron/examples/gpt2/main.py`
- `magnetron/examples/qwen3/main.py`

Goal:

- connect the runtime design to actual training and inference workflows

### Pass 8: Turn on runtime visibility

Read:

- `magnetron/docs/Environment Variables.md`

Then use:

```bash
MAG_LOG_LEVEL=info python magnetron/examples/xor/main.py
```

Goal:

- make backend loading, machine detection, and runtime startup more visible while you trace the code

---

## What To Compare Against `banhxeo`

This is the part that matters if you are using Magnetron as inspiration.

### Lazy compiler vs eager runtime

`banhxeo` is centered on lazy graph construction and later code generation.

Magnetron is centered on immediate semantic execution through backend dispatch.

Ask yourself:

- do you want the framework core to be primarily a compiler pipeline or a runtime pipeline?
- where should correctness live: in scheduling/codegen or in eager operator stubs?

### Symbolic movement vs runtime view legality

`banhxeo` cares about preserving movement ops for later lowering.

Magnetron cares about whether a tensor can legally share storage right now.

That changes how you think about:

- view representation
- aliasing
- reshape fallback behavior
- error surfaces

### Fused kernels vs handwritten backend kernels

`banhxeo` tries to fuse work into generated Triton kernels.

Magnetron uses a fixed operator surface and lets backends implement those ops directly.

This means Magnetron pays more attention to:

- backend ABI stability
- operator metadata tables
- device-specific kernel specialization
- runtime storage ownership

### Autograd storage location

In Magnetron, autodiff state is attached directly to tensors.

That is small, explicit, and easy to inspect. It is worth considering whether some parts of `banhxeo` would get simpler if more execution metadata lived on runtime objects instead of in external scheduling structures.

### Serialization as a systems feature

The `.mag` snapshot format is integrated into the runtime story.

That is a reminder that a framework is not only:

- tensors
- ops
- gradients

It is also:

- model loading
- runtime startup cost
- memory mapping
- portability of weights

---

## Concrete Files Worth Studying Closely

If you only read ten files, read these:

1. `magnetron/include/magnetron/magnetron.h`
2. `magnetron/magnetron/core/mag_context.c`
3. `magnetron/magnetron/core/mag_tensor.c`
4. `magnetron/magnetron/core/mag_operator.h`
5. `magnetron/magnetron/core/mag_op_stubs.c`
6. `magnetron/magnetron/core/mag_autodiff.c`
7. `magnetron/magnetron/core/mag_gradients.c`
8. `magnetron/magnetron/core/mag_backend.c`
9. `magnetron/magnetron/cpu/mag_cpu.c`
10. `magnetron/magnetron/core/mag_snapshot.c`

That set gives you the public surface, runtime ownership model, operator path, backward path, backend boundary, and persistence story.

---

## Questions To Ask While Reading

Use these as prompts while you study:

1. Where does Magnetron choose runtime simplicity over peak flexibility?
2. Which parts of the framework are explicitly runtime concerns rather than compiler concerns?
3. Why does attaching autodiff state to tensors keep the implementation small?
4. What would break if the backend boundary were not a shared-library ABI?
5. Which ideas are worth copying into `banhxeo`, and which only make sense for an eager runtime?

---

## Practical Study Plan

If you want a concrete sequence for a few evenings of study:

### Session 1

- Read the README and cheatsheet.
- Run through the XOR example.
- Read the Python bindings for `Tensor`.

### Session 2

- Read `mag_context` and `mag_tensor`.
- Trace how a tensor gets storage and how a view keeps aliasing metadata.

### Session 3

- Read `mag_op_stubs.c` carefully.
- Trace `view`, `reshape`, `where`, and `matmul` end to end.

### Session 4

- Read `mag_autodiff.c`, `mag_toposort.c`, and `mag_gradients.c`.
- Manually trace one backward pass for a small expression.

### Session 5

- Read `mag_backend.c` and `cpu/mag_cpu.c`.
- Understand how commands cross the backend boundary.

### Session 6

- Read `mag_snapshot.c`.
- Then inspect `examples/qwen3` to see how the runtime supports a more realistic inference flow.

---

## Final Takeaway

The best way to think about Magnetron is this:

It is not “tinygrad but in C”.

It is a compact systems-oriented ML runtime where the main abstraction is an explicit tensor runtime with:

- concrete storage ownership
- explicit views and stride solving
- operator dispatch through a backend ABI
- dynamic reverse-mode autodiff
- built-in model snapshot support

If `banhxeo` teaches you how a lazy tensor compiler can work, Magnetron teaches you how a small but serious eager runtime can be built.