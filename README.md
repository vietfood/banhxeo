# 🥞 banhxeo

> "Like a perfect Vietnamese crepe - crispy on the outside, efficient on the inside."

**banhxeo** is a tiny, lazy tensor compiler. It builds a computation graph, schedules it, and generates Triton kernels from scratch — so you can see how a deep learning framework actually works under the hood.

**It is ~3000 lines of code (with formatting).**

> [!WARNING]
> **DO NOT USE THIS FOR PRODUCTION.**
> This is an educational framework designed to demystify deep learning. It is fragile, aggressively minimalist, and lacks the safety checks found in PyTorch. Autograd is still being stabilized — treat backward passes as experimental.

## The Philosophy

This project follows the [tinygrad](https://github.com/tinygrad/tinygrad) school of thought:

1. **Build your own to understand.** You cannot really understand PyTorch or JAX until you have built the pieces yourself — lazy graphs, views, scheduling, codegen.
2. **Make it tiny.** Every abstraction must earn its place. If 10 lines can do the job of 100, use 10.
3. **Every line has meaning.** No legacy baggage, no compatibility shims, no "just in case" code. Read the source; nothing is hidden behind a macro or a code generator you did not write.

What that looks like in practice:

- **Lazy evaluation** — `x + y` does not compute anything. It builds a graph.
- **Views, not copies** — reshape, transpose, and slice change indexing, not memory.
- **Triton codegen** — the scheduler walks the graph and emits a custom GPU kernel.
- **Zero fluff** — no CPU fast paths, no multi-backend matrix. CPU is a slow interpreter for debugging.

## Quick Start

We use `uv` for dependency management.

```bash
# Clone
git clone https://github.com/lenguyen1807/banhxeo
cd banhxeo

# Install & sync
uv sync
source .venv/bin/activate
```

### Hello World (forward pass)

Autograd is not stable yet. Start with the forward path — this is the core of the compiler:

```python
from banhxeo import Tensor

# 1. Define tensors (lazy — no GPU memory allocated yet)
a = Tensor([1.0, 2.0, 3.0], device="cuda")
b = Tensor([4.0, 5.0, 6.0], device="cuda")

# 2. Build the graph
c = a + b

# 3. Realize — schedule, codegen, compile, launch
c.realize()
print(c.numpy())  # [5. 7. 9.]
```

Nothing runs until `.realize()`. That single call is the entire pipeline: topological sort → kernel boundaries → Triton source → JIT compile → launch.

### See the generated kernel

Set `DEBUG=1` (or higher) to inspect what the compiler produces:

```bash
DEBUG=1 python -c "
from banhxeo import Tensor
a = Tensor([1.0, 2.0, 3.0], device='cuda')
b = Tensor([4.0, 5.0, 6.0], device='cuda')
(a + b).realize()
"
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#debug-levels) for all debug levels.

## How It Actually Works

The compilation pipeline is documented in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — pipeline diagrams, file map, debug levels, and a tinygrad comparison for context.

At a glance:

```
Tensor.realize()
  → Backend.exec()        schedule, find kernel barriers, execute
  → TritonCodegen         walk LazyBuffer graph, emit Triton source
  → Kernel execution      JIT compile, cache, launch
```

## Development

Active development follows the self-guided course in **[docs/course/](docs/course/)**. The course rebuilds enough of a real tensor compiler — views, forward correctness, autograd, scheduling, IR, codegen — that you understand *why* production frameworks are shaped the way they are.

**Start here:** [docs/course/INDEX.md](docs/course/INDEX.md)

| Module | Topic | Status |
| --- | --- | --- |
| [Module 0](docs/course/MODULE_0.md) | Understand the current pipeline | In progress |
| [Module 1](docs/course/MODULE_1.md) | Views, strides, and indexing | In progress |
| [Module 2](docs/course/MODULE_2.md) | Forward correctness vs PyTorch | Upcoming |
| [Module 3](docs/course/MODULE_3.md) | Autograd | Upcoming |
| [Module 4](docs/course/MODULE_4.md) | Scheduling and fusion | Upcoming |
| [Module 5](docs/course/MODULE_5.md) | Basic kernel IR | Upcoming |
| [Module 6](docs/course/MODULE_6.md) | Specialized kernels (matmul, reduce) | Upcoming |
| [Module 7](docs/course/MODULE_7.md) | Memory planning | Upcoming |
| [Module 8](docs/course/MODULE_8.md) | Advanced IR and multi-backend lowering | Upcoming |
| [Module 9](docs/course/MODULE_9.md) | Compiler optimizations | Upcoming |

End-to-end projects (MNIST MLP, CNN) live in [docs/course/PROJECTS.md](docs/course/PROJECTS.md) — pick one after Modules 0–4 are stable.

## Running Tests

```bash
# Run all tests
uv run pytest tests

# Run a specific test file
uv run pytest tests/test_module_1_views.py
```

If you break it, you fix it.

## Inspiration

- [tinygrad](https://github.com/tinygrad/tinygrad) — the spiritual ancestor. Build tiny, understand deeply.
- [micrograd](https://github.com/karpathy/micrograd) — autograd from first principles.
- [Triton](https://openai.com/research/triton) — making GPU kernels writable by mortals.

## License

MIT.

*Built with curiosity and coffee in Hanoi.* ☕
