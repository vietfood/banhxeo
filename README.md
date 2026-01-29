# 🥞 banhxeo

> "Like a perfect Vietnamese crepe - crispy on the outside, efficient on the inside."

**banhxeo** is a tiny, lazy, autograd engine and tensor compiler. It implements backpropagation and Triton kernel generation from scratch. 

**It is ~2000 lines of code (with formatting).**

> [!WARNING]
> **DO NOT USE THIS FOR PRODUCTION.** 
> This is an educational framework designed to demystify deep learning. It is fragile, aggressively minimalist, and lacks 99% of the safety checks found in PyTorch. If you use this in production, you are on your own.

## The Philosophy

Modern deep learning frameworks are bloated. Reading PyTorch source code is like trying to understand a compiler by staring at assembly. **banhxeo** strips away the magic:

1.  **Lazy Evaluation**: `x + y` doesn't compute anything. It builds a graph.
2.  **Triton Codegen**: The graph is compiled into a single custom GPU kernel.
3.  **Zero Fluff**: No legacy support, no CPU optimizations (CPU is just a slow interpreter for debugging), no bloat.

## Quick Start

We use `uv` for dependency management. It's fast and better that `pip`.

```bash
# Clone
git clone https://github.com/lenguyen1807/banhxeo
cd banhxeo

# Install & Sync
uv sync
source .venv/bin/activate
```

### The "Hello World"

```python
from banhxeo import Tensor

# 1. Define tensors (Lazy - no memory allocated for data)
x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0, 0, -2.0]], requires_grad=True)

# 2. Build graph
z = y.matmul(x).sum()

# 3. Backprop (Implicitly realizes the forward pass first)
z.backward()

# 4. Check gradients
print(x.grad.numpy())
print(y.grad.numpy())
```

### See The Matrix

Want to see the Triton kernel we just generated? Set `DEBUG=1` (or larger).

```bash
DEBUG=1 python examples/mnist_mlp.py
```

Output:
```python
@triton.jit
def kernel(ptr0, ptr1, ptr2, ...):
    # Generated Triton Kernel ...
```

## How It Actually Works

The entire core logic fits in your head:

1.  **`Tensor`**: The frontend. Handles operator overloading and autograd state.
2.  **`LazyBuffer`**: The node in the computation graph. Tracks the operation (`ADD`, `MUL`) and its parents.
3.  **`View`**: Handles shapes and strides. **banhxeo** supports zero-copy reshapes, permutes, and slices.
4.  **`TritonCodegen`**: Walks the `LazyBuffer` graph, fuses compatible operations, and emits a Triton kernel string.
5.  **`Backend`**: Compiles the kernel and executes it on the GPU.

## Development

We are currently in the **v0.3 (Autograd)** phase.

- **[Roadmap & Status](docs/DEVELOPMENT.md)**: See what's working and what's broken.
- **[Stabilization Checklist](docs/CHECKLIST.md)**: The immediate plan to fix the "fragile" parts.

## Running Tests

If you break it, you fix it.

```bash
# Run all tests
uv run pytest tests

# Run specific test
uv run pytest tests/small_tests/mlp_forward.py
```

## Inspiration

- [tinygrad](https://github.com/tinygrad/tinygrad): The spiritual ancestor.
- [micrograd](https://github.com/karpathy/micrograd): For the autograd basics.
- [Triton](https://openai.com/research/triton): For making CUDA usable by mortals.

## License

MIT. 

*Built with curiosity and coffee in Hanoi.* ☕
