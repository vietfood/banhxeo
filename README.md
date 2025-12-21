# 🥞 banhxeo: A Simple, Efficient (Enough), and Educational Tensor Framework

> [!WARNING] 
> Banhxeo cannot be used (and will never be used) for production.

> [!NOTE]
> "Like a perfect Vietnamese crepe - crispy on the outside, efficient on the inside"

Banhxeo is a minimalist deep learning framework built from scratch to understand how modern ML frameworks actually work. Inspired by [Tinygrad](https://github.com/tinygrad/tinygrad) and the philosophy that **the best way to learn is to build**, this project strips away all the magic and shows you the raw mechanics of lazy evaluation, automatic differentiation, and GPU kernel generation.

**Current Status:** A week of commited - Basic tensor operations working with Triton codegen 🚀

## Why Banhxeo?

Because reading PyTorch source code is like trying to understand a compiler by staring at assembly. Banhxeo is:

- **Tiny** (~1000 LOC): Small enough to understand in an afternoon
- **Educational**: Every line exists to teach, not to handle edge cases from 2015
- **Lazy**: Builds a computation graph, compiles to Triton kernels on-the-fly
- **Transparent**: No hidden optimizations, no magic, just pure computation

## Quick Example

```python
from banhxeo import Tensor

# Create tensors (lazy - nothing computed yet)
x = Tensor([1.0, 2.0, 3.0, 4.0])
y = Tensor([2.0, 3.0, 4.0, 5.0])

# Build computation graph
z = (x + y) * x.sin()  # Still lazy!

# Execute: generates Triton kernel and runs on GPU
result = z.realize()
print(result)  # tensor([...])
```

Set `DEBUG=1` to see the generated Triton kernel:
```bash
DEBUG=1 python your_script.py
```

## Architecture in 60 Seconds

```
Tensor → LazyBuffer → Computation Graph → Triton Codegen → GPU Kernel
```

1. **Tensor**: User-facing API, operator overloading
2. **LazyBuffer**: Lazy computation graph node (op, src, view)
3. **View**: Stride-based memory layout (enables zero-copy reshape/slice)
4. **Codegen**: Walks the graph, emits Triton code
5. **realize()**: Compiles kernel, allocates output, executes

### The Core Trick: Lazy Evaluation

Instead of computing immediately:
```python
x + y  # PyTorch: allocates memory, runs kernel
```

Banhxeo builds a graph:
```python
LazyBuffer(op=ADD, src=[LazyBuffer(...), LazyBuffer(...)])
```

Only when you call `.realize()` does it:
- Topologically sort the graph
- Generate a fused Triton kernel
- Execute on GPU

**Why?** Kernel fusion. `(x + y) * z` becomes ONE kernel, not three.

## Current Features

- [x] Basic ops: `+, -, *, log, exp, sin, sqrt`
- [x] Lazy evaluation with computation graphs
- [x] View operations: `permute, slice, expand` (zero-copy!)
- [x] Broadcasting (naive but works)
- [x] Triton codegen for GPU execution

## Roadmap

### v0.2 - More Ops
- [ ] Reshape
- [ ] Matmul (the big one)
- [ ] Reduce ops: `sum, max, min, mean`
- [ ] Comparison ops: `<, >, ==`
- [ ] Where/masking operations

### v0.3 - Optimization
- [ ] Kernel fusion optimization passes
- [ ] Memory planning and reuse
- [ ] Better broadcasting rules
- [ ] Constant folding

### v0.4 - Autograd Engine
- [ ] Backward pass implementation
- [ ] Gradient accumulation
- [ ] `Tensor.backward()` API
- [ ] Simple MLP training example

### v0.5 - Real Neural Networks
- [ ] Conv2d (im2col approach)
- [ ] Pooling operations
- [ ] Batch normalization

### v1.0 - The Dream
- [ ] Train a CNN on MNIST
- [ ] Benchmark against PyTorch
- [ ] Full documentation with tutorials

## Installation

From source:

```bash
# Clone the repo
git clone https://github.com/lenguyen1807/banhxeo
cd banhxeo

# Install dependencies
uv sync && source .venv/bin/activate

# Run tests (when they exist)
python -m pytest tests/
```

From Pip:

```bash
pip install banhxeo
```

**Requirements:**
- Python 3.10+
- CUDA-capable GPU (for Triton)
- PyTorch (just for tensors/CUDA, not autograd)
- Triton

## Learn By Doing

Want to understand something? Break it:

```python
# What happens if you slice incorrectly?
x = Tensor([1, 2, 3])
x.slice(((0, 10),))  # IndexError - read the traceback

# How does broadcasting work?
x = Tensor([[1, 2, 3]])  # shape (1, 3)
y = Tensor([[1], [2]])   # shape (2, 1)
z = x + y                # shape (2, 3) - how?
```

Set `DEBUG=1` and watch the generated kernels. Modify `codegen.py` and see what breaks.

## Why "Banhxeo"?

It's a Vietnamese crispy pancake. Like this framework:
- Thin and crispy (minimal LOC)
- Made fresh to order (compilation)
- Surprisingly satisfying (when it works)

Plus, every ML framework needs a food name. It's the law.

## Resources to Pair With This Code

- [Tinygrad](https://github.com/tinygrad/tinygrad) - The OG minimalist Tensor framework
- [Micrograd](https://github.com/karpathy/micrograd) - Karpathy's autograd in 100 lines
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/) - GPU kernel language
- A good AI partner (Opus 4.5 or Gemini 3.0 Pro maybe)

## License

MIT - Do whatever you want. If you learn something, that's payment enough.

---

*Built with curiosity and coffee in Hanoi* ☕

**Remember:** The goal isn't to build production software. It's to understand the magic. Read every line. Break things. Fix them. That's how you learn.