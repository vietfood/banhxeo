## Module 3: Autograd

Build and verify the backward pass.

## Why This Module Comes After Forward Correctness

Autograd is a second graph system layered on top of the forward tensor system.

The forward pass builds `LazyBuffer` graphs. The backward pass builds more
`LazyBuffer` graphs from derivative rules. If the forward system is unstable,
the backward system will amplify the instability.

The goal here is not a full PyTorch autograd clone. The goal is to understand
reverse-mode autodiff in a tensor framework:

```text
forward Function.apply()
  -> save context
  -> build output Tensor
  -> backward topological traversal
  -> local gradient rules
  -> gradient accumulation
```

## Assignment 3.1: Trace The Current Backward Pass ⭐⭐

Trace:

```python
x = Tensor.rand((2, 3), device="cuda", requires_grad=True)
y = ((x * 2).sum())
y.backward()
```

Questions:

1. Which `Function` objects are created?
2. What does each context save?
3. What is the topological order in `backward()`?
4. Which gradient `LazyBuffer` objects are created?
5. When are gradient tensors realized?

**Why this assignment:** If you cannot trace one scalar loss backward, gradient
test failures will look like magic.

## Assignment 3.2: Implement Gradient Comparison Tests ⭐⭐

For every differentiable op, verify gradients match PyTorch.

```python
def check_gradient(banhxeo_fn, pytorch_fn, *input_shapes, rtol=1e-4, atol=1e-4):
    """
    Verify banhxeo gradients match PyTorch gradients.
    """
```

Test:

- add, sub, mul, div
- exp, log, sin, sqrt, neg
- sum, max, mean
- matmul
- reshape, permute, expand

**Why this assignment:** Local derivative formulas can look correct while their
composition is wrong. Gradient comparison tests check the composed behavior.

## Assignment 3.3: Movement Op Gradients ⭐⭐

For the movement ops from Module 1:

| Forward Op | Backward Op |
|------------|-------------|
| `pad` | `shrink` |
| `shrink` | `pad` |
| `flip` | `flip` |

**Why this assignment:** Movement gradients teach inverse views. The backward
rule for a shape op is often another shape op, not a numeric kernel.

## Assignment 3.4: Broadcasting Gradient Accumulation ⭐⭐⭐

When a tensor is broadcast in the forward pass, its gradient must be summed back
to the original shape.

```python
a = Tensor([1, 2, 3], requires_grad=True)
b = Tensor.rand((4, 3), requires_grad=True)
c = a + b
c.sum().backward()
```

`a.grad` should have shape `(3,)`, with gradients accumulated over axis `0`.

Questions:

1. Which axes were introduced by broadcasting?
2. Which axes had size `1` and expanded to a larger size?
3. Does `Expand.backward` reduce over both cases?

**Why this assignment:** Broadcasting is where many toy autograd engines lie to
you. If this is wrong, training may run while silently producing bad gradients.

## Checkpoint

At the end of this module, banhxeo should have:

- a traced explanation of backward execution
- gradient comparison tests for existing differentiable ops
- explicit expected failures for unsupported gradient cases
- no optimization changes mixed into autograd fixes
