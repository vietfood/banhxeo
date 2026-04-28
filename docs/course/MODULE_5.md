## Module 5: Autograd Correctness

Ensure gradients are computed correctly.

### Assignment 5.1: Implement Gradient Tests ⭐⭐

**Task:** For every differentiable op, verify gradients match PyTorch.

```python
# tests/test_autograd.py

def check_gradient(banhxeo_fn, pytorch_fn, *input_shapes, rtol=1e-4, atol=1e-4):
    """
    Verify banhxeo gradient matches PyTorch gradient.
    """
    import torch
    
    # Create random inputs
    inputs_np = [np.random.randn(*shape).astype(np.float32) for shape in input_shapes]
    
    # banhxeo forward + backward
    inputs_bx = [Tensor(x, device="cuda", requires_grad=True) for x in inputs_np]
    output_bx = banhxeo_fn(*inputs_bx)
    output_bx.sum().backward()
    grads_bx = [x.grad.numpy() for x in inputs_bx]
    
    # PyTorch forward + backward
    inputs_pt = [torch.from_numpy(x).cuda().requires_grad_(True) for x in inputs_np]
    output_pt = pytorch_fn(*inputs_pt)
    output_pt.sum().backward()
    grads_pt = [x.grad.cpu().numpy() for x in inputs_pt]
    
    # Compare
    for i, (g_bx, g_pt) in enumerate(zip(grads_bx, grads_pt)):
        assert np.allclose(g_bx, g_pt, rtol=rtol, atol=atol), \
            f"Gradient mismatch for input {i}"
```

**Ops to test:**
- `Add`, `Sub`, `Mul`, `Div`
- `Exp`, `Log`, `Sin`, `Sqrt`, `Neg`
- `Sum`, `Max`, `Mean`
- `Matmul`
- `Reshape`, `Permute`, `Expand`

---

### Assignment 5.2: Implement Movement Op Gradients ⭐⭐

**For the movement ops you implemented in Module 1:**

| Forward Op | Backward Op |
|------------|-------------|
| `pad` | `shrink` (crop out padding) |
| `shrink` | `pad` (add back the cropped regions as zeros) |
| `flip` | `flip` (self-inverse) |

**Implementation location:** `function.py` - Add new Function classes

```python
class Pad(Function):
    @staticmethod
    def forward(ctx, x: LazyBuffer, padding):
        ctx.save_for_backward(x.shape, padding)
        return x.pad(padding)
    
    @staticmethod  
    def backward(ctx, grad_out: LazyBuffer):
        orig_shape, padding = ctx.saved_tensors
        # Shrink grad_out to remove the padded regions
        # YOUR CODE HERE
```

**Hint for Pad.backward:**
```python
# If we padded ((2, 1),) on shape (3,) to get shape (6,):
# Original region is indices [2, 5) in the padded tensor
# shrink limits = ((2, 5),)
limits = tuple((p[0], p[0] + orig_shape[i]) for i, p in enumerate(padding))
return grad_out.shrink(limits)
```

---

### Assignment 5.3: Broadcasting Gradient Accumulation ⭐⭐⭐

**Problem:** When you broadcast a tensor, gradients need to be *summed* back.

```python
a = Tensor([1, 2, 3], requires_grad=True)  # shape (3,)
b = Tensor.rand(4, 3, requires_grad=True)   # shape (4, 3)
c = a + b  # a is broadcast to (4, 3)
c.sum().backward()
# a.grad should have shape (3,), with gradients summed over axis 0
```

**Current code:** `function.py:270-276` - `Expand` class

```python
class Expand(Function):
    @staticmethod
    def backward(ctx, grad_out: LazyBuffer) -> LazyBuffer:
        (input_shape,) = ctx.saved_tensors
        # Need to sum over expanded dimensions
        # YOUR CODE TO VERIFY/FIX
```

**Question:** Does the current implementation handle all cases?
- Expanding a middle dimension: `(2, 1, 3)` → `(2, 5, 3)`
- Expanding with new dimensions: `(3,)` → `(4, 3)`

**tinygrad reference:**
- `tinygrad/gradient.py` → Look for `Expand` gradient rule
- Key insight: backward of expand is reduce (sum) over expanded axes

---
