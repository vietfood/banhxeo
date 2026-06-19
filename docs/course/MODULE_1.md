## Module 1: View And Indexing

The `View` class is the heart of zero-copy tensor operations.

If you understand views, you understand why `transpose`, `reshape`, `expand`,
`slice`, `flip`, and many gradient rules do not need to move data. If you do
not understand views, every later compiler concept will feel haunted.

This module goes slowly on purpose.

## Why This Module Exists

Views are the foundation of the whole compiler.

Every later module assumes that a logical tensor index can be lowered into a
physical storage offset. Fusion, IR, reductions, matmul layout, gradient rules,
and memory planning all depend on that mapping being correct.

The main design decision in this module is to put indexing knowledge in `View`,
not scattered across `Tensor.__getitem__`, `TorchInterpreter`, and
`TritonCodegen`.

That is why this module is intentionally comprehensive. It is cheaper to be slow
here than to debug wrong kernels later.

## Learning Goals

By the end of this module, you should be able to explain:

1. The difference between logical shape and physical storage.
2. How a logical index like `(row, col)` becomes a storage offset.
3. Why contiguous tensors can use `linear_idx` directly.
4. Why transposes are just stride changes.
5. Why broadcasting uses stride `0`.
6. Why slicing changes offset.
7. Why flipping uses negative strides.
8. Why padding is the first movement op that needs a mask or a `where`.
9. Why tinygrad uses `ShapeTracker` instead of one mutable-ish view.

## Mental Model

A tensor view answers one question:

```text
Given a logical index, which element of the underlying storage do I read?
```

For a view:

```python
View(shape=(2, 3), strides=(3, 1), offset=0)
```

the mapping is:

```text
physical_offset = offset + i0 * stride0 + i1 * stride1
```

So:

```text
logical index (0, 0) -> 0 + 0 * 3 + 0 * 1 = 0
logical index (0, 1) -> 0 + 0 * 3 + 1 * 1 = 1
logical index (1, 0) -> 0 + 1 * 3 + 0 * 1 = 3
logical index (1, 2) -> 0 + 1 * 3 + 2 * 1 = 5
```

The storage is still one flat array. Shape and strides are the interpretation.

## Core Vocabulary

`shape`:

The logical dimensions users see.

```python
x.shape == (2, 3)
```

`strides`:

How far to move in physical storage when a logical dimension increases by one.

For row-major contiguous `(2, 3)`:

```python
strides == (3, 1)
```

Moving one row jumps 3 elements. Moving one column jumps 1 element.

`offset`:

Where the view starts inside the underlying storage.

Slicing often changes offset:

```text
x = [10, 20, 30, 40]
y = x[2:]

y.shape = (2,)
y.strides = (1,)
y.offset = 2
```

`linear_idx`:

The flat output index used by a generated kernel.

For shape `(2, 3)`, the logical indices are visited like:

```text
linear_idx 0 -> (0, 0)
linear_idx 1 -> (0, 1)
linear_idx 2 -> (0, 2)
linear_idx 3 -> (1, 0)
linear_idx 4 -> (1, 1)
linear_idx 5 -> (1, 2)
```

`physical_offset`:

The actual storage position after applying strides and offset.

## Linear Index Decomposition

Generated Triton kernels usually start from a vector of flat offsets:

```python
linear_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
```

For a logical shape `(2, 3, 4)`, convert a linear index into coordinates by
peeling dimensions from the right:

```text
i2 = linear_idx % 4
tmp = linear_idx // 4
i1 = tmp % 3
tmp = tmp // 3
i0 = tmp % 2
```

Then apply strides:

```text
physical_offset = offset + i0 * stride0 + i1 * stride1 + i2 * stride2
```

This is the key formula for the whole module.

## Worked Example 1: Contiguous

```python
view = View(shape=(2, 3), strides=(3, 1), offset=0)
linear_idx = 4
```

Decompose:

```text
col = 4 % 3 = 1
tmp = 4 // 3 = 1
row = 1 % 2 = 1
```

Apply strides:

```text
physical_offset = 0 + row * 3 + col * 1
physical_offset = 0 + 1 * 3 + 1 * 1
physical_offset = 4
```

For a contiguous view, this always equals `linear_idx + offset`.

## Worked Example 2: Transpose

Start with a contiguous `(2, 3)` tensor:

```text
shape   = (2, 3)
strides = (3, 1)
```

Transpose it:

```text
shape   = (3, 2)
strides = (1, 3)
```

No data moved. Only the interpretation changed.

For `linear_idx = 4` in the transposed logical shape `(3, 2)`:

```text
col = 4 % 2 = 0
tmp = 4 // 2 = 2
row = 2 % 3 = 2
```

Apply transposed strides:

```text
physical_offset = 0 + row * 1 + col * 3
physical_offset = 2
```

That is correct because logical transposed element `(2, 0)` points at original
storage element `(0, 2)`.

## Worked Example 3: Broadcast

Start with:

```text
shape   = (3, 1)
strides = (1, 1)
```

Expand to `(3, 4)`:

```text
shape   = (3, 4)
strides = (1, 0)
```

The second dimension has stride `0`, so every column reads the same element.

For `linear_idx = 5`:

```text
col = 5 % 4 = 1
tmp = 5 // 4 = 1
row = 1 % 3 = 1

physical_offset = row * 1 + col * 0
physical_offset = 1
```

This is why `expand` can be zero-copy.

## Worked Example 4: Slice

Start with:

```text
x.shape   = (5,)
x.strides = (1,)
x.offset  = 0
```

Slice:

```python
y = x[2:5]
```

The new view is:

```text
y.shape   = (3,)
y.strides = (1,)
y.offset  = 2
```

For `y[0]`:

```text
physical_offset = 2 + 0 * 1 = 2
```

For `y[2]`:

```text
physical_offset = 2 + 2 * 1 = 4
```

Again, no data moved.

## Worked Example 5: Flip

Start with:

```text
x = [10, 20, 30, 40]
shape   = (4,)
strides = (1,)
offset  = 0
```

Flip along axis `0`:

```text
shape   = (4,)
strides = (-1,)
offset  = 3
```

For logical index `0`:

```text
physical_offset = 3 + 0 * -1 = 3
```

For logical index `1`:

```text
physical_offset = 3 + 1 * -1 = 2
```

The same formula works. Negative strides are not special in the math. They are
only special in your confidence.

## Worked Example 6: Pad

Padding is different.

```python
x = Tensor([1, 2, 3])
y = x.pad(((2, 1),))
```

Logically:

```text
y = [0, 0, 1, 2, 3, 0]
```

The padded zeros do not exist in the original storage. A normal view can point
to existing storage, but it cannot point to values that are not there.

So `pad` needs one of these designs:

1. Store a mask on the `View`.
2. Lower `pad` into `where(in_bounds, load(...), 0)`.
3. Materialize a new padded buffer.

For learning a compiler, option 2 is the most conceptually useful. For a small
first implementation, option 1 can be okay. Option 3 is simplest but teaches the
least about zero-copy movement.

## Assignment 1.0: Trace Current View Behavior

Difficulty: ⭐

Before writing code, trace what banhxeo currently does.

Run:

```python
from banhxeo import Tensor

x = Tensor([[1, 2, 3], [4, 5, 6]], device="cuda")
y = x.T
z = (y + 1).realize()
```

Questions:

1. What is `x.lazydata.view`?
2. What is `y.lazydata.view`?
3. Does `x.T` allocate a new data buffer?
4. What does the generated indexing code look like?
5. Which variable in Triton is the logical linear index?
6. Which variable becomes the physical input offset?

Deliverable:

Add notes to `docs/SOLUTION.md` with the exact view values and generated
indexing code.

Why this assignment:

You should see the current duplication before removing it. Otherwise
`View.to_index_expr()` will feel like cleanup instead of the central abstraction
that makes codegen less fragile.

## Assignment 1.1: Implement `View.to_index_expr()`

Difficulty: ⭐⭐

Current problem:

Indexing logic is duplicated in `backend/triton.py`.

Your task:

Create one source of truth for index computation:

```python
def to_index_expr(self, linear_idx: str, var_prefix: str) -> Tuple[str, str]:
    """
    Convert a logical linear index to a physical memory offset expression.

    Args:
        linear_idx: Variable name holding the logical linear index.
        var_prefix: Prefix for generated temporary variables.

    Returns:
        A tuple of `(generated_code, offset_variable_name)`.
    """
```

Rules:

1. Generate code as strings.
2. Do not compute offsets in Python.
3. Use `self.shape`, `self.strides`, and `self.offset`.
4. Return the name of the final offset variable.
5. Make generated variable names deterministic.

Naive implementation:

Always decompose the linear index into coordinates and apply strides.

Better implementation:

Start naive first. Then optimize contiguous views in Assignment 1.2.

Suggested generated shape for `(2, 3)`:

```python
temp_offset = 0
temp_idx = linear_offsets
temp_idx_1 = temp_idx % 3
temp_idx = temp_idx // 3
temp_offset += temp_idx_1 * 1
temp_idx_0 = temp_idx % 2
temp_idx = temp_idx // 2
temp_offset += temp_idx_0 * 3
```

The exact variable names can differ, but the math should not.

Tests to write:

```python
def test_index_expr_contiguous_2d():
    view = View.create((2, 3))
    code, offset = view.to_index_expr("idx", "x")
    assert "x_offset" in offset
    assert "% 3" in code
    assert "* 1" in code
    assert "* 3" in code


def test_index_expr_transpose_2d():
    view = View.create((2, 3)).permute((1, 0))
    code, _ = view.to_index_expr("idx", "x")
    assert "* 1" in code
    assert "* 3" in code


def test_index_expr_broadcast_stride_zero():
    view = View.create((3, 1)).broadcast_to((3, 4))
    code, _ = view.to_index_expr("idx", "x")
    assert "* 0" in code
```

Study prompt:

Why does the loop run over dimensions in reverse order?

## Assignment 1.2: Optimize Contiguous Indexing

Difficulty: ⭐

Observation:

For a contiguous view:

```text
physical_offset = linear_idx + offset
```

So this:

```python
View(shape=(2, 3), strides=(3, 1), offset=0)
```

does not need modulo/divide decomposition.

Task:

Update `to_index_expr()` so contiguous views emit simpler code.

Expected output shape:

```python
x_offset = idx
```

or, if offset is nonzero:

```python
x_offset = idx + 5
```

Questions:

1. Should a scalar shape `()` be contiguous?
2. Should a zero-size tensor be contiguous?
3. Should `offset != 0` still count as contiguous?

Recommended answer for banhxeo:

Use `is_contiguous()` for strides only, then add offset separately. Contiguity
means the logical layout has row-major strides. It does not mean the view starts
at storage position zero.

## Assignment 1.3: Replace Duplicated Triton Indexing

Difficulty: ⭐⭐

Now use `View.to_index_expr()` in `backend/triton.py`.

Current duplicated places:

- `render_indexing()`
- `FROM_NUMPY` / `FROM_TORCH` handling

Goal:

All input view indexing should call the same method.

Questions:

1. Which buffers need shape and stride metadata in the kernel signature?
2. Which buffers can bake shape and stride constants into generated code?
3. Should `to_index_expr()` use `self.shape` and `self.strides` constants, or
   names like `in_0_shape_0` and `in_0_stride_0`?

Recommended first choice:

Use constants from `View` inside `to_index_expr()`. It is simpler and makes
generated code easier to read. You can generalize later if dynamic shapes become
a goal.

Checkpoint:

Run a transpose example with `DEBUG=1` and confirm the generated code still has
the same indexing behavior.

## Assignment 1.4: Build A Tiny View Test Suite

Difficulty: ⭐⭐

Create tests before adding more movement ops.

Minimum forward tests:

```python
def test_transpose_matches_torch():
    ...


def test_slice_matches_torch():
    ...


def test_expand_matches_torch():
    ...


def test_permute_then_elementwise_matches_torch():
    ...
```

What to compare:

- output shape
- output values
- generated kernel does not crash

Use tiny shapes. The goal is not performance.

Important:

Include at least one test where the input is non-contiguous before an
elementwise op:

```python
x = Tensor.rand((2, 3), device="cuda")
y = (x.T + 1).numpy()
```

This catches indexing bugs immediately.

## Assignment 1.5: Understand View Composition

Difficulty: ⭐⭐⭐

Problem:

Chained views can be tricky:

```python
x = Tensor.rand((2, 3, 4))
y = x.permute((2, 0, 1))
z = y.reshape((4, 6))
```

Questions:

1. Can this reshape be represented by one new `View`?
2. If yes, what are the new strides?
3. If no, should banhxeo error or materialize with `contiguous()`?

Current banhxeo design:

banhxeo stores one `View` on each `LazyBuffer`. If an existing `LoadOp.VIEW`
gets another movement op, the code tries to update the view and keep the same
source.

tinygrad design:

tinygrad uses a `ShapeTracker`, which can hold a stack of `View` objects.

Tradeoff:

| Design | Benefit | Cost |
| --- | --- | --- |
| Single `View` | Small and easy to print | Some compositions are hard or impossible |
| `ShapeTracker` list | More expressive | More lowering complexity |

Task:

Improve error messages in `View.reshape()` so failure explains which stride
relationship broke.

Do not implement a full ShapeTracker yet.

Study prompt:

Write down one example where a reshape after permute can be a view, and one
where it cannot.

## Assignment 1.6: Implement `View.shrink()`

Difficulty: ⭐⭐

`shrink` crops a tensor to absolute bounds.

Example:

```python
x = Tensor.rand((4, 4))
y = x.shrink(((1, 3), (1, 3)))
```

Expected logical shape:

```text
(2, 2)
```

View rule:

```text
new_shape = end_i - start_i
new_offset = old_offset + sum(start_i * stride_i)
new_strides = old_strides
```

Implementation sketch:

```python
def shrink(self, limits: Tuple[Tuple[int, int], ...]) -> "View":
    if len(limits) != len(self.shape):
        raise ValueError(...)

    new_shape = []
    new_offset = self.offset

    for i, (start, end) in enumerate(limits):
        if start < 0 or end > self.shape[i] or start > end:
            raise ValueError(...)
        new_shape.append(end - start)
        new_offset += start * self.strides[i]

    return View(tuple(new_shape), self.strides, new_offset)
```

Questions:

1. Is `shrink` just `slice` with a clearer name?
2. Should `Tensor.__getitem__` lower to `shrink` eventually?
3. How should negative bounds behave?

Recommended first choice:

Do not support negative bounds in `shrink`. Keep it absolute and boring.

## Assignment 1.7: Implement `View.flip()`

Difficulty: ⭐⭐

`flip` reverses one or more axes using negative strides.

For one axis:

```python
def flip(self, axis: int) -> "View":
    new_strides = list(self.strides)
    new_strides[axis] = -new_strides[axis]
    new_offset = self.offset + (self.shape[axis] - 1) * self.strides[axis]
    return View(self.shape, tuple(new_strides), new_offset)
```

Tests:

```python
x = Tensor([1, 2, 3, 4], device="cuda")
assert x.flip(0).numpy().tolist() == [4, 3, 2, 1]

x = Tensor([[1, 2], [3, 4]], device="cuda")
assert x.flip(0).numpy().tolist() == [[3, 4], [1, 2]]
assert x.flip(1).numpy().tolist() == [[2, 1], [4, 3]]
assert x.flip(0).flip(0).numpy().tolist() == x.numpy().tolist()
```

Questions:

1. Does `to_index_expr()` need special handling for negative strides?
2. Does `View.is_contiguous()` return false for flipped views?
3. Should `.contiguous()` on a flipped view produce a positive-stride output?

Recommended answer:

The indexing formula already handles negative strides. `.contiguous()` should
materialize into normal row-major storage.

## Assignment 1.8: Design `pad`

Difficulty: ⭐⭐⭐

Do not implement `pad` until you can explain `shrink` and `flip`.

Example:

```python
x = Tensor([1, 2, 3], device="cuda")
y = x.pad(((2, 1),))
```

Logical result:

```text
[0, 0, 1, 2, 3, 0]
```

Why this is harder:

The zero values outside the original tensor do not exist in storage.

Design A: `View.mask`

```python
@dataclass
class View:
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    offset: int = 0
    mask: Optional[Tuple[Tuple[int, int], ...]] = None
```

For `pad(((2, 1),))`, the valid logical region is:

```text
mask = ((2, 5),)
```

Codegen does:

```python
value = tl.where(in_bounds, tl.load(ptr + offset), 0.0)
```

Design B: lower pad to `where`

Instead of storing a mask in `View`, keep `pad` as a movement op and lower it to
a `where` expression during indexing/codegen.

Design C: materialize

Allocate a new padded tensor and copy values into it.

This is easiest but least educational.

Recommended first choice:

Write a design note comparing A and B before implementing either.

Questions:

1. Does a mask belong to a view, or is it really a computation?
2. How does `pad` interact with `shrink`?
3. What is the backward pass of `pad`?
4. Can `pad` be fused with `x + 1`?

## Assignment 1.9: Movement Op Gradients Preview

Difficulty: ⭐⭐

You do not need to implement these yet, but you should understand the pairings:

| Forward op | Backward op |
| --- | --- |
| `reshape` | `reshape` back |
| `permute` | inverse `permute` |
| `expand` | `sum` over expanded dimensions |
| `slice` / `shrink` | `pad` |
| `pad` | `shrink` |
| `flip` | `flip` |

Study prompt:

Why is `expand` backward a reduction?

Example:

```python
a = Tensor([1, 2, 3], requires_grad=True)
b = a.expand((4, 3))
c = b.sum()
c.backward()
```

Each element of `a` was used four times, so each gradient must sum four
contributions.

## Assignment 1.10: Read tinygrad Carefully

Difficulty: ⭐⭐⭐

Read these tinygrad files after you have attempted the assignments:

- `tinygrad/shape/view.py`
- `tinygrad/shape/shapetracker.py`
- `tinygrad/schedule/indexing.py`
- `tinygrad/tensor.py` movement ops

Questions to answer:

1. What does tinygrad store in a `View` that banhxeo does not?
2. Why does tinygrad stack views in `ShapeTracker`?
3. How does tinygrad represent masks?
4. When does tinygrad convert movement behavior into `where`?
5. Which tinygrad design would be overkill for banhxeo right now?

Deliverable:

Write a short note titled "What banhxeo should copy from tinygrad views, and
what it should postpone."

## Module 1 Checklist

- [ ] Trace current transpose behavior.
- [ ] Implement `View.to_index_expr()`.
- [ ] Optimize contiguous indexing.
- [ ] Remove duplicated indexing logic in Triton codegen.
- [ ] Add forward tests for transpose, slice, expand, and permute.
- [ ] Improve reshape failure messages.
- [ ] Implement `View.shrink()`.
- [ ] Implement `View.flip()`.
- [ ] Write a design note for `pad`.
- [ ] Read tinygrad's view and ShapeTracker code.

## Exit Criteria

You are ready for Module 2 when:

1. You can manually compute physical offsets for contiguous, transposed,
   broadcasted, sliced, and flipped views.
2. The generated Triton indexing code has one source of truth.
3. View tests compare against PyTorch for small tensors.
4. You can explain why padding needs a mask or a `where`.
5. You can explain why a full ShapeTracker is useful but not urgent.
