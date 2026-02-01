# banhxeo Backend Course

A self-guided course to build a robust tensor compiler backend. Each module contains assignments with hints, naive vs. proper approaches, and tinygrad references.

**Philosophy:** You learn best by implementing, getting stuck, then discovering better solutions.

---

## How to Use This Course

1. **Read the assignment** - Understand what you need to build
2. **Try it yourself first** - Spend 30-60 min before looking at hints
3. **Check hints when stuck** - Conceptual guidance, not code
4. **Compare approaches** - Understand why naive solutions fail
5. **Study tinygrad** - See how the masters solved it
6. **Implement & test** - Write tests before or with your code

**Difficulty ratings:**
- ⭐ Straightforward
- ⭐⭐ Requires careful thinking
- ⭐⭐⭐ Challenging, multiple approaches possible

---

## Module 0: Understanding the Current Codebase

Before implementing anything, you need to deeply understand what exists.

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

**Hint:** Add `DEBUG=2` environment variable and trace through:
- `tensor.py` → `__add__` → `Add.apply()` 
- `function.py:137-144` → `Add.forward()`
- `buffer.py:184-193` → `compute_ops()`
- `tensor.py:595` → `realize()`
- `backend/__init__.py:299-319` → `exec()`

**Deliverable:** Write a document explaining the full flow with line numbers.

---

### Assignment 0.2: Trace a View Operation ⭐⭐

**Task:** Trace what happens with:
```python
x = Tensor.rand(2, 3, device="cuda")
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

---

## Module 1: View & Indexing

The `View` class is the heart of zero-copy tensor operations. Master this first.

### Assignment 1.1: Implement `View.to_index_expr()` ⭐⭐

**Current problem:** Indexing logic is duplicated in `triton.py`:
- Lines 59-95: `render_indexing()` 
- Lines 145-168: Same logic repeated for `FROM_NUMPY/FROM_TORCH`

**Your task:** Create a single source of truth for index computation.

```python
# view.py - Add this method
def to_index_expr(self, linear_idx: str, var_prefix: str) -> Tuple[str, str]:
    """
    Convert a linear index to a physical memory offset.
    
    Args:
        linear_idx: Variable name holding the linear index (e.g., "idx")
        var_prefix: Prefix for generated temp variables (e.g., "temp")
    
    Returns:
        (generated_code, offset_variable_name)
    
    Example:
        For shape=(2,3), strides=(3,1), offset=0:
        Input linear_idx=5 should map to:
          row = 5 // 3 = 1
          col = 5 % 3 = 2  
          offset = 1*3 + 2*1 = 5 ✓
    """
    # YOUR CODE HERE
    pass
```

**Hints:**
1. Linear index `i` for shape `(d0, d1, d2)` decomposes as:
   - `i2 = i % d2`
   - `i1 = (i // d2) % d1`
   - `i0 = (i // d2 // d1) % d0`
2. Physical offset = `sum(idx_k * stride_k for k in dims) + base_offset`
3. Generate string code, not compute values (this runs at codegen time)

**Test cases to handle:**
```python
# Contiguous (2, 3) with strides (3, 1)
# linear_idx=4 → row=1, col=1 → offset=4

# Transposed (3, 2) with strides (1, 3)  
# linear_idx=4 → row=1, col=1 → offset=1*1 + 1*3 = 4

# Broadcast (3, 1) expanded to (3, 4) with strides (1, 0)
# linear_idx=5 → row=1, col=1 → offset=1*1 + 1*0 = 1
```

**Naive approach:** Generate code that always does full decomposition even for contiguous tensors.

**Better approach:** Check if contiguous first, use `linear_idx` directly.

**tinygrad reference:**
- `tinygrad/shape/shapetracker.py` → `ShapeTracker.to_indexed_uops()` 
- `tinygrad/schedule/indexing.py` → Study how indices are computed
- Key insight: tinygrad represents index computation as UOps, not strings

**Further exploration:** What if the view has a `mask` (for padding)? How would you modify the generated code?

---

### Assignment 1.2: Implement Contiguity Check Optimization ⭐

**Observation:** For contiguous tensors, index computation is trivial:
```python
# Contiguous: physical_offset = linear_idx + base_offset
# Non-contiguous: need full stride math
```

**Your task:** Optimize `to_index_expr()` for the contiguous case.

**Hint:** Check `View.is_contiguous()` (line 12-16) and emit simpler code.

**Test:** Verify generated kernel code is simpler for contiguous vs non-contiguous inputs.

---

### Assignment 1.3: Handle View Composition ⭐⭐⭐

**Problem:** What happens with chained views?
```python
x = Tensor.rand(2, 3, 4)
y = x.permute(2, 0, 1)  # View 1: shape (4,2,3), strides (1, 12, 4)
z = y.reshape(4, 6)      # View 2: shape (4,6), strides ???
```

**Question:** Can `reshape` always be done as a pure view change? When does it need a copy?

**Current code:** `view.py:133-192` - `reshape()` method

**Your task:** 
1. Understand when reshape fails for non-contiguous views
2. Add better error messages explaining *why* it failed
3. Consider: should you auto-trigger `contiguous()` or let the user handle it?

**tinygrad reference:**
- `tinygrad/shape/shapetracker.py` → `ShapeTracker` class
- Key insight: tinygrad stacks multiple `View` objects. The `ShapeTracker` composes them.
- `tinygrad/shape/view.py:150-200` → `View.reshape()` - more sophisticated logic

**Question to ponder:** banhxeo uses a single `View`. tinygrad uses `ShapeTracker` with a list of `View`s. What are the tradeoffs?

---

### Assignment 1.4: Implement `View.pad()` ⭐⭐⭐

**Current state:** `MovementOp.PAD` is a TODO stub at `buffer.py:60`

**Challenge:** Padding adds zeros around a tensor without copying data.

```python
x = Tensor([1, 2, 3])
y = x.pad(((2, 1),))  # [0, 0, 1, 2, 3, 0]
```

**Naive approach:** Actually allocate a bigger tensor and copy data into it.

**Better approach:** Use a view with a *mask* indicating valid regions. In codegen, use `tl.where(in_bounds, loaded_value, 0.0)`.

**Implementation steps:**
1. Add `mask: Optional[Tuple[Tuple[int, int], ...]]` field to `View` dataclass
2. Implement `View.pad()` that computes new shape and mask
3. Modify `TritonCodegen` to handle masked loads

**Hints for `View.pad()`:**
```python
def pad(self, padding: Tuple[Tuple[int, int], ...]) -> "View":
    # padding = ((before_0, after_0), (before_1, after_1), ...)
    
    # New shape: original + padding on each side
    new_shape = ???
    
    # Mask: which indices in new shape map to valid data
    # For padding ((2,1),) on shape (3,), valid region is indices [2, 3, 4]
    mask = ???
    
    # Offset adjustment: when we access new_idx=2, we want old_idx=0
    # So we need to subtract the "before" padding
    new_offset = ???
    
    return View(new_shape, self.strides, new_offset, mask)
```

**Hints for codegen:**
```python
# In render_indexing or similar:
if view.mask is not None:
    # Generate bounds check for each dimension
    # in_bounds = (idx_0 >= start_0) & (idx_0 < end_0) & ...
    # value = tl.where(in_bounds, loaded_value, 0.0)
```

**tinygrad reference:**
- `tinygrad/tensor.py:1200-1250` → `Tensor.pad()` - user API
- `tinygrad/schedule/indexing.py:100-150` → `convert_pad_to_where_to_keep_behavior_local()`
- Key insight: tinygrad converts PAD to WHERE during lowering, not in View

**Question:** Should the mask be stored in View, or should PAD create a WHERE op? What are the tradeoffs?

---

### Assignment 1.5: Implement `View.shrink()` ⭐⭐

**Task:** Crop a tensor to a subregion.

```python
x = Tensor.rand(4, 4)
y = x.shrink(((1, 3), (1, 3)))  # Extract center 2x2
```

**Observation:** This is very similar to `View.slice()` (line 99-131). Consider:
1. Should you just rename/refactor `slice()`?
2. What's the API difference between `slice` and `shrink`?

**Hint:** `shrink` uses absolute `(start, end)` bounds. `slice` might use Python slice semantics with steps.

**Implementation:**
```python
def shrink(self, limits: Tuple[Tuple[int, int], ...]) -> "View":
    # limits = ((start_0, end_0), (start_1, end_1), ...)
    
    # New shape: (end_i - start_i) for each dim
    new_shape = ???
    
    # New offset: original + sum(start_i * stride_i)
    new_offset = ???
    
    # Strides stay the same!
    return View(new_shape, self.strides, new_offset)
```

**Test:** `shrink` is the inverse of `pad`. Verify: `x.pad(p).shrink(computed_limits) == x`

**tinygrad reference:**
- `tinygrad/mixin/movement.py` → `Tensor.shrink()`
- Note how tinygrad handles negative indices and edge cases

---

### Assignment 1.6: Implement `View.flip()` ⭐⭐

**Task:** Reverse a tensor along an axis using negative strides.

```python
x = Tensor([1, 2, 3, 4])
y = x.flip(0)  # [4, 3, 2, 1]
```

**Key insight:** Flipping doesn't copy data. It uses a *negative stride*.

```
Original: data = [1, 2, 3, 4], stride = 1, offset = 0
  idx=0 → data[0] = 1
  idx=1 → data[1] = 2
  
Flipped: data = [1, 2, 3, 4], stride = -1, offset = 3
  idx=0 → data[3 + 0*(-1)] = data[3] = 4
  idx=1 → data[3 + 1*(-1)] = data[2] = 3
```

**Implementation:**
```python
def flip(self, axis: int) -> "View":
    # Negate the stride for this axis
    new_strides = list(self.strides)
    new_strides[axis] = -self.strides[axis]
    
    # Offset moves to the "end" of this dimension
    new_offset = self.offset + (self.shape[axis] - 1) * self.strides[axis]
    
    return View(self.shape, tuple(new_strides), new_offset)
```

**Challenge:** Does your `to_index_expr()` handle negative strides correctly?

The formula `offset += idx_i * stride_i` should work mathematically, but verify:
- `idx=0, stride=-1, offset=3` → `3 + 0*(-1) = 3` ✓
- `idx=1, stride=-1, offset=3` → `3 + 1*(-1) = 2` ✓

**Test cases:**
```python
# 1D flip
x = Tensor([1, 2, 3, 4])
assert x.flip(0).numpy() == [4, 3, 2, 1]

# 2D flip along axis 0 (flip rows)
x = Tensor([[1, 2], [3, 4]])
assert x.flip(0).numpy() == [[3, 4], [1, 2]]

# 2D flip along axis 1 (flip columns)
assert x.flip(1).numpy() == [[2, 1], [4, 3]]

# Double flip = identity
assert (x.flip(0).flip(0)).numpy() == x.numpy()
```

**tinygrad reference:**
- `tinygrad/mixin/movement.py` → `Tensor.flip()`
- `tinygrad/shape/view.py` → How View handles negative strides

---

## Module 2: Codegen Architecture

Move from string concatenation to a proper intermediate representation.

### Assignment 2.1: Analyze Current Codegen ⭐⭐

**Task:** List all the problems with the current `TritonCodegen` approach.

**Current code:** `triton.py:15-302`

**Questions to answer:**
1. Why is string concatenation fragile?
2. What happens if you want to add a new op?
3. How would you add optimization passes (e.g., constant folding)?
4. How would you add a new backend (CUDA C, Metal)?

**Problems to identify:**
- Duplicated indexing logic (lines 59-95 vs 145-168)
- Hard to test individual components
- No separation between "what to compute" and "how to render it"
- Op-specific code scattered across visit methods

---

### Assignment 2.2: Design a KernelOp IR ⭐⭐

**Task:** Design an intermediate representation that separates concerns.

**Goal:** 
```
LazyBuffer DAG  →  List[KernelOp]  →  Triton Source
                   (what to compute)   (how to render)
```

**Your IR should support:**
- Variable definitions
- Memory loads (with optional masks)
- Memory stores
- Arithmetic operations
- Loops (for future reduce fusion)
- Conditionals (for masks, bounds checks)

**Starter design:**
```python
from dataclasses import dataclass
from enum import Enum, auto

class IRType(Enum):
    DEFINE = auto()   # Define a variable
    LOAD = auto()     # Load from memory
    STORE = auto()    # Store to memory
    ALU = auto()      # Arithmetic/logic operation
    # What else do you need?

@dataclass(frozen=True)
class KernelOp:
    type: IRType
    name: str           # Output variable name
    args: Tuple[str, ...]  # Input variable names or constants
    # What other fields?
```

**Questions to consider:**
1. Should dtype be part of KernelOp?
2. How do you represent `tl.exp(x)` vs `x + y`?
3. How do you represent `tl.where(cond, a, b)`?

**tinygrad reference:**
- `tinygrad/uop/ops.py` → `UOp` class (lines 1-100)
- `tinygrad/uop/__init__.py` → `Ops` enum (all operation types)
- Key insight: UOp is *immutable* and *deduplicated* (same op = same object)

**Question:** Why does tinygrad make UOp immutable and cached?

---

### Assignment 2.3: Implement IRBuilder ⭐⭐

**Task:** Create a builder that emits KernelOps instead of strings.

```python
class IRBuilder:
    def __init__(self):
        self.ops: List[KernelOp] = []
    
    def load(self, name: str, ptr: str, offset: str, mask: Optional[str] = None):
        """Emit a load operation."""
        # YOUR CODE HERE
    
    def alu(self, name: str, op: str, *operands: str):
        """Emit an arithmetic operation."""
        # YOUR CODE HERE
    
    def store(self, ptr: str, offset: str, value: str, mask: Optional[str] = None):
        """Emit a store operation."""
        # YOUR CODE HERE
```

**Usage example:**
```python
builder = IRBuilder()
builder.load("a", "a_ptr", "idx")
builder.load("b", "b_ptr", "idx")
builder.alu("c", "+", "a", "b")
builder.store("out_ptr", "idx", "c")

# builder.ops now contains the IR
```

**Test:** Convert a simple elementwise kernel to IR, verify it captures all operations.

---

### Assignment 2.4: Implement TritonRenderer ⭐⭐

**Task:** Convert KernelOp IR to Triton source code.

```python
class TritonRenderer:
    def render(self, ops: List[KernelOp], 
               input_args: List[str], 
               output_shape: Tuple[int, ...]) -> str:
        """Generate complete Triton kernel source."""
        # YOUR CODE HERE
```

**Rendering rules:**
```python
# LOAD: name = tl.load(ptr + offset, mask=mask)
# STORE: tl.store(ptr + offset, value, mask=mask)
# ALU "+": name = a + b
# ALU "tl.exp": name = tl.exp(a)
```

**Test:** Render IR, compile with `compile_triton_src()`, execute, verify correctness.

**tinygrad reference:**
- `tinygrad/renderer/__init__.py` → `Renderer` base class
- `tinygrad/renderer/cstyle.py` → `CStyleLanguage` (lines 1-200)
- See how tinygrad maps UOps to C-style code strings

---

### Assignment 2.5: Add an Optimization Pass ⭐⭐⭐

**Task:** Implement constant folding on your IR.

**Example:**
```python
# Before optimization:
DEFINE const_0 = 2.0
DEFINE const_1 = 3.0
ALU temp = const_0 + const_1  # Can be folded to 5.0!
ALU result = x * temp

# After optimization:
ALU result = x * 5.0
```

**Implementation approach:**
1. Walk the IR list
2. For ALU ops where all inputs are constants, compute the result
3. Replace with a new constant definition

**Hint:** This is easier with immutable IR (you create new ops, don't mutate).

**tinygrad reference:**
- `tinygrad/uop/ops.py:200-300` → `UOp.simplify()` and algebraic rules
- `tinygrad/uop/ops.py:400-500` → Pattern matching for rewrites
- Key insight: tinygrad uses a `PatternMatcher` for optimization rules

**Further exploration:** What other optimizations could you add?
- Dead code elimination (remove ops whose results aren't used)
- Common subexpression elimination (reuse identical computations)
- Load/store fusion (combine adjacent memory operations)

---

## Module 3: Fusion & Scheduling

Decide which ops go into which kernels.

### Assignment 3.1: Formalize Barrier Rules ⭐⭐

**Current code:** `backend/__init__.py:105-121`

```python
def is_barrier(self, buf: LazyBuffer):
    if buf.realized is not None:
        return True
    if isinstance(buf.op, LoadOp):
        if len(buf.src) != 0 and buf.src[0].realized is None:
            return True
    return isinstance(buf.op, ReduceOp) or buf.op == BinaryOp.MATMUL
```

**Problems:**
1. Logic is implicit, scattered
2. No documentation of *why* these are barriers
3. Hard to add new barrier ops

**Your task:** Create explicit, documented barrier definitions.

```python
# buffer.py - Add after Op TypeAlias (line 71)

# Why these are barriers:
# - ReduceOp: Requires coordination across threads (not embarrassingly parallel)
# - MATMUL: Uses specialized tiled kernel, different parallelization strategy
# - Realized buffers: Already computed, just need to load
BARRIER_OPS = frozenset({BinaryOp.MATMUL, ReduceOp.SUM, ReduceOp.MAX})

def is_barrier(op: Op) -> bool:
    """Check if an op forces a kernel boundary."""
    # YOUR IMPLEMENTATION
```

**tinygrad reference:**
- `tinygrad/schedule/rangeify.py:200-300` → `split_kernels()`
- Look for conditions that trigger `split_store()`
- Key insight: tinygrad splits based on "outer range" compatibility

---

### Assignment 3.2: Implement Fusion Analysis ⭐⭐⭐

**Task:** Given a LazyBuffer DAG, determine which ops can be fused.

```python
def analyze_fusion(output: LazyBuffer) -> List[Set[LazyBuffer]]:
    """
    Partition the DAG into fusible groups.
    
    Returns:
        List of sets, where each set contains LazyBuffers 
        that can be compiled into a single kernel.
    """
    # YOUR CODE HERE
```

**Fusion rules to implement:**

| Rule | Description | Example |
|------|-------------|---------|
| Elementwise chain | Consecutive elementwise ops | `(a+b)*c` → 1 kernel |
| Broadcast fusion | Broadcast followed by elementwise | `a + b.expand()` → 1 kernel |
| View fusion | VIEW is free, fuses with anything | `x.T + y` → 1 kernel |
| Barrier break | Reduce/Matmul break fusion | `a.sum() + b` → 2 kernels |

**Algorithm hint:**
1. Start from output, work backwards
2. For each op, check if it's a barrier
3. If barrier, start new fusion group
4. Otherwise, merge with parent's group

**Test cases:**
```python
# Should be 1 kernel
a = Tensor.rand(10)
b = Tensor.rand(10)
c = (a + b) * 2 - 1

# Should be 2 kernels (reduce is barrier)
x = Tensor.rand(10, 10)
y = x.sum(axis=1) + 1

# Should be 2 kernels (matmul is barrier)
a = Tensor.rand(10, 10)
b = Tensor.rand(10, 10)
c = (a @ b) * 2
```

**tinygrad reference:**
- `tinygrad/engine/schedule.py` → `create_schedule()`
- `tinygrad/schedule/rangeify.py` → `get_rangeify_map()`, `split_kernels()`
- Study the conditions in `can_fuse()` or equivalent

---

### Assignment 3.3: Add Fusion Logging ⭐

**Task:** Add debug output showing fusion decisions.

When `DEBUG >= 2`, print:
```
[FUSION] Building kernel for: <final_op>
[FUSION]   Fused ops: ADD, MUL, SUB (3 ops)
[FUSION]   Barrier deps: SUM at shape (10,)
[FUSION]   Input buffers: 2 pointers
```

**Location:** `backend/__init__.py:150-168` (`gencode()` method)

**Why this matters:** When debugging performance, you need to know what got fused and what didn't.

---

## Module 4: Specialized Kernels

Improve matmul and reduce implementations.

### Assignment 4.1: Understand the Matmul Kernel ⭐⭐

**Task:** Annotate every line of `kernels/matmul.py` with comments explaining:
1. What each variable represents
2. Why that particular optimization is used
3. What would happen without it

**Key concepts to understand:**
- Tiling: Why BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K?
- Memory coalescing: Why the specific access pattern?
- L2 cache optimization: What does GROUP_SIZE_M do?
- Accumulator: Why accumulate in registers?

**Resources:**
- [Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [NVIDIA matmul optimization](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)

---

### Assignment 4.2: Implement Batched Matmul ⭐⭐⭐

**Current limitation:** Only 2D matmul at `backend/__init__.py:226-260`

**Task:** Support these cases:
1. `[B, M, K] @ [B, K, N]` → `[B, M, N]` (batched)
2. `[B, M, K] @ [K, N]` → `[B, M, N]` (broadcast right)
3. `[M, K] @ [B, K, N]` → `[B, M, N]` (broadcast left)

**Implementation steps:**

1. **Update `buffer.py:320-334`** - Handle batch dimensions in `matmul()`:
```python
def matmul(self, other: "LazyBuffer"):
    # Determine output shape based on input shapes
    # Handle 2D, 3D, and broadcasting cases
    # YOUR CODE HERE
```

2. **Create batched kernel** - `kernels/matmul.py`:
```python
@triton.jit
def batched_matmul_kernel(
    A, B, C,
    batch_stride_a, batch_stride_b, batch_stride_c,  # New!
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, ...
):
    # Get batch index from program_id
    batch_id = tl.program_id(2)  # New dimension!
    
    # Offset pointers by batch
    A = A + batch_id * batch_stride_a
    B = B + batch_id * batch_stride_b
    C = C + batch_id * batch_stride_c
    
    # Rest is same as 2D kernel
```

3. **Update `exec_matmul()`** - Handle batch dimension in grid:
```python
grid = lambda META: (
    triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    1,
    B,  # Batch dimension!
)
```

**Hint for broadcast:** If one input doesn't have batch dim, its `batch_stride = 0` (same as expand with stride 0).

**Test cases:**
```python
# Standard batched
a = Tensor.rand(4, 10, 20, device="cuda")
b = Tensor.rand(4, 20, 30, device="cuda")
c = a @ b
assert c.shape == (4, 10, 30)

# Broadcast right operand
a = Tensor.rand(4, 10, 20, device="cuda")
b = Tensor.rand(20, 30, device="cuda")
c = a @ b
assert c.shape == (4, 10, 30)
```

**tinygrad reference:**
- tinygrad handles batched matmul by reshaping/permuting to 2D, doing 2D matmul, then reshaping back
- Alternative: explicit batched kernel (what we're doing)

---

### Assignment 4.3: Implement Multi-Axis Reduce ⭐⭐⭐

**Current limitation:** `_reduce()` in `tensor.py:234-280` only handles single axis.

**Task:** Support `x.sum(axis=(0, 2))` reducing multiple axes at once.

**Current approach:**
```python
# For x.sum(axis=(0, 2)) with shape (2, 3, 4):
# 1. Permute to move reduce axes to end: (3, 2, 4) with axes (1, 0, 2)
# 2. Reshape to 2D: (3, 8) 
# 3. Reduce last axis: (3, 1)
# 4. Reshape to output: (3,) or (1, 3, 1) depending on keepdim
```

**Your task:** Implement this properly.

```python
def _reduce(self, op: ReduceOp, axis=None, keepdim=False):
    if axis is None:
        # Reduce all dimensions
        flat = self.reshape((math.prod(self.shape),))
        result = flat._reduce(op, axis=0)
        return result if not keepdim else result.reshape((1,) * len(self.shape))
    
    if isinstance(axis, int):
        axis = (axis,)
    
    # Normalize negative axes
    axis = tuple(a if a >= 0 else len(self.shape) + a for a in axis)
    
    # YOUR IMPLEMENTATION:
    # 1. Permute reduce axes to end
    # 2. Merge reduce axes into one
    # 3. Do single-axis reduce
    # 4. Handle keepdim
```

**Test cases:**
```python
x = Tensor.rand(2, 3, 4, device="cuda")

# Single axis
assert x.sum(axis=1).shape == (2, 4)

# Multiple axes
assert x.sum(axis=(0, 2)).shape == (3,)

# All axes
assert x.sum().shape == ()

# With keepdim
assert x.sum(axis=1, keepdim=True).shape == (2, 1, 4)
```

**tinygrad reference:**
- `tinygrad/tensor.py` → `Tensor._reduce()` 
- Note how tinygrad handles the axis parameter and keepdim

---

### Assignment 4.4: Reduce Kernel with Tree Reduction ⭐⭐⭐

**Problem:** Current reduce kernel (`kernels/reduce.py:5-37`) uses simple loop.

For very large N (reduction dimension), this is slow because:
1. One thread does all the work for each row
2. No parallelism within the reduction

**Better approach:** Tree reduction with multiple threads cooperating.

```
Level 0: [a, b, c, d, e, f, g, h]  (8 elements, 4 threads)
Level 1: [a+b, c+d, e+f, g+h]      (4 elements, 2 threads)
Level 2: [a+b+c+d, e+f+g+h]        (2 elements, 1 thread)
Level 3: [total]                    (1 element)
```

**This is advanced.** Start by understanding the current kernel, then study:
- [Triton reduce tutorial](https://triton-lang.org/main/getting-started/tutorials/04-reduction.html)
- [NVIDIA parallel reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

---

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

## Module 6: Testing Infrastructure

You have no tests. Fix this first!

### Assignment 6.1: Set Up Test Framework ⭐

**Task:** Create the test infrastructure.

```bash
# Install pytest
uv add --dev pytest pytest-xdist

# Create test structure
mkdir -p tests/backend tests/tensor tests/nn tests/integration
touch tests/conftest.py
touch tests/__init__.py
```

**`tests/conftest.py`:**
```python
import pytest
import numpy as np

@pytest.fixture
def skip_no_cuda():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

@pytest.fixture
def random_seed():
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
```

---

### Assignment 6.2: Write Backend Tests ⭐⭐

**Task:** Test every component of the backend.

**`tests/backend/test_view.py`:**
```python
from banhxeo.core.view import View

class TestViewBasics:
    def test_create_contiguous(self):
        v = View.create((2, 3))
        assert v.shape == (2, 3)
        assert v.strides == (3, 1)
        assert v.offset == 0
        assert v.is_contiguous()
    
    def test_permute(self):
        v = View.create((2, 3))
        v2 = v.permute((1, 0))
        assert v2.shape == (3, 2)
        assert v2.strides == (1, 3)
        assert not v2.is_contiguous()
    
    # Add more tests for slice, reshape, broadcast_to, etc.
```

**`tests/backend/test_codegen.py`:**
```python
from banhxeo import Tensor
import numpy as np

class TestTritonCodegen:
    def test_add(self, skip_no_cuda):
        a = Tensor([1, 2, 3], device="cuda")
        b = Tensor([4, 5, 6], device="cuda")
        c = (a + b).numpy()
        assert np.allclose(c, [5, 7, 9])
    
    def test_view_chain(self, skip_no_cuda):
        x = Tensor.rand(2, 3, 4, device="cuda")
        y = x.permute(2, 0, 1)  # (4, 2, 3)
        z = y.reshape(4, 6)     # (4, 6) - may need contiguous
        z.realize()
        assert z.shape == (4, 6)
    
    # Test all the edge cases from Module 1
```

---

### Assignment 6.3: Property-Based Testing ⭐⭐⭐

**Advanced:** Use hypothesis for property-based testing.

```python
from hypothesis import given, strategies as st

@given(st.lists(st.integers(1, 10), min_size=1, max_size=4))
def test_view_roundtrip(shape):
    """Test that various view operations maintain data integrity."""
    shape = tuple(shape)
    v = View.create(shape)
    
    # Property: is_contiguous should be True for freshly created views
    assert v.is_contiguous()
    
    # Property: reshape to same shape should give identical view
    v2 = v.reshape(shape)
    assert v2.shape == shape
    assert v2.strides == v.strides
```

---

## Module 7: Memory Management (Advanced)

### Assignment 7.1: Implement Buffer Pooling ⭐⭐

**Problem:** Every op allocates a new buffer. Wasteful for temporary results.

```python
# memory.py (new file)
class BufferPool:
    def __init__(self):
        self.free_buffers: Dict[Tuple[Tuple[int,...], str], List[torch.Tensor]] = {}
    
    def allocate(self, shape, dtype, device) -> torch.Tensor:
        """Get a buffer from pool, or allocate new if none available."""
        # YOUR IMPLEMENTATION
    
    def free(self, buffer: torch.Tensor):
        """Return buffer to pool for reuse."""
        # YOUR IMPLEMENTATION
```

**Integration:** Modify `RawBuffer.create()` to use the pool.

**tinygrad reference:**
- `tinygrad/device.py` → `LRUAllocator`
- Study how tinygrad tracks buffer lifetimes

---

### Assignment 7.2: Analyze Buffer Lifetimes ⭐⭐⭐

**Task:** Determine when buffers can be reused.

```python
x = Tensor.rand(1000, 1000)
y = x + 1        # temp_0 = x + 1
z = y * 2        # temp_1 = temp_0 * 2, temp_0 can be freed!
w = z - 1        # temp_2 = temp_1 - 1, temp_1 can be freed!
w.realize()
```

**Question:** At each step, which buffers are "live" (still needed)?

**tinygrad reference:**
- `tinygrad/engine/schedule.py` → `memory_planner()`
- Study how tinygrad tracks `uop_refcount`

---

## Final Projects

### Project A: MNIST MLP ⭐⭐

**Goal:** Train a 2-layer MLP on MNIST to >90% accuracy.

**Requirements:**
- `nn.Linear` layer
- `SGD` optimizer
- Cross-entropy loss (already in `functional.py`, but uses numpy hack)

**Steps:**
1. Implement `nn.Linear` (Assignment in Module N1)
2. Implement `SGD` optimizer
3. Load MNIST data
4. Training loop
5. Verify convergence

---

### Project B: MNIST CNN ⭐⭐⭐

**Goal:** Train a CNN on MNIST to >98% accuracy.

**Requirements (builds on Project A):**
- `View.pad()` and `View.shrink()` working
- `nn.Conv2d` via im2col
- `nn.MaxPool2d`

**This is significantly harder** because Conv2d requires:
- PAD for same-padding
- Sliding window extraction (im2col)
- Batched matmul for the actual convolution
- SHRINK for output sizing

---

### Project C: Kernel Visualization ⭐⭐

**Goal:** Build a visualization tool for generated kernels.

Show:
- The LazyBuffer DAG
- Fusion boundaries
- Generated Triton code with annotations
- Memory access patterns

Extend `utils/viz.py` with richer visualization.

---

## Appendix A: tinygrad Code Map

### Essential Files

| File | Purpose | Read When |
|------|---------|-----------|
| `CLAUDE.md` | Architecture overview | First! |
| `tinygrad/tensor.py` | User API | Understanding API design |
| `tinygrad/uop/ops.py` | UOp IR | Building your IR |
| `tinygrad/uop/__init__.py` | Op definitions | Adding new ops |
| `tinygrad/shape/view.py` | View implementation | Module 1 |
| `tinygrad/shape/shapetracker.py` | View composition | Module 1 |
| `tinygrad/schedule/indexing.py` | Movement lowering | Module 1 |
| `tinygrad/engine/schedule.py` | Kernel creation | Module 3 |
| `tinygrad/schedule/rangeify.py` | Fusion logic | Module 3 |
| `tinygrad/renderer/cstyle.py` | Code generation | Module 2 |
| `tinygrad/gradient.py` | Autograd rules | Module 5 |

### Study Order

1. `CLAUDE.md` - High-level architecture
2. Trace `Tensor.realize()` through the codebase
3. Study `UOp` class in `uop/ops.py`
4. Understand `create_schedule()` in `engine/schedule.py`
5. Read `CStyleLanguage.render()` in `renderer/cstyle.py`

---

## Appendix B: Debugging Tips

### Environment Variables

```bash
DEBUG=1 python script.py  # Print generated kernels
DEBUG=2 python script.py  # + Schedule visualization
DEBUG=3 python script.py  # + Kernel timing
DEBUG=4 python script.py  # + Buffer values
```

### Common Issues

**"Cannot reshape non-contiguous view"**
- Solution: Add `.contiguous()` before reshape
- Root cause: Non-contiguous strides can't always be reshaped in-place

**Kernel produces wrong values**
- Check indexing math in generated code
- Verify stride computation for non-contiguous inputs
- Add `DEBUG=1` to see generated kernel

**Kernel fails to compile**
- Check generated code syntax
- Ensure all variables are defined before use
- Look for type mismatches

---

## Appendix C: Checklist by Module

### Module 1: View & Indexing
- [ ] 1.1 Implement `View.to_index_expr()`
- [ ] 1.2 Optimize for contiguous case
- [ ] 1.3 Handle view composition correctly
- [ ] 1.4 Implement `View.pad()` with mask
- [ ] 1.5 Implement `View.shrink()`
- [ ] 1.6 Implement `View.flip()` with negative strides

### Module 2: Codegen Architecture
- [ ] 2.1 Analyze current codegen problems
- [ ] 2.2 Design KernelOp IR
- [ ] 2.3 Implement IRBuilder
- [ ] 2.4 Implement TritonRenderer
- [ ] 2.5 Add constant folding pass

### Module 3: Fusion & Scheduling
- [ ] 3.1 Formalize barrier rules
- [ ] 3.2 Implement fusion analysis
- [ ] 3.3 Add fusion logging

### Module 4: Specialized Kernels
- [ ] 4.1 Annotate matmul kernel
- [ ] 4.2 Implement batched matmul
- [ ] 4.3 Implement multi-axis reduce
- [ ] 4.4 Tree reduction (optional)

### Module 5: Autograd
- [ ] 5.1 Gradient verification tests
- [ ] 5.2 Movement op gradients
- [ ] 5.3 Broadcasting gradient accumulation

### Module 6: Testing
- [ ] 6.1 Set up pytest
- [ ] 6.2 Backend tests
- [ ] 6.3 Property-based tests (optional)

---

Good luck! Remember: the best way to learn is to implement, get stuck, then discover why the proper solution works better.
