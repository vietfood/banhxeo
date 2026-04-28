## Module 3: Scheduling & Kernel Fusion

This module teaches you how to decide which operations become which kernels. This is one of the most important parts of a tensor compiler - good fusion decisions can make your code 10x faster.

### Conceptual Foundation: Why Scheduling Matters

**The Problem:**

When you write `(a + b) * c`, you could execute it two ways:

```
Approach 1: Separate kernels (naive)
┌─────────────────┐     ┌─────────────────┐
│ Kernel 1: a + b │ ──► │ Kernel 2: * c   │
│ Read: a, b      │     │ Read: temp, c   │
│ Write: temp     │     │ Write: output   │
└─────────────────┘     └─────────────────┘
Memory traffic: 5 buffers read/written

Approach 2: Fused kernel (optimal)
┌─────────────────────────────┐
│ Kernel: (a + b) * c         │
│ Read: a, b, c               │
│ Write: output               │
│ temp stays in registers!    │
└─────────────────────────────┘
Memory traffic: 4 buffers read/written
```

**Key Insight:** Memory bandwidth is the bottleneck. Fusion keeps intermediate values in registers, avoiding expensive memory round-trips.

**But not everything can be fused:**
- Reductions require synchronization across threads
- Matmul uses specialized tiled algorithms
- Some ops have incompatible parallelization strategies

The scheduler's job is to find the optimal partition of operations into kernels.

---

### Assignment 3.1: Understand banhxeo's Current Scheduling ⭐⭐

**Task:** Trace through banhxeo's scheduling to understand how it works.

**Key code locations:**

| Function | File:Line | Purpose |
|----------|-----------|---------|
| `schedule()` | `backend/__init__.py:15-35` | Topological sort of LazyBuffer DAG |
| `is_barrier()` | `backend/__init__.py:107-124` | Determines kernel boundaries |
| `get_barriers()` | `backend/__init__.py:126-151` | Finds all barrier dependencies |
| `exec()` | `backend/__init__.py:302-327` | Main execution loop |

**Trace this example:**
```python
x = Tensor.rand(10, 10, device="cuda")
y = x.sum(axis=1)    # Barrier: reduction
z = y + 1            # Elementwise
z.realize()
```

**Questions to answer:**
1. What does `schedule()` return? Draw the topological order.
2. What does `get_barriers(z.lazydata)` return?
3. In what order are kernels launched?

**Current flow:**
```
exec(z) is called
    │
    ▼
get_barriers(z) returns {y}  ← y is a reduce, must be done first
    │
    ▼
exec(y) is called recursively  ← Kernel 1: reduction
    │
    ▼
exec_elementwise(z)            ← Kernel 2: z = y + 1
```

---

### Assignment 3.2: Formalize Barrier Rules ⭐⭐

**Current Problem:** The barrier logic in `backend/__init__.py:107-124` is implicit and undocumented:

```python
def is_barrier(self, buf: LazyBuffer):
    if buf.realized is not None:
        return True
    if isinstance(buf.op, LoadOp):
        if len(buf.src) != 0 and buf.src[0].realized is None:
            return True
    return isinstance(buf.op, ReduceOp) or buf.op == BinaryOp.MATMUL
```

**Why are these barriers?**

| Condition | Why it's a barrier |
|-----------|-------------------|
| `buf.realized is not None` | Already computed, just load it |
| `ReduceOp` | Requires thread synchronization (not embarrassingly parallel) |
| `BinaryOp.MATMUL` | Uses specialized tiled kernel with different parallelization |
| `LoadOp with unrealized src` | VIEW/CONTIGUOUS of a computed value |

**Your Task:** Refactor into explicit, documented rules:

```python
# scheduling.py (new file in src/banhxeo/backend/)

from banhxeo.core.buffer import BinaryOp, LazyBuffer, LoadOp, ReduceOp

# Operations that force kernel boundaries
# Rationale for each:
BARRIER_OPS = {
    # HINT: Add ReduceOp values (SUM, MAX)
    # HINT: Add BinaryOp.MATMUL
    # YOUR CODE HERE
}

def is_barrier_op(op) -> bool:
    """Check if an operation type forces a kernel boundary."""
    # HINT: Just check if op is in BARRIER_OPS
    # YOUR CODE HERE
    pass

def is_barrier(buf: LazyBuffer) -> bool:
    """
    Check if a LazyBuffer forces a kernel boundary.
    
    A barrier means this buffer must be fully computed before
    any dependent operations can run.
    
    Returns True if:
    1. Buffer is already realized (computed)
    2. Buffer is a reduction or matmul (specialized kernel)
    3. Buffer is a VIEW/CONTIGUOUS of an unrealized compute op
    """
    # HINT: Check if buf.realized is not None
    # HINT: Check if is_barrier_op(buf.op) returns True
    # HINT: Check if buf.op is LoadOp.VIEW or LoadOp.CONTIGUOUS
    #       and has unrealized source
    # YOUR CODE HERE
    pass

def get_kernel_groups(output: LazyBuffer) -> list:
    """
    Partition LazyBuffer DAG into kernel groups.
    
    Each group contains buffers that can be fused into a single kernel.
    """
    # YOUR IMPLEMENTATION
    pass
```

**Test your implementation:**
```python
# Test 1: Pure elementwise - should NOT be barriers
a = LazyBuffer(LoadOp.FROM_PYTHON, View.create((3,)), args=[[1,2,3]])
b = LazyBuffer(LoadOp.FROM_PYTHON, View.create((3,)), args=[[4,5,6]])
add = a.compute_ops(BinaryOp.ADD, b)
assert not is_barrier(add)  # Elementwise, not a barrier

# Test 2: Reduce - should be barrier
x = LazyBuffer(LoadOp.RAND, View.create((10,10)), args=[42])
reduced = x.reduce_ops(ReduceOp.SUM, (10,))
assert is_barrier(reduced)  # Reduction IS a barrier

# Test 3: Matmul - should be barrier
matmul_result = a.matmul(b)
assert is_barrier(matmul_result)  # Matmul IS a barrier
```

---

### Assignment 3.3: Understand tinygrad's Scheduling ⭐⭐⭐

Before implementing fusion analysis, study how tinygrad does it.

**tinygrad's approach:**

1. **UOp Graph Construction**: All operations become UOp nodes in a DAG
2. **Rangeify**: Convert tensor operations to explicit loop structures
3. **Kernel Splitting**: Operations in same "outer range" can fuse
4. **Topological Sort**: Determine execution order
5. **Memory Planning**: Optimize buffer allocation

**Key concept: "Outer Range" compatibility**

```python
# These can fuse (same iteration pattern):
for i in range(N):
    a[i] = x[i] + y[i]     # Same loop bounds
    b[i] = a[i] * 2        # Same loop bounds

# These CANNOT fuse (different iteration patterns):
for i in range(N):         # Kernel 1: reduction
    for j in range(M):
        acc += x[i,j]
# barrier here
for i in range(N):         # Kernel 2: post-process
    result[i] = acc[i] + 1
```

**tinygrad files to study:**

| File | Function | What it does |
|------|----------|--------------|
| `tinygrad/engine/schedule.py` | `create_schedule()` | Main scheduling entry point |
| `tinygrad/schedule/rangeify.py` | `split_kernels()` | Determines kernel boundaries |
| `tinygrad/schedule/rangeify.py` | `split_store()` | Actually splits at barriers |

**Task:** Answer these questions by reading tinygrad source:

1. How does tinygrad represent kernel boundaries?
2. What conditions trigger `split_store()`?
3. How does tinygrad handle VIEW operations in scheduling?

---

### Assignment 3.4: Implement Fusion Analysis ⭐⭐⭐

**Task:** Implement a function that partitions operations into fusible groups.

```python
# scheduling.py (continued)

from typing import List, Set, Dict
from collections import defaultdict

def analyze_fusion(output: LazyBuffer) -> List[Set[LazyBuffer]]:
    """
    Partition the LazyBuffer DAG into fusible groups.
    
    Each group contains LazyBuffers that can be compiled into a single kernel.
    Groups are returned in topological order (dependencies first).
    
    Algorithm:
    1. Traverse DAG from output backwards
    2. When hitting a barrier, start a new group
    3. Non-barriers inherit their parent's group
    4. Return groups in execution order
    
    Returns:
        List of sets, where each set = one kernel's operations
    """
    groups: List[Set[LazyBuffer]] = []
    buf_to_group: Dict[LazyBuffer, int] = {}
    visited: Set[LazyBuffer] = set()
    
    def assign_group(buf: LazyBuffer, current_group: int) -> int:
        """
        Assign buf to a group. Returns the group ID for children to use.
        """
        # HINT: Check if already visited - if so, return its group
        # HINT: Mark as visited
        # HINT: If is_barrier(buf) and not realized:
        #       - Create new group
        #       - Add buf to that group
        #       - Recursively process sources with next group index
        # HINT: If not a barrier:
        #       - Ensure current_group exists in groups list
        #       - Add buf to current group
        #       - Recursively process sources in same group
        # HINT: Store buf's group in buf_to_group
        # YOUR CODE HERE
        pass
    
    # HINT: Start from output with group 0
    # HINT: Reverse groups list to get execution order
    # YOUR CODE HERE
    pass

def visualize_fusion(output: LazyBuffer):
    """Print a visualization of fusion decisions."""
    groups = analyze_fusion(output)
    
    print(f"\n{'='*50}")
    print(f"FUSION ANALYSIS: {len(groups)} kernel(s)")
    print(f"{'='*50}")
    
    for i, group in enumerate(groups):
        print(f"\nKernel {i}:")
        for buf in group:
            barrier_str = " [BARRIER]" if is_barrier(buf) else ""
            print(f"  - {buf.op}{barrier_str} shape={buf.shape}")
    
    print(f"{'='*50}\n")
```

**Test your implementation:**

```python
# Test 1: Pure elementwise chain - 1 kernel
a = Tensor.rand(10, device="cuda")
b = Tensor.rand(10, device="cuda")
c = (a + b) * 2 - 1
groups = analyze_fusion(c.lazydata)
assert len(groups) == 1, f"Expected 1 kernel, got {len(groups)}"

# Test 2: Reduce then elementwise - 2 kernels
x = Tensor.rand(10, 10, device="cuda")
y = x.sum(axis=1) + 1
groups = analyze_fusion(y.lazydata)
assert len(groups) == 2, f"Expected 2 kernels, got {len(groups)}"

# Test 3: Matmul then elementwise - 2 kernels
a = Tensor.rand(10, 10, device="cuda")
b = Tensor.rand(10, 10, device="cuda")
c = (a @ b) * 2
groups = analyze_fusion(c.lazydata)
assert len(groups) == 2, f"Expected 2 kernels, got {len(groups)}"

# Test 4: Complex chain
x = Tensor.rand(10, 10, device="cuda")
y = x + 1           # Fused with next
z = y.sum(axis=1)   # Barrier
w = z * 2           # New kernel
result = w - 1      # Fused with previous
groups = analyze_fusion(result.lazydata)
# x+1 can fuse with input to reduce, then z*2-1 fuses
# Actual number depends on how you handle reduce inputs
print(f"Got {len(groups)} kernels")
```

---

### Assignment 3.5: Add Fusion Logging ⭐

**Task:** Add debug output showing fusion decisions.

```python
# In backend/__init__.py, modify gencode()

def gencode(self, output: LazyBuffer):
    linear_graph = self.schedule(output)
    
    if DEBUG >= 2:
        # Import your new scheduling module
        from banhxeo.backend.scheduling import analyze_fusion, visualize_fusion
        
        print(f"\n[FUSION] Building kernel for: {output.op}")
        
        # Count op types
        op_counts = defaultdict(int)
        for buf in linear_graph:
            op_counts[type(buf.op).__name__] += 1
        
        print(f"[FUSION]   Ops in schedule: {dict(op_counts)}")
        print(f"[FUSION]   Total: {len(linear_graph)} operations")
        
        # Show barriers
        barriers = [b for b in linear_graph if is_barrier(b)]
        if barriers:
            print(f"[FUSION]   Barriers: {[str(b.op) for b in barriers]}")
        
        from banhxeo.utils.viz import visualize_schedule_cli
        visualize_schedule_cli(linear_graph)
    
    # ... rest of gencode
```

**Expected output:**
```
[FUSION] Building kernel for: BinaryOp.MUL
[FUSION]   Ops in schedule: {'LoadOp': 2, 'BinaryOp': 3}
[FUSION]   Total: 5 operations
[FUSION]   Barriers: []

╭─ Schedule ─╮
│ in_0 (RAND)
│ in_1 (RAND)
│ temp_0 = in_0 + in_1
│ temp_1 = temp_0 * 2
│ temp_2 = temp_1 - 1
╰────────────╯
```

---

### Assignment 3.6: Understand Advanced Fusion (Research) ⭐⭐⭐

**Beyond basic barriers:** Real-world schedulers consider more factors.

**Shape Compatibility:**
```python
# Can fuse (same shape):
a = Tensor.rand(10, 20)
b = a + 1       # shape (10, 20)
c = b * 2       # shape (10, 20) - fusible!

# Cannot fuse (shape change):
a = Tensor.rand(10, 20)
b = a.sum(axis=1)  # shape (10,) - barrier!
c = b + 1          # shape (10,) - new kernel
```

**Memory Pressure:**
```python
# Might NOT want to fuse (too many live values):
a = Tensor.rand(1000, 1000)
b = a + 1
c = a + 2  # Both b and c need 'a' - that's 3 buffers in registers
d = b + c  # Might exceed register pressure
```

**Device Constraints:**
```python
# Cannot fuse across devices:
a = Tensor.rand(10, device="cuda:0")
b = a.to("cuda:1")  # Barrier - device transfer
c = b + 1           # Runs on cuda:1
```

**Task:** Research how tinygrad handles these cases:
1. How does tinygrad track "live values" in a kernel?
2. How does the `memory_planner()` work?
3. What is the `LRUAllocator` and how does it reduce allocation overhead?

**tinygrad files:**
- `tinygrad/engine/schedule.py` → `memory_planner()`
- `tinygrad/device.py` → `LRUAllocator`
- `tinygrad/uop/ops.py` → UOp reference counting

**Deliverable:** Write a 1-page analysis of tinygrad's advanced scheduling features and which ones would be valuable for banhxeo.

---
