## Module 7: Memory Management (Advanced)

Memory management comes after correctness, scheduling, and IR.

## Why This Module Is Advanced

Buffer reuse is not hard because allocation APIs are hard. It is hard because
you must know when a buffer is dead.

That depends on scheduling, fusion boundaries, realized buffers, and graph
ownership. If those are unclear, a buffer pool will create spooky correctness
bugs: reused memory that still has live readers.

In this module, prefer analysis before implementation.

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

**Why this assignment:** Buffer pooling teaches allocation policy. It should be
small and boring; the real lesson is knowing when it is legal to reuse memory.

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

**Why this assignment:** Lifetime analysis is where the compiler stops being
only about codegen and starts managing resources.

---
