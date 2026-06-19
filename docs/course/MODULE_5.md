## Module 5: Basic Kernel IR And Codegen

This module teaches you how to build a proper compiler IR. Understanding IRs is fundamental to building any serious compiler - they're the bridge between "what the user wants" and "what the hardware executes."

## Why This Module Comes After Scheduling

An IR is not magic. It is just a structured representation of decisions you have
already made.

Do this module only after:

1. view/indexing behavior is centralized in `View`
2. forward tests catch basic semantic bugs
3. scheduling can explain which operations belong to one kernel

Reason: the first banhxeo IR should represent **one fused elementwise kernel**.
If you do not know the kernel boundary yet, the IR design will drift toward a
fake tinygrad clone. If indexing is still wrong, the IR will preserve wrong
loads and stores with nicer names.

The goal is not to invent the final compiler representation. The goal is to
replace string-concatenation codegen with a small structure that is inspectable,
testable, and disposable.

### Conceptual Foundation: What is an IR?

**The Problem with Direct Code Generation:**

banhxeo currently generates Triton code by concatenating strings in `TritonCodegen`. This is like writing a book by gluing letters together - it works, but it's fragile and hard to improve.

```
User Code → LazyBuffer DAG → String Concatenation → Triton Source
                                    ↑
                              (hard to optimize,
                               hard to add backends,
                               hard to debug)
```

**The Solution - Intermediate Representation (IR):**

An IR is a structured, machine-readable format that represents computation. It sits between the high-level user code and low-level machine code:

```
User Code → LazyBuffer DAG → IR → Optimization Passes → IR → Renderer → Triton/CUDA/Metal
                              ↑                          ↑
                        (structured,             (optimized,
                         analyzable)              simplified)
```

**Why IRs Matter:**
1. **Separation of concerns**: "What to compute" is separate from "how to render it"
2. **Optimization**: Transform the IR before rendering (constant folding, dead code elimination)
3. **Multi-backend**: Same IR can render to Triton, CUDA C, Metal, etc.
4. **Testability**: Each stage can be tested independently
5. **Debugging**: Inspect the IR to understand what's happening

---

### Assignment 5.1: Analyze Current Codegen Problems ⭐⭐

**Task:** Study `triton.py:15-303` and document all the problems.

**Questions to answer:**
1. What happens if you want to add a new unary op like `tanh`?
2. How many places need changes to add a new backend (CUDA C)?
3. Where would you insert constant folding optimization?
4. Why is the indexing logic duplicated (lines 59-96 vs 146-169)?

**Problems to identify:**
```python
# Problem 1: No separation between IR and rendering
def visit_BinaryOp(self, buf, name):
    # This mixes "what op" with "how to render"
    self.code.append(f"    {name} = {src0} + {src1}")  # String!

# Problem 2: Op-specific logic scattered everywhere
# Adding TANH requires changes in:
# - UnaryOp enum in buffer.py
# - visit_UnaryOp in triton.py
# - op_map dictionary
# - Maybe torch.py backend too

# Problem 3: No optimization possible
# Once you generate "x = 2.0 + 3.0", you can't fold it to "x = 5.0"
```

**Deliverable:** Write a 1-page document listing problems and how an IR would solve each.

**Why this assignment:** Before adding an abstraction, prove the current code is
hurting you in a specific way. "IRs are good" is not enough.

---

### Assignment 5.2: Understand tinygrad's UOp Design ⭐⭐⭐

Before designing your own IR, study how tinygrad does it. This assignment is research-focused.

**Key Concepts in tinygrad's UOp:**

1. **UOp is a DAG node**, not a list item:
```python
# tinygrad's approach (simplified)
@dataclass(frozen=True)  # Immutable!
class UOp:
    op: Ops           # What operation (ADD, MUL, LOAD, etc.)
    dtype: DType      # Data type of result
    src: Tuple[UOp, ...]  # Input UOps (parents in DAG)
    arg: Any          # Operation-specific argument
```

2. **UOps are deduplicated** - same inputs = same object:
```python
# Creating ADD(x, y) twice returns the SAME object
add1 = UOp(Ops.ADD, float32, (x, y))
add2 = UOp(Ops.ADD, float32, (x, y))
assert add1 is add2  # Same object! (via __new__ caching)
```

3. **Why immutability and deduplication?**
   - Graph rewrites become safe (no aliasing bugs)
   - Pattern matching is fast (object identity checks)
   - Memory efficient (no duplicate nodes)
   - Enables caching of compiled kernels

**Task:** Answer these questions by reading tinygrad source:

| Question | Where to Look |
|----------|---------------|
| How is UOp's `__new__` implemented for caching? | `tinygrad/uop/ops.py` |
| What operations exist in the `Ops` enum? | `tinygrad/uop/__init__.py` |
| How does `UOp.replace()` work with immutability? | `tinygrad/uop/ops.py` |
| What is `UPat` and how does pattern matching work? | `tinygrad/uop/ops.py` |

**Key Insight:** tinygrad's UOp is more than just IR - it's a unified representation for *everything* (tensor ops, control flow, memory, indexing). This is powerful but complex.

---

### Assignment 5.3: Design Your KernelOp IR ⭐⭐

**Task:** Design a simpler IR for banhxeo, focused on kernel code generation.

**Design Goals:**
1. Represent all operations needed for elementwise kernels
2. Support optimization passes
3. Easy to render to Triton (and later, other backends)

**Recommended Structure:**

```python
# ir.py (new file in src/banhxeo/backend/)

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, Union

class OpType(Enum):
    # Memory operations
    LOAD = auto()      # Load from pointer
    STORE = auto()     # Store to pointer
    CONST = auto()     # Constant value

    # Arithmetic (binary)
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MAX = auto()
    CMPLT = auto()

    # Arithmetic (unary)
    NEG = auto()
    EXP = auto()
    LOG = auto()
    SIN = auto()
    SQRT = auto()

    # Control flow
    WHERE = auto()     # Ternary select

    # Indexing
    INDEX = auto()     # Compute memory offset

@dataclass(frozen=True)  # Immutable!
class KernelOp:
    """A single operation in the kernel IR."""
    op: OpType
    name: str                          # Result variable name
    dtype: str                         # "float32", "int32", etc.
    args: Tuple[Union[str, float, int], ...]  # Inputs (var names or constants)

    def __post_init__(self):
        # HINT: Validate arity here
        # - Binary ops need 2 args
        # - Unary ops need 1 arg
        # - WHERE needs 3 args (condition, true_val, false_val)
        # YOUR CODE HERE
        pass
```

**Questions to answer in your design:**
1. How do you represent `tl.load(ptr + offset, mask=mask)`?
2. How do you handle ops with different arities (unary vs binary)?
3. Should the IR be a list or a DAG? (Hint: list is simpler to start)

**Deliverable:** Complete `ir.py` with your KernelOp design and unit tests.

**Why this assignment:** A list IR is deliberately less powerful than tinygrad's
UOp DAG. That constraint keeps the lesson focused on lowering one kernel before
you learn graph rewriting.

---

### Assignment 5.4: Implement IRBuilder ⭐⭐

**Task:** Create a builder that converts LazyBuffer DAG to IR.

```python
# ir.py (continued)

class IRBuilder:
    """Builds IR from LazyBuffer operations."""

    def __init__(self):
        self.ops: List[KernelOp] = []
        self.var_counter = 0
        self.var_map: Dict[LazyBuffer, str] = {}  # Track variable names

    def new_var(self, prefix: str = "v") -> str:
        """Generate a unique variable name."""
        # HINT: Use self.var_counter to generate unique names like "v_0", "v_1", etc.
        # Don't forget to increment the counter!
        # YOUR CODE HERE
        pass

    def emit_load(self, ptr: str, offset: str, mask: Optional[str] = None) -> str:
        """Emit a load and return the result variable name."""
        # HINT: Create a new variable name
        # HINT: Create a KernelOp with OpType.LOAD
        # HINT: Args should be (ptr, offset, mask) or (ptr, offset) if mask is None
        # HINT: Append the op to self.ops
        # HINT: Return the variable name
        # YOUR CODE HERE
        pass

    def emit_binary(self, op: OpType, left: str, right: str) -> str:
        """Emit a binary operation and return result variable name."""
        # HINT: Similar to emit_load but for binary ops
        # HINT: Args are (left, right)
        # YOUR CODE HERE
        pass

    def emit_unary(self, op: OpType, src: str) -> str:
        """Emit a unary operation and return result variable name."""
        # HINT: Similar to emit_binary but only one arg
        # YOUR CODE HERE
        pass

    def emit_store(self, ptr: str, offset: str, value: str, mask: Optional[str] = None):
        """Emit a store operation."""
        # HINT: Similar to emit_load but OpType.STORE
        # HINT: Args are (ptr, offset, value, mask) or (ptr, offset, value)
        # YOUR CODE HERE
        pass

    def build_from_schedule(self, schedule: List[LazyBuffer]) -> List[KernelOp]:
        """Convert a LazyBuffer schedule to IR."""
        for buf in schedule:
            self._visit(buf)
        return self.ops

    def _visit(self, buf: LazyBuffer) -> str:
        """Visit a LazyBuffer and return its variable name."""
        # HINT: Check if already visited using self.var_map
        if buf in self.var_map:
            return self.var_map[buf]

        # HINT: Handle different op types
        # HINT: For BinaryOp - recursively visit left and right, then emit_binary
        # HINT: For UnaryOp - recursively visit source, then emit_unary
        # HINT: Use op_map dictionaries to convert LazyBuffer ops to OpType
        # HINT: Store result in self.var_map[buf] and return it
        # YOUR CODE HERE
        pass
```

**Test your builder:**
```python
# Build IR for: (a + b) * 2
a = LazyBuffer(LoadOp.FROM_PYTHON, View.create((3,)), args=[[1,2,3]])
b = LazyBuffer(LoadOp.FROM_PYTHON, View.create((3,)), args=[[4,5,6]])
add = a.compute_ops(BinaryOp.ADD, b)
const = add.const(2.0)
mul = add.compute_ops(BinaryOp.MUL, const)

builder = IRBuilder()
schedule = [a, b, add, const, mul]  # Simplified
ir = builder.build_from_schedule(schedule)

# Verify IR contains: LOAD, LOAD, ADD, CONST, MUL
```

---

### Assignment 5.5: Implement Triton Renderer ⭐⭐

**Task:** Convert IR to Triton source code.

```python
# renderer.py (new file)

class TritonRenderer:
    """Renders IR to Triton kernel source code."""

    # Map IR ops to Triton syntax
    BINARY_OPS = {
        OpType.ADD: "{a} + {b}",
        OpType.SUB: "{a} - {b}",
        OpType.MUL: "{a} * {b}",
        OpType.DIV: "{a} / {b}",
        OpType.MAX: "tl.maximum({a}, {b})",
        OpType.CMPLT: "{a} < {b}",
    }

    UNARY_OPS = {
        OpType.NEG: "-{a}",
        OpType.EXP: "tl.exp({a})",
        OpType.LOG: "tl.log({a})",
        OpType.SIN: "tl.sin({a})",
        OpType.SQRT: "tl.sqrt({a})",
    }

    def render_op(self, op: KernelOp) -> str:
        """Render a single IR op to Triton code."""
        # HINT: Check op.op type and use appropriate dictionary
        # HINT: For BINARY_OPS, format with a=op.args[0], b=op.args[1]
        # HINT: For UNARY_OPS, format with a=op.args[0]
        # HINT: For LOAD, render as "tl.load(ptr + offset, mask=mask)"
        # HINT: For STORE, render as "tl.store(ptr + offset, value, mask=mask)"
        # HINT: For CONST, render as "name = value"
        # HINT: For WHERE, render as "tl.where(cond, true_val, false_val)"
        # HINT: Return formatted string like "    {op.name} = {expression}"
        # YOUR CODE HERE
        pass

    def render(self, ops: List[KernelOp], input_ptrs: List[str],
               output_ptr: str, N: int) -> str:
        """Generate complete Triton kernel source."""

        # HINT: Build kernel signature with all pointers and BLOCK_SIZE
        # HINT: Add @triton.jit decorator
        # HINT: Add boilerplate: pid, offsets, linear_mask
        # HINT: Loop through ops and call render_op() for each
        # HINT: Join all lines with newlines
        # YOUR CODE HERE
        pass
```

**Verification:** Render your IR, compile with `compile_triton_src()`, run on GPU, check correctness.

**Why this assignment:** Rendering is where you learn whether the IR actually
separated "what to compute" from "how Triton spells it." If your renderer needs
to inspect random `LazyBuffer` details, the IR boundary is too weak.

---

### Assignment 5.6: Implement Optimization Passes ⭐⭐⭐

**Task:** Implement two optimization passes on your IR.

**Pass 1: Constant Folding**

```python
def constant_fold(ops: List[KernelOp]) -> List[KernelOp]:
    """Fold operations where all inputs are constants."""
    constants: Dict[str, float] = {}  # var_name -> value
    new_ops: List[KernelOp] = []

    for op in ops:
        # HINT: If op is CONST, add to constants dict and keep the op
        # HINT: Otherwise, check if all string args are in constants
        # HINT: If yes, compute the result based on op type
        # HINT: Replace with a CONST op containing the computed result
        # HINT: If no, keep the original op
        # YOUR CODE HERE
        pass

    return new_ops
```

**Pass 2: Dead Code Elimination**

```python
def eliminate_dead_code(ops: List[KernelOp], output_var: str) -> List[KernelOp]:
    """Remove ops whose results are never used."""
    # HINT: Start with a set containing just output_var
    # HINT: Work backwards through ops
    # HINT: If op.name is in used set, add all its args to used set
    # HINT: Keep only ops whose names are in used set (or STORE ops)
    # YOUR CODE HERE
    pass
```

**Test your passes:**
```python
# Before optimization:
# v0 = 2.0
# v1 = 3.0
# v2 = v0 + v1    # Can fold to 5.0
# v3 = x * v2

# After constant folding:
# v0 = 2.0
# v1 = 3.0
# v2 = 5.0        # Folded!
# v3 = x * v2

# After dead code elimination:
# v2 = 5.0        # v0, v1 removed (unused)
# v3 = x * v2
```

**tinygrad reference:**
- `tinygrad/uop/ops.py` → `graph_rewrite()` function
- Study the `PatternMatcher` class for declarative rewrite rules
- Key insight: tinygrad expresses optimizations as pattern → replacement rules

**Further exploration:** Implement Common Subexpression Elimination (CSE):
```python
# Before CSE:
# v0 = a + b
# v1 = a + b  # Duplicate!
# v2 = v0 * v1

# After CSE:
# v0 = a + b
# v2 = v0 * v0  # Reuse v0
```

Do not implement CSE until constant folding and dead code elimination are boring.
If basic passes are still confusing, CSE will hide the confusion under cleverness.

---
