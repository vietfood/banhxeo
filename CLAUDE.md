# banhxeo - AI Agent Guide

This repository contains **banhxeo**, a minimalist educational deep learning framework. It implements lazy evaluation and Triton kernel generation from scratch.

This project uses `uv` for dependency management.

## 🎨 Code Style & Conventions

Adhere strictly to the following conventions to maintain the "minimalist and educational" philosophy.

### 1. General Philosophy
- **Tiny & Readable:** 
  - Keep code concise. Prefer simple implementations over complex optimizations unless necessary for the educational goal.
  - Every line must earn its keep. Prefer readability over cleverness. We believe that if carefully designed, 10 lines can have the impact of 1000. Never mix functionality changes with whitespace changes. All functionality changes must be tested.
- **Minimum code that solves the problem. Nothing speculative.**
  - No features beyond what was asked.
  - No abstractions for single-use code.
  - No "flexibility" or "configurability" that wasn't requested.
  - No error handling for impossible scenarios.
  - If you write 200 lines and it could be 50, rewrite it.
  Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.
- **Don't assume. Don't hide confusion. Surface tradeoffs.** Before implementing:
  - State your assumptions explicitly. If uncertain, ask.
  - If multiple interpretations exist, present them - don't pick silently.
  - If a simpler approach exists, say so. Push back when warranted.
  - If something is unclear, stop. Name what's confusing. Ask.
- **Touch only what you must. Clean up only your own mess.** When editing existing code:
  - Don't "improve" adjacent code, comments, or formatting.
  - Don't refactor things that aren't broken.
  - Match existing style, even if you'd do it differently.
  - If you notice unrelated dead code, mention it - don't delete it.
  - When your changes create orphans:
    - Remove imports/variables/functions that YOUR changes made unused.
    - Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 2. Formatting & Imports
- **Formatter:** Code must be formatted with `ruff` (Black-compatible).
- **Imports:** Sorted by `ruff` (isort style).
  1. Standard Library (`import math`, `import time`)
  2. Third Party (`import numpy as np`, `import torch`)
  3. Local (`from banhxeo.core...`)
- **Line Length:** Follow strict line limits (default 88 chars).

### 3. Naming Conventions
- **Classes:** `CamelCase` (e.g., `Tensor`, `LazyBuffer`, `Function`).
- **Functions/Methods:** `snake_case` (e.g., `realize`, `_broadcasted`).
- **Variables:** `snake_case`.
- **Constants:** `UPPER_CASE` (e.g., `DEFAULT_DEVICE`, `LoadOp.RAND`).
- **Internal/Private:** Prefix with `_` (e.g., `_ctx`, `_reduce`).

### 4. Typing
- **Type Hints:** REQUIRED for function signatures and class attributes.
- Use `typing` module (`Optional`, `Union`, `Tuple`, `List`, `ClassVar`).
- **Jaxtyping:** Used in some places but standard hints are preferred for core logic.

### 5. Error Handling
- Use specific exceptions: `IndexError`, `ValueError`, `NotImplementedError`.
- Use `assert` statements for internal invariants and shape checking.

### 6. Documentation
- **Docstrings:** Google-style docstrings are configured in `ruff` (though some rules are ignored).
- **Comments:** Use section separators for grouping methods in large classes:
  ```python
  # ---------- Binary Ops ----------
  ```
- **TODOs:** Mark incomplete features with `TODO`.

## 📂 Repository Structure

- `src/banhxeo/core/`: Core abstractions (`buffer.py`, `device.py`, `dtype.py`, `function.py`, `view.py`).
- `src/banhxeo/backend/`: Hardware backends and kernel generation (`triton.py`, `torch.py`).
- `src/banhxeo/nn/`: Neural network layers and optimizers.
- `src/banhxeo/tensor.py`: Main user-facing `Tensor` class.
- `tests/`: Test suite.