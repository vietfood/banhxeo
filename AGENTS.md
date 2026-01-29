# banhxeo - AI Agent Guide

This repository contains **banhxeo**, a minimalist educational deep learning framework. It implements lazy evaluation and Triton kernel generation from scratch.

## 🛠️ Build & Environment

This project uses `uv` for dependency management.

- **Install/Sync dependencies:**
  ```bash
  uv sync
  source .venv/bin/activate
  ```
- **Build package:**
  ```bash
  uv build
  ```
- **Run Docs:**
  ```bash
  make docs
  ```

## 🧪 Testing & Linting

Tests are located in `tests/`. `pytest` is the test runner.

- **Run all tests:**
  ```bash
  uv run pytest tests
  # OR via make
  make test
  ```
- **Run a single test file:**
  ```bash
  uv run pytest tests/small_tests/relu.py
  ```
- **Run a specific test function:**
  ```bash
  uv run pytest tests/small_tests/relu.py::test_activations
  ```
- **Linting & Formatting:**
  Uses `ruff` for both linting and formatting.
  ```bash
  uv run ruff check .
  uv run ruff format .
  ```
- **Type Checking:**
  Uses `pyright`.
  ```bash
  uv run pyright .
  ```

## 🎨 Code Style & Conventions

Adhere strictly to the following conventions to maintain the "minimalist and educational" philosophy.

### 1. General Philosophy
- **Tiny & Readable:** Keep code concise. Prefer simple implementations over complex optimizations unless necessary for the educational goal.
- **Lazy Evaluation:** Operations should generally be lazy, building a computation graph (`LazyBuffer`) rather than executing immediately.
- **Triton Codegen:** The backend targets Triton kernels.

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

## 🐛 Debugging

- **Environment Variable:** Set `DEBUG=1` (or higher) to see generated kernels and execution details.
  ```bash
  DEBUG=1 uv run python main.py
  ```
