# BUG LEDGER

This ledger is for concrete bugs discovered while running the course.

Do not fill this with guesses. A bug belongs here only after you have a repro,
an oracle, and a first guess about which concept owns it.

## Rules

- Minimize the repro before writing the entry.
- Record the oracle: manual, PyTorch, NumPy, or traceback only.
- Classify the concept owner.
- Fix at most one bug per patch.
- Do not mix bug fixes with cleanup.
- If the behavior is unsupported rather than wrong, mark it unsupported.

## Entry Template

```text
ID:
Title:
Status:
  - open
  - fixed
  - documented unsupported
  - deferred
Minimal repro:
Oracle:
  - manual expected value
  - one-off PyTorch comparison
  - one-off NumPy comparison
  - traceback only
Expected:
Actual:
Traceback or wrong output:
First suspicious file/function:
Concept owner:
  - Module 1 views/indexing
  - Module 2 forward correctness
  - Module 3 autograd
  - Module 4 scheduling
  - Module 5 IR/codegen
  - Module 6 specialized kernels
  - Module 7 memory management
Severity:
  - blocks tracing
  - wrong result
  - unsupported feature
  - confusing design
Next action:
  - fix now because it blocks tracing
  - convert to failing test in later module
  - document as unsupported
Notes:
```

## Open Bugs

Add entries below this line.
