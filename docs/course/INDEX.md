# banhxeo Backend Course

This course is a self-guided path for turning banhxeo from a fragile first
compiler into a clearer educational ML framework.

The point is not to copy tinygrad or Magnetron. The point is to rebuild enough
of their ideas that you understand why those ideas exist.

## How To Use This Course

1. Read the assignment.
2. Try it yourself for 30-60 minutes.
3. Write down what confused you.
4. Check hints only after you have a concrete question.
5. Implement the smallest working version.
6. Add a test or a trace note.
7. Compare with tinygrad or Magnetron after your own attempt.

Difficulty ratings:

- ⭐ Straightforward
- ⭐⭐ Requires careful thinking
- ⭐⭐⭐ Challenging, multiple approaches possible

## Recommended Order

Follow this order even if another module looks shinier:

1. [`MODULE_0.md`](MODULE_0.md) - understand the current pipeline
2. [`MODULE_1.md`](MODULE_1.md) - master views, strides, offsets, and indexing
3. [`MODULE_2.md`](MODULE_2.md) - introduce a tiny kernel IR
4. [`MODULE_3.md`](MODULE_3.md) - make scheduling and fusion explicit
5. [`MODULE_5.md`](MODULE_5.md) - build correctness tests and fix gradients
6. [`MODULE_4.md`](MODULE_4.md) - improve specialized kernels
7. [`MODULE_7.md`](MODULE_7.md) - study memory reuse after correctness is stable
8. [`PROJECTS.md`](PROJECTS.md) - choose an end-to-end project

The appendix is reference material:

- [`APPENDIX.md`](APPENDIX.md)

## Module Map

| Module | Topic | Main Question |
| --- | --- | --- |
| [`MODULE_0.md`](MODULE_0.md) | Current codebase | What happens when a tensor is realized? |
| [`MODULE_1.md`](MODULE_1.md) | Views and indexing | How does logical tensor indexing map to storage? |
| [`MODULE_2.md`](MODULE_2.md) | IR and codegen | What should be structured before it becomes code? |
| [`MODULE_3.md`](MODULE_3.md) | Scheduling and fusion | Which ops become one kernel? |
| [`MODULE_4.md`](MODULE_4.md) | Specialized kernels | Why do matmul and reduce need different kernels? |
| [`MODULE_5.md`](MODULE_5.md) | Autograd correctness | Do gradients match PyTorch? |
| [`MODULE_7.md`](MODULE_7.md) | Memory management | When can buffers be reused? |

## Magnetron Study Track

Magnetron is useful as contrast:

- banhxeo is a lazy compiler path
- Magnetron is an eager runtime path

Read Magnetron when you want to understand runtime ownership, operator tables,
backend boundaries, tensor headers, storage, and native systems design. Do not
use it as a reason to rewrite banhxeo in C++ yet.

Suggested Magnetron checkpoints:

1. Read `mag_tensor.c` after Module 1.
2. Read `mag_op_stubs.c` after Module 3.
3. Read `mag_operator.h` before designing a cleaner op table.
4. Read `mag_autodiff.c` while doing Module 5.

## Ground Rules

- Keep patches small.
- Do not mix refactors with behavior changes.
- Every functionality change needs a test or a documented trace.
- Prefer one clear implementation over a clever general one.
- If you cannot explain the generated kernel, pause and trace it.
