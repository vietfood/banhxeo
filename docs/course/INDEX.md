# banhxeo Backend Course

This course is a self-guided path for turning banhxeo from a fragile first
compiler into a clearer educational tensor compiler.

The point is not to copy tinygrad or Magnetron. The point is to rebuild enough
of their ideas that you understand why those ideas exist.

## Course Thesis

banhxeo should stay compiler-first during this course.

That means the main path is:

```text
Tensor API
  -> lazy graph
  -> view/indexing semantics
  -> forward correctness
  -> autograd
  -> scheduling and fusion
  -> basic kernel IR
  -> Triton rendering
  -> specialized kernels
  -> memory planning
  -> advanced IR and optimizations
```

Magnetron and eager runtimes are still useful, but as contrast. They teach
runtime ownership, storage, dispatch, and backend boundaries. They should not
pull this course away from compiler internals until the lazy path is clear.

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
3. [`MODULE_2.md`](MODULE_2.md) - build forward correctness tests
4. [`MODULE_3.md`](MODULE_3.md) - build and verify autograd
5. [`MODULE_4.md`](MODULE_4.md) - make scheduling and fusion explicit
6. [`MODULE_5.md`](MODULE_5.md) - introduce a tiny kernel IR
7. [`MODULE_6.md`](MODULE_6.md) - improve specialized kernels
8. [`MODULE_7.md`](MODULE_7.md) - study memory reuse after correctness is stable
9. [`MODULE_8.md`](MODULE_8.md) - study advanced IR and multi-backend lowering
10. [`MODULE_9.md`](MODULE_9.md) - add compiler optimizations
11. [`PROJECTS.md`](PROJECTS.md) - choose an end-to-end project

Why this order:

- Module 1 comes early because wrong views poison every later layer.
- Module 2 checks forward semantics before backward rules enter the picture.
- Module 3 builds autograd after forward behavior has a reference.
- Module 4 comes before Module 5 because scheduling decides what a kernel is;
  the IR should represent one kernel, not guess where kernels begin.
- Module 6 waits until the compiler path is inspectable. Matmul optimization is
  GPU programming, not the foundation of the framework.
- Modules 8 and 9 are advanced because MLIR-style design and optimization need
  pressure from a working smaller compiler first.

The appendix is reference material:

- [`APPENDIX.md`](APPENDIX.md)

## Module Map

| Module | Topic | Main Question |
| --- | --- | --- |
| [`MODULE_0.md`](MODULE_0.md) | Current codebase | What happens when a tensor is realized? |
| [`MODULE_1.md`](MODULE_1.md) | Views and indexing | How does logical tensor indexing map to storage? |
| [`MODULE_2.md`](MODULE_2.md) | Forward correctness | Does forward behavior match PyTorch? |
| [`MODULE_3.md`](MODULE_3.md) | Autograd | Do backward rules compose correctly? |
| [`MODULE_4.md`](MODULE_4.md) | Scheduling and fusion | Which ops become one kernel? |
| [`MODULE_5.md`](MODULE_5.md) | Basic IR and codegen | What should be structured before it becomes code? |
| [`MODULE_6.md`](MODULE_6.md) | Specialized kernels | Why do matmul and reduce need different kernels? |
| [`MODULE_7.md`](MODULE_7.md) | Memory management | When can buffers be reused? |
| [`MODULE_8.md`](MODULE_8.md) | Advanced IR | How do IR layers prepare for multiple backends? |
| [`MODULE_9.md`](MODULE_9.md) | Optimizations | Which rewrites preserve semantics and improve execution? |

## Ground Rules

- Keep patches small.
- Do not mix refactors with behavior changes.
- Every functionality change needs a test or a documented trace.
- Prefer one clear implementation over a clever general one.
- If you cannot explain the generated kernel, pause and trace it.
