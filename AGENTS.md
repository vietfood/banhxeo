# banhxeo

Read this file and `README.md` before working in this repository.

## Project

`banhxeo` is an early research project for a small LLM inference engine. Treat
it as contributor-facing, non-production software with no stable API yet.

The project is not a general tensor framework, a training stack, or a broad
serving platform. Before implementation begins, define one model family, one
hardware target, one execution substrate, and one correctness baseline.

## Engineering priorities

- Make request, batch, model, and KV-cache ownership explicit.
- Separate engine semantics from kernel optimization.
- Prefer one concrete execution path before introducing abstractions.
- Validate deterministic small cases against an independent implementation.
- Record model, weights, hardware, configuration, inputs, and exact commands
  for performance results.
- Do not claim throughput, memory efficiency, compatibility, or production
  readiness without supporting evidence.

Weights, datasets, caches, generated artifacts, and raw profiler outputs do not
belong in Git.

## Contributions

Architecture changes must state their assumptions, consequences, rejected
alternatives, and verification plan. Keep changes narrow and avoid speculative
flexibility.

Treat review requests as read-only unless edits are explicitly requested.
Preserve unrelated work in dirty worktrees and never rewrite history for
convenience.

## Verification

No build or test command exists yet. Add them with the first implementation and
document exactly what each command verifies. Do not report an unexecuted path as
tested.

## Style

- Prefer brief, explicit names.
- Keep comments sparse and mechanism-focused.
- Use `rg` for search.
- Use `apply_patch` for manual edits.
- Never revert user changes unless explicitly asked.
