# banhxeo

`banhxeo` is an early research project for a small LLM inference engine. Its
goal is to make model execution, memory management, batching, KV-cache behavior,
and performance decisions explicit.

> [!WARNING]
> The project is in a pre-implementation reset. It has no stable API, runnable
> engine, or production guarantees. Use future releases at your own risk.

## Direction

The initial design will choose one model family, one hardware target, and one
execution substrate before adding generality. The project is not intended to
be a tensor framework, training system, or general-purpose serving platform.

The first technical milestone must define:

- the supported model and weight format;
- the target hardware;
- the runtime-versus-compiler boundary;
- request, batch, and KV-cache ownership;
- a correctness oracle and measurable baseline.

## Status

No implementation, build environment, or dependency set is defined yet.

## Contributing

Architecture proposals should make their assumptions and rejected alternatives
explicit. Code should begin only after the first milestone has a narrow,
testable contract.

See [`AGENTS.md`](AGENTS.md) for repository-specific engineering guidance and
[`LICENSE`](LICENSE) for licensing terms.
