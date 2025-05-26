.PHONY: test
test:
	uv run pytest tests

.PHONY: run
run:
	uv run python main.py

.PHONY: build
build:
	uv build 

.PHONY: docs
docs:
	uv run sphinx-apidoc -o docs/api src/banhxeo --force --separate --module-first && \
	uv run sphinx-build -M html docs docs/_build