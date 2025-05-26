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
	(cd docs && make clean) && \
	uv run sphinx-apidoc -o docs/api src/banhxeo --force --separate --module-first && \
	(cd docs && make html)