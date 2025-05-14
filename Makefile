.PHONY: test
test:
	uv run pytest tests

.PHONY: run
run:
	uv run python main.py

.PHONY: build
build:
	uv build 
