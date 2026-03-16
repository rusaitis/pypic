# pypic

A Python toolkit for reading, analyzing, and plotting plasma simulation output.

## Dev commands

```sh
uv run pytest -v                      # full suite (tests + doctests)
uv run ruff check src tests           # lint
uv run ruff format --check src tests  # format check
uv run mypy src                       # type check
```

Optional dead code test using vulture:
```
uv run vulture src/
```
Optional Sloppy code test:

```
uvx sloppylint src/ tests/
```
