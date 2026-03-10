PyPIC

TODO: Description and example run commands.

## Test Commands

  # Full suite (pytest tests + doctests + smoke)
  uv run pytest -v

  # Just the unit tests file
  uv run pytest tests/test_units.py -v

  # Just the doctests from source
  uv run pytest src/pypic/units.py --doctest-modules -v

  # Linting + formatting
  uv run ruff check src tests
  uv run ruff format --check src tests

  # Type checking
  uv run mypy src

  # All at once
  uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src && uv run pytest -v
