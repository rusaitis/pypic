## What and why

<!-- What changes, and what problem it solves. Prefer the "why" over restating
     the diff — the diff is right there. -->

## Checks

`./scripts/check.sh` runs everything CI gates, over the paths CI covers.

- [ ] `./scripts/check.sh` passes (lint, format, schema, types, tests, docs)

## If applicable

- [ ] Physics changes cite a source (NRL Formulary, textbook, or paper).
- [ ] Schema changes edit `src/pypic/schema/_models.py` first, then regenerate:
      `uv run pypic schema export -o src/pypic/schema/simulation.schema.v1.0.json`
      (`./scripts/check.sh schema` verifies it)
- [ ] A fix to one reader was checked against its siblings (iPIC3D has three
      variants, BATSRUS two).
- [ ] New public functions carry type hints, a NumPy-style docstring, and a
      runnable doctest.
