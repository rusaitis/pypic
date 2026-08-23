# Project: pypic

See @README.md for what the project is and how it installs.

## Architecture and conventions

@docs/architecture.md is authoritative — design rules, module layout, Python
and naming conventions, docstring shape, testing expectations, and dependency
policy all live there, in the form contributors read. Change a rule there; do
not restate it here.

## Reference

- @TASKS.md — roadmap, what is done, what is next.
- @docs/schema.md — the `simulation.toml` contract and canonical field names
  shared by the three projects in this toolchain: this Python analysis library,
  a Three.js/WebGPU viewer, and a Rust PIC/MHD solver.
- @docs/equations.md — every derived quantity with its formula and SI factor.
- @docs/conventions.md — the physics and grid conventions that differ between
  textbooks (thermal speed, γ, temperature in energy units, ...).

## Dev Commands

```
# scripts/check.sh owns the path list CI covers — don't inline `src tests`,
# it under-covers scripts/, benchmarks/ and the committed examples.
./scripts/check.sh                    # everything CI gates, in CI order
./scripts/check.sh lint               # ruff check
./scripts/check.sh format             # ruff format --check
./scripts/check.sh types              # mypy, strict
./scripts/check.sh test               # pytest (suite + doctests)
./scripts/check.sh docs               # mkdocs build --strict
./scripts/check.sh schema             # bundled JSON Schema is in sync

# Regenerate the bundled JSON Schema after any change to
# pypic/schema/_models.py. The drift test in
# tests/test_schema_export.py and the lint-job step in
# .github/workflows/ci.yml will fail otherwise.
uv run pypic schema export -o src/pypic/schema/simulation.schema.v1.0.json

# Lint one or more simulation.toml files against the v1.0 schema.
uv run pypic schema validate path/to/simulation.toml
# Diff two JSON Schema documents (second defaults to bundled current).
uv run pypic schema diff path/to/proposed.json
```
