# Contributing to pypic

Bug reports, new simulation-code readers, and physics corrections are all
welcome. pypic is research software: a correction to a formula or a convention
is as valuable as a code contribution, and citing the source (NRL Formulary, a
textbook, a paper) is the fastest way to get it merged.

## Development setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/rusaitis/pypic.git && cd pypic
uv sync --all-extras --all-groups
```

## Checks

These four are what CI runs; all must pass.

```sh
uv run pytest -v                      # full suite (tests + doctests)
uv run ruff check src tests           # lint
uv run ruff format --check src tests  # format check
uv run mypy src                       # type check (strict)
```

The docs build is also gated, because `--strict` turns unresolved
cross-references and broken internal links into failures:

```sh
uv run mkdocs build --strict
uv run mkdocs serve                   # live preview at localhost:8000
```

Optional analyses, not run by CI:

```sh
uv run vulture src/ tests/ vulture_whitelist.py   # dead code
uvx sloppylint src/ tests/                        # sloppy-code heuristics
```

## Visual checks

Some plotting behavior is easier to verify by eye than by assertion.

```sh
uv run python tests/visual_plots.py                # all themes -> tests/output/
uv run python tests/visual_plots.py --theme dark   # single theme
uv run python tests/visual_dipole_3d.py            # interactive 3D dipole
```

## Benchmarks

Manual performance benchmarks live in `benchmarks/`. Not run by CI — use them to
confirm the batched/vectorized paths still pay off after a kernel change. Each
script is self-contained, uses a seeded RNG, and prints a small table.

```sh
uv run python benchmarks/bench_batched_tracer.py   # batched vs scalar tracer
```

## Conventions

The full architecture and style rules live in [CLAUDE.md](CLAUDE.md). The ones
that most often trip up a first contribution:

- **xarray is the container, NumPy is the compute engine.** Functions in
  `derived.py` and `diagnostics.py` are pure: arrays in, arrays out, no
  `FieldDataset` dependency and no side effects.
- **Normalized internally, converted at boundaries.** All computation happens in
  code units. SI conversion belongs at I/O and display only.
- **Selection APIs fail loud.** Anything accepting user-supplied names
  (`fields=`, `columns=`) raises `KeyError` on a name that matches nothing.
  Logging a warning instead is not acceptable — warnings get swallowed in
  notebooks, turning a typo into a confusing failure much later.
- **Readers destagger on load.** A `FieldDataset` always represents one
  co-located grid; the source convention is recorded in `StaggerInfo` as
  provenance.
- **Scan sibling readers when fixing a pattern in one.** iPIC3D has three reader
  variants and BATSRUS two; duplicated sign conventions and drifted docstrings
  tend to travel together.
- **Docstrings** are NumPy-style with `r"""` raw strings, carry the LaTeX
  equation where one applies, and include a runnable doctest. Doctests are part
  of the suite (`--doctest-modules`).
- **Type hints** are required on public signatures, in modern syntax
  (`X | None`, `list[int]`). mypy runs in strict mode.

## Adding a reader

Adding support for a simulation code is one new module under
`src/pypic/readers/`. The job is to map that code's native output onto the
canonical field names in [docs/schema.md](docs/schema.md) § 3 and return a
`FieldDataset`; everything downstream then works unchanged.

Use small synthetic fixtures under `tests/data/`, not real simulation output —
the test suite must not depend on network access or large files.

## Changing the schema

`src/pypic/schema/_models.py` is authoritative for `simulation.toml`. Edit it
first; `readers/config.py` and `docs/schema.md` follow from it. Regenerate the
bundled JSON Schema afterwards or CI will fail on the drift check:

```sh
uv run pypic schema export -o src/pypic/schema/simulation.schema.v1.0.json
```

## Commits and pull requests

Prefix commit subjects with `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, or
`chore:`, and say *why* the change is being made rather than restating the diff.
Keep the four checks above green in each commit where practical.
