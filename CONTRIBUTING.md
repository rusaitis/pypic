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

CI gates six things, and `scripts/check.sh` runs all six in CI's order.
It is the single source of truth for *which paths* each check covers —
scripts/, benchmarks/ and the committed examples are covered too, not
just `src` and `tests`.

```sh
./scripts/check.sh          # everything
./scripts/check.sh lint     # ruff check
./scripts/check.sh format   # ruff format --check
./scripts/check.sh schema   # bundled JSON Schema is in sync
./scripts/check.sh types    # mypy, strict
./scripts/check.sh test     # pytest (suite + doctests)
./scripts/check.sh docs     # mkdocs build --strict
```

`--strict` turns unresolved cross-references and broken internal links
into build failures, so docs rot is caught here rather than on the
published site. For a live preview while editing:

```sh
uv run mkdocs serve                   # localhost:8000
```

Optional analyses, not run by CI:

```sh
uvx sloppylint src/ tests/            # sloppy-code heuristics
```

## Visual checks

Some plotting behavior is easier to verify by eye than by assertion.

They live in `scripts/visual/` rather than `tests/` — pytest collects
`tests/` with `--doctest-modules`, so a module there is imported at
collection time, and these force a matplotlib backend and require pyvista.

```sh
uv run python scripts/visual/visual_plots.py                # all themes -> tests/output/
uv run python scripts/visual/visual_plots.py --theme dark   # single theme
uv run python scripts/visual/visual_poincare.py             # Poincaré sections
uv run python scripts/visual/visual_dipole_3d.py            # interactive 3D dipole
```

## Benchmarks

Manual performance benchmarks live in `benchmarks/`. Not run by CI — use them to
confirm the batched/vectorized paths still pay off after a kernel change. Each
is self-contained, uses a seeded RNG, and prints a small table.

```sh
uv run python benchmarks/bench_batched_tracer.py   # batched vs scalar tracer
```

## Conventions

The full architecture and style rules live in
[docs/architecture.md](docs/architecture.md). The ones that most often trip up
a first contribution:

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

## Test layout

The flat modules under `tests/` are worked examples and regressions —
a specific input with a specific expected number, often hand-calculated
or cross-checked against the NRL Formulary. `tests/test_invariants/` is
the complementary half: Hypothesis property tests asserting identities
that must hold for every input in a generated domain, drawing on the
strategies in `tests/strategies.py`. Each module there opens with a
`# Source:` and a `# Claim:` line, and inherits `deadline=None` from the
`pypic` Hypothesis profile registered in `tests/conftest.py`.

Fixtures shared across suites live in `tests/conftest.py`
(`cartesian_3d`, `spherical_3d`), with synthetic-data factories in
`tests/_helpers.py` and on-disk simulation trees in
`tests/_sim_fixtures.py`.

## Adding a reader

Adding support for a simulation code is one new module under
`src/pypic/readers/`. The job is to map that code's native output onto the
canonical field names in [docs/schema.md](docs/schema.md) § 3 and return a
`FieldDataset`; everything downstream then works unchanged.

[`examples/custom_reader_example.py`](examples/custom_reader_example.py) is the
worked version: it generates a synthetic HDF5 file, declares a field map, and
reads it back through `open_simulation`. It runs with no external data, and
[`examples/README.md`](examples/README.md) indexes the numbered on-ramp that
builds up to it.

Anything you add under `examples/` is picked up automatically — but
`tests/test_examples.py` asserts that every committed script is listed there,
so add the filename when you add the file.

Use small synthetic fixtures under `tests/data/`, not real simulation output —
the test suite must not depend on network access or large files. The generators
in `scripts/` produce the committed fixtures (`generate_batsrus_fixture.py`,
`generate_ipic3d_fixture.py`, `generate_openggcm_fixture.py`); extend one rather
than hand-rolling a new binary blob.

Reader tests that need a real, uncommittable run are gated behind `--sim-data`:

```sh
uv run pytest --sim-data                 # scan examples/
uv run pytest --sim-data /path/to/runs   # scan a directory of simulations
```

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
Keep `./scripts/check.sh` green in each commit where practical.

## Conduct and security

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). To
report a security issue, follow [SECURITY.md](SECURITY.md) rather than opening
a public issue. Notable changes are recorded in [CHANGELOG.md](CHANGELOG.md).
