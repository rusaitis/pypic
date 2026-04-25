# pypic — Architecture & Code Review

*A walk through the current `pypic/` codebase: how it is organized,
the design decisions that shaped it, how it is tested, what it does
exceptionally well, and what is still missing before it is a fully
production-grade scientific Python package.*

---

## 1. Overview

`pypic` is a Python 3.13+ toolkit for reading, analyzing, and plotting
output from PIC and MHD plasma simulations (iPIC3D, BATSRUS,
OpenGGCM, plus a generic HDF5 reader). It exposes one canonical
in-memory representation (`FieldDataset`, a thin wrapper around
`xr.Dataset`) and a large library of pure-function derived
quantities — plasma beta, Alfvén speed, skin depths, gyroradii,
pressure-tensor decomposition, entropy, Poynting flux, magnetosonic
speeds, and so on — all computed on raw NumPy arrays in normalized
code units.

Rough scale:

| Layer                | Approx. LoC |
|----------------------|-------------|
| `src/pypic/` (core)  | ~17 k       |
| Plotting (`plotting/` + `plotting/pyvista/`) | ~6–9 k      |
| I/O (`io/`, zarr/arrow) | ~3.2 k   |
| Readers (`readers/`) | ~2.4 k      |
| Tests (`tests/`)     | ~27 k       |
| **Total src + tests**| **~62 k**   |

The project sits in the same ecological niche as PlasmaPy, SpacePy,
and `yt`, but with a different emphasis: unify the field outputs of
kinetic and fluid plasma codes under one schema and one
normalization framework, so a notebook written against iPIC3D data
works unchanged against BATSRUS or OpenGGCM.

---

## 2. Code organization

### Top-level layout of `src/pypic/`

```
src/pypic/
├── __init__.py          # public API re-exports (150+ names)
├── types.py             # PEP 695 type aliases (FloatArray, Vector3)
├── grid.py              # GridInfo — structured grid metadata
├── containers.py        # StaggerInfo, SimulationConfig, TabularData, ParticleData
├── units.py             # Normalization, SpeciesInfo, PhysicsParams
├── dataset.py           # FieldDataset (xarray container + metadata)
├── fields.py            # FieldInfo registry, QuantityType StrEnum
├── compute.py           # Recipe-based derived-quantity dispatch
├── derived.py           # Pure-function physics library (100+ quantities)
├── diagnostics.py       # div_B, div_E, L2/Linf errors, field energy
├── comparison.py        # Cross-grid field comparison
├── regrid.py            # Uniform-to-uniform regridding
├── selections.py        # BoxSelection, PlaneSelection, SphereSelection
├── reconnection.py      # X-point detection, flux function, rec. rate
├── spectral.py          # 1D/2D/3D power spectral density
├── cli.py               # typer + rich CLI (info, fields, stats, plot, convert, …)
├── _aliases.py          # geometry-aware field-name alias tables
├── coordinates/         # CoordinateGeometry, operators, frame transforms
├── readers/             # iPIC3D, BATSRUS, OpenGGCM, SimpleReader + registry
├── io/                  # Zarr v3, VirtualiZarr, Icechunk, Parquet/Arrow, DuckDB
├── plotting/            # matplotlib 2D + pyvista 3D, themes, colormaps
└── traces/              # field-line and particle tracing, sampling
```

### Dependency direction

The layering is deliberate and enforced by CLAUDE.md:

```
types  →  grid  →  containers  →  units  →  dataset  →  compute / derived / ...
                                               ↑
                                 readers / io / plotting / coordinates
```

A reader depends on `FieldDataset` and the public container types; it
does not know about `compute.py`. `derived.py` does not know about
xarray or readers — it is pure NumPy in, NumPy out.

---

## 3. Architectural decisions

These are the load-bearing choices that recur throughout the code.

### 3.1 xarray as container, NumPy for computation

`FieldDataset` (`src/pypic/dataset.py`) wraps an `xr.Dataset` and
adds pypic-specific metadata (grid, normalization, species, physics,
frame transforms, alias maps). Every derived function in
`derived.py` takes raw `NDArray` arguments and returns an `NDArray`.
xarray never enters the compute path, because `astropy.units` and
xarray overhead dominate hot loops at 10–100×. xarray only provides
the container, the labels, and the I/O substrate.

### 3.2 Normalized internally, SI at the boundaries

All computation happens in code (normalized) units where `μ₀ = 1`
(SI-rationalized, not Gaussian). SI factors live in a single place
(`Normalization` in `units.py`), computed once per quantity type
(`density`, `b_field`, `pressure`, `energy_flux`, …) and applied at
I/O and display. This keeps `derived.py` mathematically clean — no
`4π`, no `μ₀`, no `k_B` littering the physics — and reserves unit
handling to the specific call sites that cross a boundary
(`in_si()`, `in_units("nT")`, `to_zarr`, plotting).

### 3.3 Pure-function physics

`derived.py` and `diagnostics.py` are entirely free of
`FieldDataset` knowledge. They accept arrays, return arrays, raise
`ValueError` on obvious shape mismatches, and nothing else. That
discipline is why they are trivially property-testable with
Hypothesis, doctestable inline, and reusable outside pypic — a user
can call `alfven_speed(b, rho_m)` from a standalone notebook.

### 3.4 Reader protocol + confidence-scored auto-detection

`SimulationReader` (`src/pypic/readers/_protocols.py`) is a
`Protocol`: `open_step(step) -> FieldDataset`, `steps`,
`available_fields`, plus optional particle and auxiliary-data hooks.
Adding a new simulation code means writing one `_reader.py` module
and one line in `register_reader()`. `open_simulation()` walks the
registry, asks each reader for a confidence score, and chooses the
best match. When every candidate fails, the failures are re-raised
as an `ExceptionGroup` so the user sees *why* each reader rejected
the input (`src/pypic/readers/_registry.py`), not a single
misleading error.

### 3.5 Co-located grids + `StaggerInfo` for provenance

Many of the simulation codes write on a staggered (Yee or MAC) mesh:
B on faces, E on edges, currents on edges, density at cell centers.
The canonical `FieldDataset` is always co-located (all fields on a
single grid — cells or nodes). Destaggering is a *reader concern*;
the original convention is recorded in `StaggerInfo`
(`src/pypic/containers.py`) for provenance and nothing else. This
removes a whole class of "which grid is this on?" bugs from
downstream code.

### 3.6 Canonical numbered field names, geometry-driven aliases

The schema is `B1/B2/B3` and `E1/E2/E3`, *not* `Bx/By/Bz`. Letter
aliases are installed at load time based on `geometry`: for
Cartesian data, `Bx → B1`; for spherical, `Br → B1`, `Btheta → B2`.
For non-Cartesian runs the letter aliases are simply absent. This
lets the same compute and plotting code work across geometries, and
it means the HDF5/Zarr layout has exactly one canonical spelling.
Per-species naming follows the same idea: `J1_s0`, `P11_s1`, with
`e/i` aliases (`Pe`, `Vi3`) as a convenience for the standard
two-species case.

### 3.7 Frozen dataclasses with slots, immutable-first

Every container type — `GridInfo`, `StaggerInfo`, `SimulationConfig`,
`TabularData`, `ParticleData`, `FieldInfo`, the `_Recipe` entries in
the compute registry — is declared as
`@dataclass(frozen=True, slots=True)`. Internal dicts that need to
be exposed go through `types.MappingProxyType` rather than being
copied or leaked mutably. Modified copies use
`copy.replace(obj, field=new)`. The upshot is that a `FieldDataset`
can be passed through an entire analysis pipeline without any
defensive copying, and the type is thread-safe for read-parallel
workflows.

### 3.8 Relativistic physics as an opt-in `c` kwarg

Functions with relativistic generalizations (Alfvén speed,
magnetosonic, kinetic energy density, …) accept a
`c: float | None = None` parameter. When `None`, the
non-relativistic formula runs. When `physics.relativistic = true`
in the simulation config, `compute.py` auto-injects `c` into every
registered recipe that advertises `supports_relativistic=True` —
users never pass `c` by hand. The non-relativistic limit is the
default, but it is exact, not approximate.

### 3.9 Fail loud on unmatched selection names

Any function that accepts a user-supplied list of field/column/
component names (`fields=`, `columns=`, …) raises `KeyError` on
names that match nothing, or exposes an explicit `strict_fields:
bool` kwarg that does. Logging-only warnings are forbidden — a
silent typo in a notebook is a worse failure mode than a loud one.
`Simulation.read` in `readers/_registry.py` is the reference shape.

### 3.10 Sibling-scan invariant for reader fan-out

Readers with multiple variants (iPIC3D has parallel/serial/H5hut,
BATSRUS has IDL/HDF5) are required to share physics helpers. The
canonical example is `correct_pressure_tensor_component` in
`readers/ipic3d/_field_map.py` — one shared implementation across
three readers. CLAUDE.md codifies the rule: when you fix a
convention bug in one reader, grep the siblings before committing.

### 3.11 Extras keep the core import lean

The core install (`numpy`, `scipy`, `xarray`, `h5py`) is minimal;
everything else is an optional extra:

```
plot      -> matplotlib
3d        -> pyvista
lazy      -> dask
cli       -> typer + rich
zarr      -> zarr v3 + numcodecs + virtualizarr + icechunk
icechunk  -> icechunk + numcodecs + zarr
arrow     -> pyarrow
duckdb    -> duckdb + pyarrow
```

Importing `pypic` on a minimal environment never drags in
matplotlib, pyvista, or zarr. Each optional integration is gated
behind `try: import X` with a helpful install message.

---

## 4. Modern Python features actually used

This is not a list of "things 3.13 supports" — these are in active
use in `src/pypic/`:

- **Frozen + slotted dataclasses** everywhere
  (`containers.py`, `grid.py`, `fields.py`, `compute.py`,
  `selections.py`).
- **`StrEnum`** for `QuantityType` (`fields.py`) and the private
  `_SpeciesArgs` enum (`compute.py`).
- **PEP 695 `type` aliases** for `FloatArray`, `Vector3` in
  `types.py`; imported inside `if TYPE_CHECKING:` blocks to avoid
  runtime cost.
- **Pattern matching** (`match/case`) for grid-geometry dispatch
  (`grid.py`) and derived-quantity recipe dispatch (`compute.py`).
- **`typing.assert_never`** after exhaustive enum matches — the
  codebase explicitly rejects `raise ValueError(...)` as the
  fall-through (CLAUDE.md § Python).
- **`copy.replace()`** for modified copies of frozen dataclasses
  (e.g. regrid, geometry transforms, CLI option overrides).
- **`types.MappingProxyType`** as the public face of internal
  dicts — `FieldDataset.aliases`, `StaggerInfo.field_locations`.
- **`ExceptionGroup`** in `readers/_registry.py` when multiple
  readers fail validation simultaneously.
- **`tomllib`** for parsing `simulation.toml` — no `toml`/`tomli`.
- **`pathlib.Path`** throughout — `os.path` is never used.

Tooling is equally current: ruff for both lint and format (with a
curated `allowed-confusables` list for physics notation like `ρ`,
`μ`, `σ`, `γ`, `Δ`, `∇`, `×`), mypy in strict mode, pytest with
`--doctest-modules`, uv for dependency and environment management.

---

## 5. Testing

### 5.1 Organization

`tests/` is flat at the top level (43 files, one per major module:
`test_compute.py`, `test_derived.py`, `test_plotting.py`,
`test_ipic3d_synthetic.py`, …) with a dedicated `tests/
test_invariants/` subdirectory (~31 files) for mathematical and
structural properties. Synthetic fixtures live in
`tests/_helpers.py` (`make_test_dataset`, `make_harris_dataset`,
`make_dipole_dataset`) and Hypothesis strategies in
`tests/strategies.py`. There are no committed simulation binaries —
`tests/data/` holds small hand-built synthetic reader fixtures;
real-data smoke tests are gated behind the `--sim-data` flag
registered in `conftest.py`.

### 5.2 Scale

Roughly **1,650 `def test_`** functions, plus every public
docstring runs as a doctest under `pytest --doctest-modules`. The
test-to-source ratio is ~1.6:1 by line count, which is unusually
high for a scientific Python project of this size and reflects the
correctness-first bent of the codebase.

### 5.3 Testing patterns

The test suite uses a layered approach:

- **Small-synthetic unit tests** with `np.testing.assert_allclose`
  and explicit `rtol`/`atol` for every numerical assertion. No
  real simulation data is required to run `pytest`.
- **Parametrized tests** — `@pytest.mark.parametrize` across
  normalizations, geometries, and species counts
  (`test_compute.py`, `test_config.py`, `test_batsrus.py`).
- **Property-based tests** via Hypothesis with 50–400 examples
  per test. `tests/strategies.py` defines ~40 strategies
  (`normalizations()`, `finite_physical_floats(1e-20, 1e20)`,
  `rotations()` for Rodrigues-generated SO(3), `frame_transforms()`
  for associativity). Round-trip laws like
  `to_si(normalize(x)) ≈ x` and vector identities
  (`∇·(∇×F) = 0`, Lagrange's `|E×B|² + (E·B)² = |E|²|B|²`) are
  stated directly as hypotheses rather than spot-checked.
- **Doctests** in public-facing `derived.py`, `compute.py`, and
  operator modules — the NumPy-style `Examples` blocks in
  docstrings *are* test cases.
- **Aggregated structural invariants** — tests like "every entry
  in the derived-quantity registry has a known unit type" are
  single-test-with-descriptive-message instead of N parametrized
  copies (CLAUDE.md § Testing).
- **Visual/manual tests** — `tests/visual_plots.py` and
  `tests/visual_dipole_3d.py` are opt-in scripts that render PNGs
  or launch pyvista windows. They are not in CI; they exist as a
  harness for human-eye verification before a release.

### 5.4 CI

`.github/workflows/ci.yml` has three jobs, all on
`ubuntu-latest` with `astral-sh/setup-uv`:

1. **lint** — `uv run ruff check src tests` + `ruff format --check`.
2. **type-check** — `uv run mypy src` in strict mode.
3. **test** — `uv run pytest` (collects doctests + invariants).

Each job does `uv sync --all-extras --all-groups`, so `pytest
--doctest-modules` can walk every file including the
pyvista-dependent plotting subpackage.

---

## 6. Strengths

### 6.1 The canonical-schema strategy is the core asset

`docs/schema.md` defines one set of field names (numbered +
geometry-aware aliases), one normalization contract, and one
per-species convention. Every reader translates native layouts to
that schema. As a result, `compute("beta")` or
`plasma_beta(p, b)` work identically against iPIC3D and BATSRUS
output, and adding a new code (ARMS, VPIC, Vlasiator) is "write one
file and register it." The schema itself is documented in long
form with explicit discussion of CGS vs SI-rationalized, Debye
length conventions, thermal-speed conventions, and gyrotropic
entropy exponents — this is unusually rigorous for a research-era
toolkit.

### 6.2 The derived registry is the real extensibility seam

`register_recipe(name, func, deps=..., quantity_type=...,
species_args=..., supports_relativistic=True)` is the single entry
point for user-defined quantities. The registry carries enough
metadata to auto-inject `c`, expand species-parameterized names
(`v_A_s2`), resolve dependencies (`Pi` loads the ion pressure
tensor), and report sensible errors. The same machinery powers
`compute("...")`, CLI `pypic fields`, and the documentation site
via `available_quantities()`.

### 6.3 Three complementary test layers catch three bug classes

- Unit tests catch "this equation is wrong."
- Hypothesis catches "this equation is right on the happy path
  but wrong at the numerical extremes."
- Doctests catch "the docstring describes an API that no longer
  exists."

Projects that only have unit tests routinely fail at 1e-20 or
1e+20; projects that only have property tests silently document
stale APIs. Running all three in one CI pass is the right choice.

### 6.4 StaggerInfo as an explicit provenance slot

The decision to destagger eagerly at read time *and* record the
original stagger convention in a frozen metadata object is the
cleanest resolution of a real tension: downstream code should be
grid-uniform (so compute and plotting stay simple), but users
still need to know the mesh semantics of their source code for
debugging and for publication methods sections.

### 6.5 A modern I/O tier that rivals larger projects

Zarr v3 with BloscCodec (zstd + bitshuffle), VirtualiZarr for
lazy HDF5 references, Icechunk for git-like versioning, Parquet
with Morton Z-order for particles, DuckDB for SQL-over-Parquet —
this stack is far more sophisticated than typical plasma-analysis
tooling. The care taken on edge cases (zarr versions 3.0.0–3.0.7
yanked for a data-loss bug; `BloscCodec` versus naked `ZstdCodec`;
Morton preferred over Hilbert for cheaper bit-interleaving) shows
in TASKS.md.

### 6.6 CLAUDE.md codifies convention

CLAUDE.md is a 10 kB living style guide: architecture invariants,
Python feature preferences, naming rules, docstring schema,
testing rules, dependency policy, dev commands. It exists
specifically so that code contributed to the project (by humans or
by LLM agents) stays coherent with what is already there. That
is visible in the codebase as *actual consistency* across 50+
modules, not just an aspirational document.

---

## 7. Weaknesses

None of these are architectural — they are all about the last mile
of packaging, observability, and community hygiene.

### 7.1 Not yet released

- No PyPI release. `pyproject.toml` has `version = "0.1.0"` as a
  static string — no `setuptools_scm` / `hatch-vcs`, so the
  version will drift from reality the moment a change lands.
- No `CHANGELOG.md`, `HISTORY.md`, or release tags.
- No conda-forge recipe; no Docker image; no archival DOI
  (Zenodo integration with GitHub releases). For a tool meant to
  be cited in plasma physics papers, a DOI is the single biggest
  missing piece.

### 7.2 No community-facing boilerplate

Missing files that reviewers and institutional users expect:

- `CITATION.cff`
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`

### 7.3 No code-coverage reporting

CI runs the full suite but never reports coverage. A threshold
(e.g. `fail_under = 85` on `src/`) plus a Codecov/Coveralls badge
would surface silent regressions.

### 7.4 No benchmark suite

There is no `pytest-benchmark`, `asv`, or CodSpeed baseline for
the hot paths (`compute()` dispatch, `to_zarr`/`from_zarr` round-
trip, reader load time, derived-quantity hot loops). Performance
regressions are currently invisible to CI.

### 7.5 No reader-contract test

Each reader has its own bespoke tests. There is no parametrized
test that says "every registered reader, given a synthetic golden
fixture, produces a `FieldDataset` with divergence-free B, the
right species count, and a round-trip through Zarr." That
structural invariant is exactly the kind of thing CLAUDE.md
promotes for other parts of the codebase — it hasn't been applied
to readers yet.

### 7.6 `vulture` is a dev dep but not in CI

`vulture>=2.15` is declared; `vulture_whitelist.py` exists at the
repo root; CI never invokes it. Dead-code detection is silently
off.

### 7.7 No pre-commit hooks

Contributors run ruff and mypy manually. A `.pre-commit-config.yaml`
that wired ruff, ruff-format, mypy, and a "trailing-whitespace"
hook would cost ten minutes and prevent the commits that currently
fail CI for format-only reasons.

### 7.8 No GPU / distributed path

Legitimate scope decision, but worth naming: `pypic` is NumPy +
xarray. Dask is an optional extra but the compute path assumes
in-memory arrays. For billion-cell datasets the current story is
"use `open_virtual` + chunked Zarr and `compute()` one tile at a
time" — it works, but it isn't as ergonomic as a CuPy / JAX
backend or a dask-native compute registry would be. This is a
future direction, not a current flaw.

### 7.9 No snapshot tests for plots or CLI output

`tests/visual_plots.py` is a manual harness. A syrupy or
`pytest-regressions` snapshot of CLI `info`/`stats` text and
small deterministic PNGs would catch unintended formatting
regressions without requiring a human to re-render everything.

---

## 8. Concrete improvements to the test suite

Ranked by value-per-effort:

1. **Reader-contract parametric test.** One `test_reader_contract.py`
   file that parametrizes over every registered reader, loads a
   tiny synthetic fixture (committed once, under 100 kB), and
   asserts:
   - `FieldDataset` round-trips through `to_zarr`/`from_zarr`.
   - `div_B` of a curl-of-A field is zero to machine precision.
   - Species indexing is stable (`_s0`, `_s1`).
   - Alias resolution respects geometry.
   - `StaggerInfo` is populated.

2. **Coverage gate in CI.** `coverage run -m pytest` with
   `fail_under = 85` on `src/`. Gate the PR merge on it. Start at
   the current baseline, raise quarterly.

3. **Benchmark suite.** `pytest-benchmark` baselines for
   `compute("beta")`, `to_zarr`, `open_simulation`, `regrid`,
   `plasma_beta`, `curl`. Commit a `benchmarks.json`; compare on
   PRs. CodSpeed integration would surface PR-level regressions
   automatically.

4. **Mutation testing** (`mutmut` or `cosmic-ray`) on `derived.py`
   and `compute.py`. Sign errors in physics code are the worst
   kind of silent bug; mutation testing is the tool that finds
   them.

5. **A shared synthetic golden dataset** under
   `tests/fixtures/golden.zarr`, built once by a `conftest.py`
   session fixture and consumed by reader tests, IO tests, regrid
   tests, and comparison tests uniformly. Right now each of those
   layers rolls its own small array.

6. **`nbval` or `pytest-examples`** to execute the fenced code
   blocks in `docs/getting-started.md` and `docs/tutorial.md` as
   tests. Documentation rot is invisible today.

7. **CI matrix on Python and NumPy.**
   `python-version: [3.13, 3.14-dev]` × `numpy: [2.0, latest]`.
   Catches the day the NumPy 2 upgrade tightens a cast.

8. **Snapshot tests** (syrupy) for the CLI `info` and `stats`
   subcommands. Text output is easy to snapshot and catches
   formatting-change regressions trivially.

---

## 9. Is this modern? Is it production-ready science code?

**Modern:** Yes, uncompromisingly. Python 3.13 as a hard minimum,
`StrEnum`, PEP 695 type aliases, `match/case`, `ExceptionGroup`,
`assert_never`, frozen+slotted dataclasses with `copy.replace()`,
strict mypy, `uv`, ruff, `tomllib`, `pathlib`. The optional I/O
stack (Zarr v3 + Icechunk + VirtualiZarr + Parquet + DuckDB) is
2025-current and rare among plasma physics tools.

**Production-ready as a library:** Architecturally, yes. The
public API is explicit, the container types are immutable, the
reader protocol is plugin-friendly, the derived registry is
extensible, and the test suite catches three distinct classes of
bug. A scientist at another institution could clone the repo, run
`uv sync && uv run pytest`, and trust the result.

**Production-ready as a distributable package:** Not quite. The
missing pieces are administrative, not technical:

- first PyPI release with a dynamic version (`hatch-vcs`);
- `CITATION.cff` and an archival DOI via Zenodo;
- `CHANGELOG.md` and `CONTRIBUTING.md`;
- coverage gate and a benchmark suite in CI;
- a reader-contract test that the next reader must pass;
- pre-commit hooks.

Estimated effort to close those gaps: on the order of a week of
focused polish, none of it touching the core architecture. The
hard engineering — the thing that takes years — is already done.

---

*Review written: 2026-04-19. Based on the `main` branch at commit
`ea3fdfa`. File paths and line numbers are accurate as of that
commit and may drift.*
