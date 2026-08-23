# Project: pypic

See @README.md for the project information.

## Architecture

Mirrored for contributors at @docs/architecture.md — update both when a rule changes.

- **xarray as container, NumPy for computation.** `FieldDataset` wraps `xr.Dataset`. All derived/diagnostic functions take and return raw NumPy arrays. xarray never enters the computation path.
- **Normalized internally, convert at boundaries.** All computation in code units. SI conversion only at I/O and display. See schema.md.
- **Pure functions for physics.** `derived.py` and `diagnostics.py` functions are pure: arrays in, arrays out. No FieldDataset dependency. No side effects.
- **Readers produce FieldDataset.** Each reader is a self-contained module. Adding a new simulation code = adding one .py file.
- **Core containers at top level.** `FieldDataset` (`dataset.py`), `GridInfo` (`grid.py`), and `SimulationConfig`/`TabularData`/`ParticleData`/`StaggerInfo` (`containers.py`) live at the `pypic/` top level — not in `readers/`. Reader protocols (`SimulationReader`, etc.) stay in `readers/_protocols.py`. Dependency direction: `grid` ← `containers` ← `dataset` ← everything else.
- **Selections describe regions, not data.** `PlaneSelection`, `BoxSelection` etc. are frozen dataclasses. `apply(data) → FieldDataset` returns a new standard FieldDataset.
- **Explicit public API.** Every package `__init__.py` re-exports public names and declares `__all__`. Users import from `pypic` or `pypic.coordinates`, never from internal modules.
- **Server is optional, not core.** `pypic.server` provides a Starlette/FastAPI data-serving layer for the Three.js/WebGPU viewer (webpic), gated behind a `server` extra. Core library imports never trigger server dependencies.
- **Pydantic validator is authoritative for `simulation.toml`.** `pypic.schema` holds the Pydantic v2 models and `validate_simulation_toml()` entry point for the v1.0 schema. `readers.config.load_config()` is a thin translator that delegates all shape validation to Pydantic, then maps the validated `SimulationSchema` to the internal `SimulationConfig`/`GridInfo`/`Normalization`/`SpeciesInfo`. The subpackage has zero pypic-internal imports (only stdlib + pydantic) so it can be lifted into a standalone distribution. The annotated reference template lives at `pypic.simulation.toml` at the repo root. When changing the schema, touch `pypic/schema/_models.py` first — `readers/config.py` and `docs/schema.md` follow from it, not the other way around.
- No `astropy.units` in computation path (10-100x overhead).
- No hardcoded coordinate frame names (GSM, GSE, etc.) in function signatures.
- No `# --- Section Header ---` comment blocks. Use module structure instead.
- **Readers destagger to co-located grids.** Staggered-mesh codes (ARMS, Athena++, BATSRUS face-centered) store fields on different grid locations (B on faces, E on edges, etc.). Each reader interpolates to a single co-located (cell-center or node) grid on load. `FieldDataset` always represents one co-located grid. Original stagger convention recorded in `StaggerInfo` metadata for provenance — not used in computation. Destaggering is a reader concern, not a regridding concern.
- **Compare in SI by default, code units when appropriate.** Cross-model comparison converts to SI via `in_si()` at the comparison boundary — different normalizations make code units incomparable. Same-model comparisons (identical normalization) can compare in code units directly, and dimensionless quantities (beta, Mach, entropy) need no conversion at all. Comparison functions should accept a `units` parameter: `"si"` (default for cross-model safety), `"code"` (same-normalization runs), or a display unit string.
- **Relativistic via `c=None` kwarg.** Derived functions that have relativistic generalizations accept `c: float | None = None`. When `None` (default), the non-relativistic formula is used. When provided, the relativistic branch activates. `compute.py` auto-injects `c` via `supports_relativistic=True` on `Recipe` when `physics.relativistic` is set in the dataset config — no manual kwarg passing needed for registered quantities.
- **Selection APIs fail loud on unmatched names.** Functions that accept a user-supplied list of field/column/component names (`fields=`, `columns=`, ...) must raise `KeyError` on names that match nothing, or expose an explicit `strict_fields: bool = False` kwarg that does. Logging-only warnings are forbidden — they get swallowed in notebooks and pipelines, turning a typo into a silent `KeyError` far downstream. See `Simulation.read` in `readers/_registry.py` for the reference shape.
- **Scan sibling readers when fixing a pattern in one.** iPIC3D has three reader variants (parallel, serial, H5hut); BATSRUS has two (IDL, HDF5); future readers will follow the same fan-out. Before committing a fix in one reader, grep the sibling files for the same pattern — duplicated sign/weight conventions, dead guard code, and drifted docstrings tend to travel together. The iPIC3D pressure-correction helper (`correct_pressure_tensor_component` in `readers/ipic3d/_field_map.py`) is the reference outcome: one shared implementation across all three readers.

## Python

- **Python 3.13+** — enables `copy.replace()` for frozen dataclasses (the hard dependency), improved error messages. `type` statement (PEP 695) is available from 3.12+.
- **Tooling:** ruff (check + format, line length 88), uv, pytest with `--doctest-modules`, mypy strict.
- **Type hints:** Required on public signatures. Modern syntax: `X | None`, `list[int]`, `tuple[float, ...]`.
- **Type aliases:** PEP 695 `type` statements, not `TypeAlias`. `FloatArray` for array signatures, `Vector3` for 3-tuples. Defined in `pypic/types.py`.
- **Dataclasses:** `@dataclass(frozen=True, slots=True)` for immutable data. `copy.replace()` for modified copies.
- **Thread safety:** Prefer immutable data (frozen dataclasses, tuples, frozensets) and pure functions. No shared mutable state across threads. Expose internal dicts as `MappingProxyType` via properties. Use `concurrent.futures` for parallelism, locks only for unavoidable mutations.
- **Enums:** `StrEnum` for string enumerations.
- **Pattern matching:** `match/case` where it improves readability over if/elif chains. After exhaustive enum matches, use `case _ as unreachable: assert_never(unreachable)` — not `raise ValueError`.
- **Exception groups:** `ExceptionGroup` when a reader encounters multiple validation errors outside the TOML schema path. `simulation.toml` validation goes through `pydantic.ValidationError` — which already aggregates every field violation into one exception, so don't wrap it.
- **Type alias imports:** Import from `pypic.types` inside `if TYPE_CHECKING:` blocks.
- **Paths:** `pathlib.Path`, never `os.path`.
- **TOML:** `tomllib` (stdlib), not `toml` or `tomli`.
- **Scalars vs arrays:** `math` for scalar constants (`math.inf`, `math.isfinite`). `numpy` for array operations.
- **Diagnostics:** No `print()` — use `logging` or return values.
- **mypy + NumPy:** Some NumPy ufunc returns need `# type: ignore[no-any-return]` even with `FloatArray` (mixed scalar/array arithmetic). Don't add these preemptively — let mypy tell you which are needed.

## Naming

- **Functions:** Descriptive English — `magnetic_field_magnitude()`, `plasma_beta()`, `alfven_speed()`
- **Parameters:** Short scientific — `bx`, `rho`, `dt`, `q_over_m`. The docstring provides the full description.
- **Variables:** Descriptive in running code — `electron_density` not `ne`. Math symbols in docstrings.
- **Field keys:** Short scientific strings — `"B_1"`, `"rho_c"`, `"P"` (see schema.md)
- **Constants:** `UPPER_SNAKE_CASE`. Use `scipy.constants` for physical constants, not hand-typed values.
- **Booleans:** Name as questions — `is_periodic`, `has_field`.

## Docstrings

NumPy-style with `r"""` raw strings (for LaTeX). `$...$` inline, `$$...$$` display (MkDocs + mathjax).
Required sections: one-line summary, LaTeX equation (if applicable), Parameters, Returns, Examples (runnable doctest).

```python
def alfven_speed(
    b: FloatArray,
    rho_m: FloatArray,
    *,
    c: float | None = None,
) -> FloatArray:
    r"""Compute the Alfvén speed.

    $$v_A = \frac{B}{\sqrt{\mu_0 \rho_m}}$$

    In normalized MHD units where $\mu_0 = 1$: $v_A = B / \sqrt{\rho_m}$.

    When *c* is provided, uses the relativistic form:
    $v_A = c\sqrt{\sigma / (1 + \sigma)}$ where $\sigma = B^2 / (\rho_m c^2)$.

    Parameters
    ----------
    b : NDArray
        Magnetic field magnitude in normalized units.
    rho_m : NDArray
        Mass density in normalized units.
    c : float or None
        Speed of light. When provided, the relativistic formula is used.

    Returns
    -------
    NDArray
        Alfvén speed in normalized units.

    Examples
    --------
    >>> import numpy as np
    >>> alfven_speed(np.array([1.0]), np.array([4.0]))
    array([0.5])
    """
    if c is not None:
        sigma = b**2 / (rho_m * c**2)
        return c * np.sqrt(sigma / (1.0 + sigma))
    return b / np.sqrt(rho_m)
```

Docs built with MkDocs Material + mkdocstrings.

## Markdown

- No `---` horizontal rules between sections — headings provide enough separation.
- Tasks use numbered steps with `- [ ]` / `- [x]` checkboxes, checked off when complete.

## Testing

- One assert per test where practical. Descriptive test names. Structural invariants ("every entry in registry X satisfies property Y") are better as a single test with a descriptive assertion message than N parametrized copies of the same check.
- `@pytest.mark.parametrize` for numerical validation.
- Derived quantities: test against hand calculations and NRL Formulary values.
- `np.testing.assert_allclose` with explicit `rtol`/`atol`.
- Round-trip tests: `norm.to_si(norm.normalize(x)) == x`.
- Conservation tests: `div_b` of a curl field should be zero to machine precision.
- Small synthetic arrays as fixtures, not large data files.
- Test edge cases: empty arrays, single elements, NaN handling.
- No network, external files, or specific simulation data. No trivial tests.

## Dependencies

Core: `numpy`, `scipy`, `xarray`, `h5py`, `pydantic` (v2, for `simulation.toml` validation)
Optional, each behind its own extra: `plot` (matplotlib), `3d` (pyvista), `cli`, `zarr`, `icechunk`, `arrow`, `duckdb`, `server`
Dev: `pytest`, `ruff`, `mypy`, `mkdocs-material`, `mkdocstrings`

Do not add dependencies without justification. Prefer standard library where possible.

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

## Progress

See @TASKS.md for current implementation plan and progress.

## SCHEMA

See @docs/schema.md for the architecture between three modern overlapping projects in development: a Python PIC/MHD analysis/basic visualization tool, a Three.js/WebGPU 3D visualizer/analyzer, and a Rust MHD/PIC simulation code.

## Equations

- See @docs/equations.md for detailed physics equation formulas to use.
- See @docs/conventions.md for the physics and grid conventions.
