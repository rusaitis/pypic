# Project: pypic

See @README.md for the project information.

## Architecture

- **xarray as container, NumPy for computation.** `FieldDataset` wraps `xr.Dataset`. All derived/diagnostic functions take and return raw NumPy arrays. xarray never enters the computation path.
- **Normalized internally, convert at boundaries.** All computation in code units. SI conversion only at I/O and display. See SCHEMA.md.
- **Pure functions for physics.** `derived.py` and `diagnostics.py` functions are pure: arrays in, arrays out. No FieldDataset dependency. No side effects.
- **Readers produce FieldDataset.** Each reader is a self-contained module. Adding a new simulation code = adding one .py file.
- **Selections describe regions, not data.** `PlaneSelection`, `BoxSelection` etc. are frozen dataclasses. `apply(data) → FieldDataset` returns a new standard FieldDataset.
- **Explicit public API.** Every package `__init__.py` re-exports public names and declares `__all__`. Users import from `pypic` or `pypic.coordinates`, never from internal modules.
- **No server in the library.** FastAPI lives in a separate project.
- No `astropy.units` in computation path (10-100x overhead).
- No hardcoded coordinate frame names (GSM, GSE, etc.) in function signatures.
- No `# --- Section Header ---` comment blocks. Use module structure instead.

## Python

- **Python 3.13+** — enables `copy.replace()` for frozen dataclasses, `type` statement (PEP 695), improved error messages.
- **Tooling:** ruff (check + format, line length 88), uv, pytest with `--doctest-modules`, mypy strict.
- **Type hints:** Required on public signatures. Modern syntax: `X | None`, `list[int]`, `tuple[float, ...]`.
- **Type aliases:** `type Vector3 = tuple[float, float, float]` (PEP 695), not `TypeAlias`.
- **Dataclasses:** `@dataclass(frozen=True, slots=True)` for immutable data. `copy.replace()` for modified copies.
- **Thread safety:** Prefer immutable data (frozen dataclasses, tuples, frozensets) and pure functions. No shared mutable state across threads. Use `concurrent.futures` for parallelism, locks only for unavoidable mutations.
- **Enums:** `StrEnum` for string enumerations.
- **Pattern matching:** `match/case` where it improves readability over if/elif chains.
- **Exception groups:** `ExceptionGroup` when a reader encounters multiple validation errors.
- **Type alias imports:** Import from `pypic.types` inside `if TYPE_CHECKING:` blocks.
- **Paths:** `pathlib.Path`, never `os.path`.
- **TOML:** `tomllib` (stdlib), not `toml` or `tomli`.
- **Scalars vs arrays:** `math` for scalar constants (`math.inf`, `math.isfinite`). `numpy` for array operations.
- **Diagnostics:** No `print()` — use `logging` or return values.

## Naming

- **Functions:** Descriptive English — `magnetic_field_magnitude()`, `plasma_beta()`, `alfven_speed()`
- **Parameters:** Short scientific — `bx`, `rho`, `dt`, `q_over_m`. The docstring provides the full description.
- **Variables:** Descriptive in running code — `electron_density` not `ne`. Math symbols in docstrings.
- **Field keys:** Short scientific strings — `"B1"`, `"rho_c"`, `"P"` (see SCHEMA.md)
- **Constants:** `UPPER_SNAKE_CASE`. Use `scipy.constants` for physical constants, not hand-typed values.
- **Booleans:** Name as questions — `is_periodic`, `has_field`.

## Docstrings

NumPy-style with `r"""` raw strings (for LaTeX). `$...$` inline, `$$...$$` display (MkDocs + mathjax).
Required sections: one-line summary, LaTeX equation (if applicable), Parameters, Returns, Examples (runnable doctest).

```python
def alfven_speed(b: NDArray, rho_m: NDArray) -> NDArray:
    r"""Compute the Alfvén speed.

    $$v_A = \frac{B}{\sqrt{\mu_0 \rho_m}}$$

    In normalized MHD units where $\mu_0 = 1$: $v_A = B / \sqrt{\rho_m}$.

    Parameters
    ----------
    b : NDArray
        Magnetic field magnitude in normalized units.
    rho_m : NDArray
        Mass density in normalized units.

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
    return b / np.sqrt(rho_m)
```

Docs built with MkDocs Material + mkdocstrings.

## Markdown

- No `---` horizontal rules between sections — headings provide enough separation.

## Testing

- One assert per test where practical. Descriptive test names.
- `@pytest.mark.parametrize` for numerical validation.
- Derived quantities: test against hand calculations and NRL Formulary values.
- `np.testing.assert_allclose` with explicit `rtol`/`atol`.
- Round-trip tests: `to_si(normalize(x)) == x`.
- Conservation tests: `div_b` of a curl field should be zero to machine precision.
- Small synthetic arrays as fixtures, not large data files.
- Test edge cases: empty arrays, single elements, NaN handling.
- No network, external files, or specific simulation data. No trivial tests.

## Dependencies

Core: `numpy`, `scipy`, `xarray`, `h5py`
Optional: `matplotlib` (plotting), `dask` (lazy I/O for large files)
Dev: `pytest`, `ruff`, `mypy`, `mkdocs-material`, `mkdocstrings`

Do not add dependencies without justification. Prefer standard library where possible.

## Dev Commands

```
uv run pytest -v                      # full suite (tests + doctests)
uv run ruff check src tests           # lint
uv run ruff format --check src tests  # format check
uv run mypy src                       # type check
```

## Progress

See @TASKS.md for current implementation plan and progress.

## SCHEMA

See @SCHEMA.md for the artchitecture between three modern overlaping projects in development: a Python PIC/MHD analysis/basic visualization tool, a Three.js/WebGPU 3D visualizer/analyzer, and a RUST MHD/PIC simulation code.

## Equations

- See @docs/equations.md for detailed physics equation formulas to use.
- See @docs/conventions.md for the physics and grid conventions.
