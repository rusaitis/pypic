---
title: Architecture
---

# Architecture and Code Conventions

The design rules behind pypic, and the style conventions a contribution is
expected to follow. [CONTRIBUTING.md](https://github.com/rusaitis/pypic/blob/main/CONTRIBUTING.md)
covers the development setup and the checks CI runs; this page covers *why the
code is shaped the way it is*.

## Core design rules

**xarray is the container, NumPy is the compute engine.** `FieldDataset` wraps
an `xr.Dataset`, but every derived and diagnostic function takes and returns raw
NumPy arrays. xarray never enters the computation path — it holds coordinates,
names, and metadata, nothing more.

**Pure functions for physics.** Functions in `derived.py` and `diagnostics.py`
are pure: arrays in, arrays out, no `FieldDataset` dependency, no side effects.
This is what makes them testable against hand calculations and reusable outside
pypic's own containers.

**Normalized internally, converted at boundaries.** All computation happens in
code units. SI conversion is applied only at I/O and display boundaries, via
`in_si()` / `in_units()`. See [Schema](schema.md) for the normalization contract.

**Readers produce `FieldDataset`.** Each reader is a self-contained module;
adding support for a new simulation code means adding one module under
`pypic/readers/`, not touching anything downstream.

**Readers destagger to co-located grids.** Staggered-mesh codes (ARMS,
Athena++, BATSRUS face-centered B) store field components at different grid
locations. Each reader interpolates to a single co-located grid on load, so a
`FieldDataset` always represents one co-located grid. The original convention is
recorded in `StaggerInfo` as provenance and is not used in computation.
Destaggering is a reader concern, not a regridding concern.

**Selections describe regions, not data.** `PlaneSelection`, `BoxSelection`, and
`SphereSelection` are frozen dataclasses; `apply(data)` returns an ordinary new
`FieldDataset`.

**Selection APIs fail loud on unmatched names.** Anything that accepts a
user-supplied list of field, column, or component names (`fields=`, `columns=`,
…) must raise `KeyError` on a name that matches nothing — or expose an explicit
`strict_fields: bool = False` kwarg that does. Logging a warning instead is not
acceptable: warnings get swallowed in notebooks and pipelines, turning a typo
into a confusing failure several steps downstream. `Simulation.read` in
`readers/_registry.py` is the reference shape.

**Compare in SI by default, code units when appropriate.** Different
normalizations make code units incomparable across models, so cross-model
comparison converts to SI at the comparison boundary. Same-model comparisons
(identical normalization) can compare in code units directly, and dimensionless
quantities (beta, Mach numbers, entropy) need no conversion at all. Comparison
functions take a `units` parameter: `"si"` (default, cross-model safe),
`"code"`, or a display unit string.

**Relativistic generalizations via a `c=None` kwarg.** Derived functions with a
relativistic form accept `c: float | None = None`; `None` selects the
non-relativistic formula. `compute.py` auto-injects `c` via
`supports_relativistic=True` on the `Recipe` when `physics.relativistic` is set
in the dataset config, so registered quantities need no manual kwarg.

**The Pydantic validator is authoritative for `simulation.toml`.**
`pypic.schema` holds the Pydantic v2 models and the `validate_simulation_toml()`
entry point. `readers.config.load_config()` is a thin translator that delegates
all shape validation to Pydantic and then maps the validated `SimulationSchema`
onto the internal `SimulationConfig` / `GridInfo` / `Normalization` /
`SpeciesInfo`. The subpackage deliberately has zero pypic-internal imports (only
stdlib and pydantic) so it can be lifted into a standalone distribution. When
changing the schema, edit `pypic/schema/_models.py` first — `readers/config.py`
and `docs/schema.md` follow from it, never the reverse.

**The server is optional, not core.** `pypic.server` is a Starlette/FastAPI
data-serving layer gated behind the `server` extra. Core library imports never
trigger server dependencies.

**Scan sibling readers when fixing a pattern in one.** iPIC3D has three reader
variants (parallel, serial, H5hut) and BATSRUS has two (IDL, HDF5). Before
committing a fix in one, grep the siblings for the same pattern — duplicated
sign and weight conventions, dead guard code, and drifted docstrings tend to
travel together. `correct_pressure_tensor_component` in
`readers/ipic3d/_field_map.py` is the reference outcome: one shared
implementation across all three variants.

### Module layout

Core containers live at the `pypic/` top level, not under `readers/`:
`FieldDataset` in `dataset.py`, `GridInfo` in `grid.py`, and
`SimulationConfig` / `TabularData` / `ParticleData` / `StaggerInfo` in
`containers.py`. Reader protocols (`SimulationReader` and friends) stay in
`readers/_protocols.py`.

The dependency direction is one-way:

```
grid ← containers ← dataset ← everything else
```

**Explicit public API.** Every package `__init__.py` re-exports its public names
and declares `__all__`. Users import from `pypic` or `pypic.coordinates`, never
from internal modules.

### Deliberate exclusions

- No `astropy.units` in the computation path — 10-100× overhead.
- No hardcoded coordinate frame names (GSM, GSE, …) in function signatures.
- No `# --- Section Header ---` comment blocks; use module structure instead.

## Python conventions

**Python 3.13+.** The hard dependency is `copy.replace()` for frozen
dataclasses. PEP 695 `type` statements are available from 3.12.

**Tooling:** ruff (check + format, line length 88), uv, pytest with
`--doctest-modules`, mypy in strict mode.

| Topic | Convention |
|---|---|
| Type hints | Required on public signatures. Modern syntax: `X \| None`, `list[int]`, `tuple[float, ...]`. |
| Type aliases | PEP 695 `type` statements, not `TypeAlias`. `FloatArray` for array signatures, `Vector3` for 3-tuples — defined in `pypic/types.py`, imported inside `if TYPE_CHECKING:` blocks. |
| Dataclasses | <code>&#64;dataclass(frozen=True, slots=True)</code> for immutable data; `copy.replace()` for modified copies. |
| Thread safety | Prefer immutable data and pure functions. No shared mutable state across threads. Expose internal dicts as `MappingProxyType` via properties. `concurrent.futures` for parallelism; locks only for unavoidable mutations. |
| Enums | `StrEnum` for string enumerations. |
| Pattern matching | `match`/`case` where it beats an if/elif chain. After an exhaustive enum match use `case _ as unreachable: assert_never(unreachable)`, not `raise ValueError`. |
| Exception groups | `ExceptionGroup` when a reader hits multiple validation errors outside the TOML path. `simulation.toml` validation goes through `pydantic.ValidationError`, which already aggregates every violation — don't wrap it. |
| Paths | `pathlib.Path`, never `os.path`. |
| TOML | `tomllib` from the stdlib, not `toml` or `tomli`. |
| Scalars vs arrays | `math` for scalar constants (`math.inf`, `math.isfinite`); `numpy` for array operations. |
| Diagnostics | No `print()` — use `logging` or return values. |
| mypy + NumPy | Some ufunc returns need `# type: ignore[no-any-return]` even with `FloatArray`. Don't add these preemptively; let mypy tell you which are needed. |

## Naming

- **Functions** — descriptive English: `magnetic_field_magnitude()`,
  `plasma_beta()`, `alfven_speed()`.
- **Parameters** — short and scientific: `bx`, `rho`, `dt`, `q_over_m`. The
  docstring carries the full description.
- **Variables** — descriptive in running code: `electron_density`, not `ne`.
  Math symbols belong in docstrings.
- **Field keys** — short scientific strings: `"B_1"`, `"rho_c"`, `"P"`. The
  canonical set is defined in [Schema § 3](schema.md#3-canonical-field-names).
- **Constants** — `UPPER_SNAKE_CASE`. Use `scipy.constants` for physical
  constants rather than hand-typed values.
- **Booleans** — name them as questions: `is_periodic`, `has_field`.

## Docstrings

NumPy-style, in `r"""` raw strings so LaTeX survives. `$...$` inline and
`$$...$$` display, rendered by MkDocs Material + MathJax. Required sections: a
one-line summary, the LaTeX equation where one applies, Parameters, Returns, and
a runnable doctest under Examples — doctests are part of the suite via
`--doctest-modules`.

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

## Testing

- One assert per test where practical, with descriptive test names.
- Structural invariants ("every entry in registry X satisfies property Y") read
  better as a single test with a descriptive assertion message than as N
  parametrized copies of the same check.
- <code>&#64;pytest.mark.parametrize</code> for numerical validation.
- Derived quantities are tested against hand calculations and NRL Formulary
  values.
- `np.testing.assert_allclose` with explicit `rtol` / `atol`.
- Round-trip tests: `norm.to_si(norm.normalize(x)) == x`.
- Conservation tests: `div_b` of a curl field is zero to machine precision.
- Small synthetic arrays as fixtures, not large data files.
- Cover the edge cases: empty arrays, single elements, NaN handling.
- No network access, no external files, no specific simulation data. No trivial
  tests.

## Dependencies

Core: `numpy`, `scipy`, `xarray`, `h5py`, `pydantic` (v2, for `simulation.toml`
validation).

Optional, each behind its own extra: `plot` (matplotlib), `3d` (pyvista),
`cli`, `zarr`, `icechunk`, `arrow`, `duckdb`, `server`.

Dev: `pytest`, `ruff`, `mypy`, `mkdocs-material`, `mkdocstrings`.

Do not add dependencies without justification — prefer the standard library.

## Markdown

- No `---` horizontal rules between sections; headings provide enough
  separation.
- Task lists use numbered steps with `- [ ]` / `- [x]` checkboxes, checked off
  as work completes.
