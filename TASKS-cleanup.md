# pypic — Code & Physics Quality Sweep

## Context

After several feature-heavy commits (deep FieldDataset/pyvista integration, custom
colorbars, Poynting/energy-flux SI separation, iPIC3D pressure tensor fix), the
codebase has accumulated some predictable cruft: duplicated reader maps, parallel
matplotlib/pyvista helper code, hand-coded test fixtures repeated across files, a
stub module, and a few SCHEMA/registry consistency gaps. None of it is broken —
the suite is green and the physics formulas check out against `docs/equations.md`
— but it will get harder to evolve if left alone.

This plan organizes the cleanup into 14 independently-executable units, each
small enough to land in a single PR/commit but meaningful on its own. Units are
ordered by **confidence × risk**: trivially-safe wins first, then test
consolidation, then refactors that span backends, finally the API-breaking
operator hardening. Every unit has a verification step so we know it landed
cleanly.

Three explore agents audited readers, plotting, and compute (against
`SCHEMA.md` + `docs/equations.md` + `docs/conventions.md`). Findings are
distilled below — speculative/aesthetic complaints were dropped. Findings
verified inline before plan-writing are marked ✓.

The good news up front: **no physics formula bugs were found**. All derived
quantities in `derived.py` match `equations.md`, SI factors in `fields.py` match
the conversion table, NRL thermal-speed convention is honored, cyclotron
frequencies are positive-by-convention, the gyrotropic entropy exponent of 5
correctly comes from CGL invariants (not γ=5/3), and node-centered grid
treatment in `diagnostics.py` is consistent with `conventions.md`. The work
below is structural quality, not correctness rescue.

---

## Unit 1 — Quick wins (low-risk dedup + stub removal)

**Goal:** clear obvious cruft in a single small commit before touching
anything structural.

Files:
- `src/pypic/readers/ipic3d/_field_map.py` ✓ (verified)
- `src/pypic/plotting/overlays.py` ✓ (verified — 3 lines, just a docstring)

Actions:
1. Merge `_PHDF5_EFLUX_MAP` and `_H5HUT_EFLUX_MAP` (lines 114–124 are
   character-identical — both map `EFx/EFy/EFz` → `EF1/EF2/EF3`) into a single
   `_EFLUX_MAP`. Update `per_species_eflux_canonical()` line 175 to read from
   the unified map. Grep for any other `_PHDF5_EFLUX_MAP` / `_H5HUT_EFLUX_MAP`
   refs in `_parallel.py`, `_serial.py`, `_h5hut.py` and switch them.
2. Decide on `plotting/overlays.py`: it's a 3-line file with only a docstring
   and `from __future__ import annotations`. Either delete it (and update
   `plotting/__init__.py` if it's re-exported) or fold it into
   `plotting/_format.py` / `plotting/_badge.py`. Default: **delete** —
   `_badge.py` already owns the overlay logic.

Verification:
- `uv run pytest -q tests/test_ipic3d_synthetic.py tests/test_plotting.py`
- `uv run ruff check src tests && uv run mypy src`

---

## Unit 2 — Move shared test fixtures into `tests/_helpers.py`

**Goal:** stop duplicating synthetic-grid setup across reader and plotting
tests.

Files:
- `tests/_helpers.py` (already exists, currently sparse)
- `tests/conftest.py`
- `tests/test_simple_reader.py`, `tests/test_readers_base.py` — local
  `_sample_grid()`, `_sample_config()`, `_make_fields()` helpers
- `tests/visual_plots.py` — `_make_harris_fields()` (~620 lines context)
- `tests/visual_dipole_3d.py` — `_dipole_field()`, `_seed_points()` (~370 lines)
- `tests/test_plotting.py` — its own synthetic fixtures

Actions:
1. Audit each `tests/test_*.py` and `tests/visual_*.py` for inline grid /
   field-array helpers. Promote the genuinely shared ones to `_helpers.py`:
   - `make_uniform_grid(nx, ny, nz, dx=1.0, geometry="cartesian")`
   - `make_synthetic_fielddataset(grid, fields=("B1","B2","B3"))`
   - `make_harris_dataset()` — for the double-Harris fixture
   - `make_dipole_dataset()` — for the 3D dipole fixture
2. Each function should be ≤30 lines, no `print`, type-hinted, with one
   doctest.
3. Update test files to import from `_helpers` rather than re-defining.

Verification:
- `uv run pytest -q` (full suite)
- `uv run python tests/visual_plots.py` and `uv run python tests/visual_dipole_3d.py`
  must still produce output without diffs.

Non-goals: don't merge `visual_plots.py` and `visual_dipole_3d.py` themselves —
they are intentionally separate eyeball tests.

---

## Unit 3 — Parametrize repetitive reader probe tests

**Goal:** collapse three near-identical probe test classes into one
parametrized test.

Files:
- `tests/test_registry.py` (lines ~243–333 cover iPIC3D / BATSRUS / OpenGGCM
  probes with the same empty-dir / positive / full-match structure)

Actions:
1. Replace the three duplicated test classes with a single
   `@pytest.mark.parametrize` over `(reader_name, fixture_factory,
   expected_min_score)`.
2. Where the per-reader probes have unique signals worth their own assertion
   (e.g. BATSRUS `.h` timestamp pattern from `8818813`), keep that as a
   single dedicated test — don't lose the regression coverage.
3. Apply the same parametrization principle to any `test_derived.py` cluster
   where ~13 magnitude functions are tested in a copy-pasted pattern (see
   audit finding 4 of compute pass): convert those to parametrized tests over
   `(func, input, expected)` tuples.

Verification:
- `uv run pytest -q tests/test_registry.py tests/test_derived.py`
- Test count should *drop*; coverage report unchanged.

---

## Unit 4 — Reader probe helper

**Goal:** eliminate the score-accumulation pattern duplicated across
`ipic3d/_probe.py`, `batsrus/_probe.py`, `openggcm/_probe.py`.

Files:
- `src/pypic/readers/base.py` — add helper
- The three `_probe.py` modules

Actions:
1. Add `score_signals(path, signals)` to `readers/base.py`:
   ```python
   def score_signals(
       path: Path, signals: Sequence[tuple[str, float]]
   ) -> float:
       """Sum signal weights for files matching glob patterns under *path*.

       Returns min(score, 1.0). Used by reader probe functions.
       """
   ```
2. Refactor each `_probe.py` to express its detection rules as a
   `signals: list[tuple[str, float]]` constant + a single
   `score_signals(path, _SIGNALS)` call. Keep any code that needs to *read*
   file contents (BATSRUS `.h` timestamp regex, OpenGGCM grid header) as a
   bonus check on top of the score, not inside the helper.
3. The probe helper itself gets one test with synthetic dirs.

Verification:
- `uv run pytest -q tests/test_registry.py`
- Manually open one fixture per format (`tests/data/...` if present, or
  whatever the reader test fixtures use) to ensure detection scores are
  unchanged.

Non-goals: don't change the probe contract or `ProbeResult` shape — public
API stable.

---

## Unit 5 — iPIC3D / BATSRUS config-builder dedup

**Goal:** the two `to_simulation_config()` builders share the same
"GridInfo + Normalization + species + frame" assembly logic. Pull the
common bits out.

Files:
- `src/pypic/readers/ipic3d/_config.py` (~569 lines)
- `src/pypic/readers/batsrus/_config.py` (~321 lines)
- `src/pypic/readers/base.py` (host for shared helpers) **or** new
  `src/pypic/readers/_config_helpers.py` if it'd bloat `base.py`

Actions:
1. Identify the truly shared helpers (likely candidates: building a
   `GridInfo` from `(dimensions, spacing, origin, geometry)`,
   building a `Normalization.pic_*` from a species + density, building
   a `[[species]]` list from raw `(name, charge, mass, ...)` tuples).
2. Move them out to one place. Each reader's `to_simulation_config()` should
   then read like a flat sequence of "parse native config → call shared
   builder" steps.
3. Be careful: iPIC3D supports both `.inp` and `settings.hdf` paths — the
   shared helper should not absorb that branching.

Verification:
- `uv run pytest -q tests/test_config.py tests/test_batsrus.py tests/test_ipic3d_synthetic.py`
- Round-trip: load a fixture, dump `SimulationConfig.__dict__`, compare to
  baseline (we can capture this baseline as part of the unit).

Non-goals: do **not** merge the readers themselves. The "one .py file per
sim code" rule from CLAUDE.md still holds.

---

## Unit 6 — Standardize reader error types

**Goal:** consistent error vocabulary across readers.

Files:
- `src/pypic/readers/ipic3d/*.py`, `batsrus/*.py`, `openggcm/*.py`,
  `_simple.py`

Actions:
1. Audit error raises in each reader. Establish convention:
   - `KeyError` → unknown canonical/native field name (already standard for
     dict-style lookup)
   - `ValueError` → malformed config, bad shape, unit mismatch
   - `FileNotFoundError` → missing file
   - `ExceptionGroup` → multi-error reader probe failures only
2. Apply consistently. Add one regression test per reader for the most
   common failure (unknown field name).

Verification:
- `uv run pytest -q`
- `uv run mypy src` clean.

Non-goals: do not invent custom exception classes — stdlib types are enough.

---

## Unit 7 — Extract shared plotting overlay/contrast helpers

**Goal:** kill the largest duplication in the project. Both backends
re-implement luminance, contrast ratio, overlay color picking, and overlay
defaults.

Files:
- `src/pypic/plotting/_badge.py` (683 lines ✓)
- `src/pypic/plotting/pyvista/_badge.py` (462 lines ✓)
- New: `src/pypic/plotting/_overlay_common.py`

Actions:
1. Extract pure functions to `_overlay_common.py`:
   - `_lum(rgb)` — relative luminance per WCAG
   - `_contrast_ratio(c1, c2)`
   - `_pick_overlay_variant(colors)` — chooses light/dark variant
   - `_resolve_overlay_colors(theme, variant)` — returns the dict the
     backends consume
   - `_format_status_text` already lives in `_format.py`; verify both
     backends use it (the matplotlib `overlays.py` re-export pattern hints
     they don't always).
2. Both `_badge.py` modules import these instead of re-implementing.
3. Expected payoff: ~150 LOC removed.

Verification:
- `uv run pytest -q tests/test_plotting.py`
- `uv run python tests/visual_plots.py` then visual diff against the
  baseline screenshots (the visual tests were updated in `7ab162d`).
- `uv run python tests/visual_dipole_3d.py` for the pyvista path.

Non-goals: do not unify the backend-specific drawing code (matplotlib uses
`AnchoredOffsetbox`, pyvista uses 2D actors) — only the pure helpers move.

---

## Unit 8 — Unify colormap resolution between backends

**Goal:** one source of truth for "given a field, pick the right colormap".

Files:
- `src/pypic/plotting/_colormaps.py`
- `src/pypic/plotting/pyvista/_theme.py` (currently has `resolve_cmap()`)
- `src/pypic/plotting/pyvista/_meshes.py` (calls `is_positive_definite()`
  inline at lines ~161–171 and stitches the result onto `resolve_cmap()`
  manually)

Actions:
1. In `_colormaps.py`, expose a single
   `resolve_field_colormap(name, values, theme, info, cmap=None) -> tuple[str, Colormap]`
   that returns both the name (for serialization/logging) and the
   instantiated `Colormap` object. It should call `is_positive_definite()`
   internally.
2. Matplotlib backend uses the `name`, pyvista uses the `Colormap` object.
3. Delete the now-redundant `resolve_cmap()` shim in pyvista's `_theme.py`.
4. Add one parametrized test that asserts both backends pick the *same*
   colormap for a curated list of field names (`B1`, `rho_c`, `n_s0`,
   `J_dot_E`, `psi`, `div_B`, `|B|`).

Verification:
- `uv run pytest -q tests/test_plotting.py`
- Colormap-parity test must pass.
- Visual smoke: `uv run python tests/visual_plots.py --theme dark` (one
  theme is enough).

---

## Unit 9 — Plotting backend parity audit + fix

**Goal:** the recent commit message claims "mpl backend parity" — verify it
and close any remaining gaps.

Files:
- `src/pypic/plotting/slices.py` (matplotlib `plot_field_slice`)
- `src/pypic/plotting/pyvista/_meshes.py` (`add_equatorial_surface` and
  friends)
- `src/pypic/plotting/comparison.py`, `panels.py`, `cross_section.py`,
  `kymograph.py`, `lines.py`, `vectors.py`, `scatter.py` (none of these
  were deeply audited; this unit covers them)

Actions:
1. For each public matplotlib function with a pyvista counterpart, write a
   one-line "parameter parity table" in a tracking file. Flag every
   parameter present in only one backend.
2. Decide for each gap: (a) lift to both, (b) document as
   backend-specific (e.g. opacity transfer functions are pyvista-only;
   streamplot is matplotlib-only — both are intentional), or (c) delete.
3. Apply the (a) cases. Specifically, the audit flagged `extremes`,
   `colorbar_ticks`, `badge` as missing from pyvista — confirm whether
   each is portable.
4. Audit `vectors.py` line 88 (`color_name = f"|{field}_{{plane}}|"`) — the
   literal `{plane}` braces in the rendered label are intentional? Decide:
   change to LaTeX `|B_\mathrm{plane}|` or to plain `|B|` (in-plane). The
   original audit called this a "bug"; on inspection ✓ it's a cosmetic
   choice, not a lookup bug — `field_info()` is called with a different
   string on the next line. Still, it reads weird in a plot title.

Verification:
- `uv run python tests/visual_plots.py` (full theme matrix)
- `uv run python tests/visual_dipole_3d.py`
- New test: `tests/test_plotting_parity.py` listing the parity matrix as
  data, with a single `assert symmetric_diff == set()` check.

---

## Unit 10 — Move plotting magic numbers into `styles.py` / `PlotTheme`

**Goal:** stop having `4.5 * ncols`, `1.78`, `font_overlay * 0.6`, etc.
sprinkled across the plotting modules.

Files:
- `src/pypic/plotting/styles.py` — extend `PlotTheme` with the missing
  fields
- `src/pypic/plotting/panels.py` (lines ~115, 118)
- `src/pypic/plotting/slices.py` (line ~250: `label_fontsize: float = 7`)
- `src/pypic/plotting/_badge.py` (lines ~413, 649: progress bar / arrow
  sizes)

Actions:
1. Add fields to `PlotTheme` for: `figsize_per_col`, `figsize_per_row`,
   `overlay_fontsize_scale`, `contour_label_fontsize`,
   `progress_bar_height`, `arrow_size`, `figure_dpi`, `savefig_dpi`.
2. Replace literals in callers. Where the literal is genuinely arbitrary
   (e.g. `1.78` appears to be eyeballed), keep the same default — don't
   "improve" the visuals.
3. `styles.py` adds a doctest showing the new field exists and round-trips
   through theme save/load.

Verification:
- `uv run pytest -q tests/test_plotting.py`
- `uv run python tests/visual_plots.py` — pixel-diff against baseline
  should be empty (we are *not* changing values, only their location).

---

## Unit 11 — Compute/registry consistency audit

**Goal:** mechanically prove that every name mentioned in `SCHEMA.md` is
either a registered primitive or computable via `compute()`, and vice
versa.

Files:
- `src/pypic/fields.py` — `_FIELD_INFO` registry
- `src/pypic/compute.py` — `_REGISTRY` of recipes
- `src/pypic/_aliases.py` — alias maps
- `tests/test_fields.py` or new `tests/test_registry_consistency.py`

Actions:
1. Parse the `## 3. Canonical Field Names` tables of `SCHEMA.md` (or
   maintain a hand-curated list — the parsing is brittle, plain Python
   constant is probably better) into a set of canonical names.
2. New test:
   ```python
   def test_every_schema_field_is_known():
       for name in CANONICAL_NAMES:
           assert (
               name in PRIMITIVE_FIELDS
               or name in compute._REGISTRY
               or name in _aliases.ALIAS_MAP
           ), f"{name} from SCHEMA.md is unreachable"
   ```
3. New test: every entry in `_FIELD_INFO` has a defined `quantity_type`
   that resolves to a non-None SI factor.
4. New test: every recipe in `compute._REGISTRY` references only field
   names that are themselves in the registry, in `PRIMITIVE_FIELDS`, or
   in `ALIAS_MAP`. (Catches typos in recipe dependencies.)

Verification:
- `uv run pytest -q tests/test_registry_consistency.py`
- Expect this to **fail on first run** for ~3–6 names — that's the point.
  Either fix the registry or delete the orphaned name from SCHEMA.md.

---

## Unit 12 — Edge-case test gaps in `derived.py` / `diagnostics.py`

**Goal:** parametrized coverage for the cases the existing tests skip
(empty arrays, ±inf, charge sign invariance for cyclotron/plasma freqs).

Files:
- `tests/test_derived.py`, `tests/test_diagnostics.py`

Actions:
1. Add one parametrized "edge cases" class per derived family (magnitudes,
   characteristic scales, energy densities). Cover:
   - Empty arrays (`shape=(0,)`)
   - Single-element arrays
   - NaN propagation
   - Negative density input → NaN, not exception
   - Charge sign invariance (`omega_ce`, `omega_pe`, `lambda_D`,
     `gyroradius` should be identical for `q=+e` vs `q=-e`)
2. Replace the tautological "known value" tests for `lambda_D` (audit
   finding 6 of compute pass) with **scaling property** tests:
   - `lambda_D(2T, n) == sqrt(2) * lambda_D(T, n)`
   - `lambda_D(T, 4n) == 0.5 * lambda_D(T, n)`
   These survive formula refactors that hand-coded values can't.

Verification:
- `uv run pytest -q tests/test_derived.py tests/test_diagnostics.py`
- Coverage: existing tests should still pass; new tests add ≤30 cases.

Non-goals: don't replace **all** known-value tests — the NRL Formulary
cross-checks are valuable as anchor points.

---

## Unit 13 — Operators: remove default geometry (TASKS Step 31)

**Goal:** force every caller of `divergence` / `curl` / `gradient` to be
explicit about geometry, so a future spherical bug can't be silently
masked. ✓ Confirmed `geometry: GeometryType = GeometryType.CARTESIAN` is
still the default at `operators.py:48` (and analogous lines for `curl`
and `gradient`).

Files:
- `src/pypic/coordinates/operators.py`
- All callers (likely: `diagnostics.py`, `compute.py`, `derived.py`,
  selected tests)

Actions:
1. Drop the `= GeometryType.CARTESIAN` default from `divergence`, `curl`,
   `gradient`.
2. Run the test suite — every red mark is a caller that needs `geometry=`
   added explicitly. Pass `data.grid.geometry.geometry_type` from
   FieldDataset paths.
3. Update `TASKS.md` Step 31 status to ✅.

Verification:
- `uv run pytest -q && uv run mypy src && uv run ruff check src tests`
- No new `# type: ignore` allowed.

Risk: this is the only API-breaking unit. Land it last so earlier units
aren't blocked.

---

## Unit 14 — Second-pass audit of un-covered modules

**Goal:** the explore agents focused on readers, badge/colormap plotting,
derived/compute. The following modules were not audited deeply and likely
hide their own small wins. Spend one focused session on each.

Modules to skim with the same checklist (dead code, magic numbers,
missing tests, SCHEMA drift, redundant helpers):

- `src/pypic/reconnection.py`
- `src/pypic/spectral.py`
- `src/pypic/traces/` (whole package — `_fieldline.py`,
  `_particletrace.py`, `_sampling.py`, `_tracing.py`, `_analysis.py`)
- `src/pypic/plotting/cross_section.py`
- `src/pypic/plotting/kymograph.py`
- `src/pypic/plotting/comparison.py`
- `src/pypic/plotting/scatter.py`
- `src/pypic/plotting/spectral.py`
- `src/pypic/plotting/lines.py`
- `src/pypic/plotting/annotations.py`
- `src/pypic/plotting/_resolve.py`, `_theme_io.py`

Actions:
1. For each: read top to bottom. Take notes inline in this plan as a
   sub-checklist before fixing. Convert each genuine issue into a
   numbered sub-unit (`14a`, `14b`, …) so they can be tracked.
2. Particularly look for: physics functions in `reconnection.py` /
   `spectral.py` whose formulas should appear in `equations.md` Sections 9
   and 10 — add them if missing.

Verification:
- Each sub-unit gets its own `pytest -q tests/test_<module>.py` line.

---

## Files we are NOT touching (out of scope)

- `pyproject.toml`, CI workflows, ruff/mypy configs — they're working.
- `docs/equations.md`, `docs/conventions.md`, `SCHEMA.md` — these are
  authoritative; the cleanup is making the *code* match them, not the
  other way around. Exception: if Unit 11 turns up genuine SCHEMA cruft
  (a name listed but never produced by anyone), we'll prune SCHEMA in
  that unit.
- `examples/` — separate concern.
- TASKS.md roadmap items in Phases 4–10 (relativistic functions, regrid,
  CLI, Zarr, interop, VLasiator) — those are *new feature* work, not
  cleanup.
- The "one `.py` file per simulation code" rule from CLAUDE.md is an
  aspirational shape; current readers are subdirectories, which is
  functionally equivalent. Don't flatten them.

---

## Suggested execution order

```
Unit 1  ──┐  no deps, low risk, 30-min commits
Unit 2  ──┤
Unit 3  ──┘
Unit 4  ──┐  reader refactor block
Unit 5  ──┤
Unit 6  ──┘
Unit 7  ──┐  plotting refactor block
Unit 8  ──┤
Unit 9  ──┤
Unit 10 ──┘
Unit 11 ──┐  physics correctness block
Unit 12 ──┘
Unit 14   ┄  rolling, pick subunits as time allows
Unit 13   ┄  last (API-breaking)
```

Each unit is meant to be one commit with the existing
`Assisted-by: Claude [model version]` trailer. The "Why" portion of each
commit message should reference the unit number from this file so we can
trace decisions back when we revisit in six months.

## Verification command summary

After every unit:
```sh
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest -q
```

After units that touch plotting (1, 7, 8, 9, 10, 14 sub-units):
```sh
uv run python tests/visual_plots.py
uv run python tests/visual_dipole_3d.py
```

Optional dead-code sweep, run once after Unit 14:
```sh
uv run vulture src/
```
