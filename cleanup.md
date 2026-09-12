# Code-quality cleanup — remaining phases

Checklist for the last four phases of the September 2026 code-quality audit.
Phases 0–4 landed on `main` (bug fixes, vector triplets from the field
registry, `ReaderBase`, alias/layering consolidation, the `_recipes` /
`_field_table` / `cli/` / `tests/test_plotting/` splits and the `reduce`
decomposition; see `git log 499a846..4562923`). Phase 5 landed on 2026-09-11;
Phases 6–8 are approved but unexecuted. Their line anchors were re-verified on
2026-09-10 (the pickle anchors on 2026-09-11) and Phase 5 did not touch those
files; re-check them before editing, they drift.

Working rules for every phase:

- Moves land in separate commits from edits, so `git log --follow` stays useful.
- Commit prefixes `refactor:` / `fix:` / `test:` / `docs:` / `style:` / `chore:`,
  message says why.
- Gates: `./scripts/check.sh lint`, `format`, `types`, then the targeted pytest
  selection, then the full suite. `check.sh` stops at the format gate on untracked
  `examples/*/` scripts, so run the subcommands individually when those exist.
  `docs` gate for anything that moves a module.
- Additive API only. No kwarg renames, no options dataclasses, no new core deps.
- Private tables import from their defining module (`pypic._recipes`,
  `pypic._field_table`); mypy strict rejects re-exported underscore names.

## Phase 5 — plotting dedup

Landed 2026-09-11, `git log 4562923..f3eb6b8` (11 commits). src: 15 files,
+917 / −1104. Deviations from the text below: `resolve_norm` returns one
`Normalize`, not a tuple; `finish_axes` leaves saving to the caller (inside
the theme context savefig would switch to the theme's dpi/bbox) and rounds
every plot, not only owned ones; multi-panel `ax=` takes the panels the
function would create. Four latent bugs surfaced and were fixed first, each
in its own `fix:` commit: theme colormaps ignored when `cmap=None`,
`load_theme` crashing on omitted colors, quiver `units=` coloring in code
units, and the pyvista line/trajectory divergences (see CHANGELOG).

1. - [x] **`resolve_norm` in `plotting/_colormaps.py`.** Signature
   `resolve_norm(values, *, log_scale, symlog, symmetric, vmin, vmax, linthresh)
   -> tuple[Normalize | None, float | None, float | None]`. Its ingredients
   (`is_positive_definite` :69, `symmetric_clim` :193, `_auto_linthresh` :294)
   already live there. Replaces the three copies at `slices.py:178-212`,
   `comparison.py:192-230` (note: this one takes `values_a` and a shared
   `combined_min/max`, so the helper must accept an explicit range), and
   `kymograph.py:125-160` (only log + symmetric today). Keep the
   `LogNorm`/`SymLogNorm` imports local inside the helper so matplotlib stays
   behind the extra. Test: one parametrized test over the three call sites
   asserting the same norm class and limits for the same input.

2. - [x] **`finish_axes` in `plotting/_resolve.py`.** Signature
   `finish_axes(ax, theme, *, xlabel, ylabel, aspect, title, info, step, time,
   badge, owned)`. Replaces the six epilogues: `slices.py:250-281`,
   `vectors.py:428-445` and `:657-674` (these two also carry a third title form
   at `:438` / `:667`), `scatter.py:185-195`, `kymograph.py:172-187`, and the
   four in `lines.py` (around `:149` and `:433`; grep `set_title`). The
   `add_badge` import stays local. `owned` decides whether `tight_layout` and
   `maybe_save` run (only when the function created the figure).

3. - [x] **`_theme_io.py` table-driven load/save.** `load_theme` (:125-297)
   and `save_theme` (:298-426) each hand-walk every field; `save_theme` has 53
   `lines.append` calls. Replace with one `_THEME_FIELDS: tuple[(section, key,
   kind), ...]` driving both directions via `dataclasses.fields(PlotTheme)`,
   where `kind` is `rgba | float | str | bool | rgba_list`. File is 481 lines;
   expect ~250. Test: every bundled theme round-trips `load → save → load`
   field-for-field (`tests/test_plotting/test_themes.py`). The `[webpic]`
   block must survive untouched (webpic reads it).

4. - [x] **`vectors.py` prelude.** `plot_streamlines` (:105) and `plot_quiver`
   (:457) share plane resolution, component lookup, colour resolution and the
   axes setup. Extract `_vector_prelude(...)` returning the resolved arrays,
   axes and colour spec; `_resolve_plane_components` (:32) and
   `_resolve_vector_colors` (:62) fold into it.

5. - [x] **`pyvista/_lines.py`: plurals implement, singulars wrap.**
   `add_field_line` (:75) / `add_field_lines` (:179) and `add_trajectory`
   (:300) / `add_trajectories` (:397) are near-copies. Make the singular a
   one-element call of the plural.

6. - [x] **Additive kwargs.** `ax=` on `plot_comparison` (`comparison.py:20`),
   `plot_cross_section` (`cross_section.py:20`), `plot_field_grid`
   (`panels.py:21`); `vmin`/`vmax` on `plot_quiver` (`vectors.py:457`) and
   `plot_scatter` (`scatter.py:21`); `units=` on `plot_kymograph`
   (`kymograph.py:21`); `save=` on `plot_poincare_section` (`poincare.py:18`);
   `add_contours` (`annotations.py`) returns the `QuadContourSet`. One test
   per kwarg that the passed axes is drawn on / the file is written.

7. - [x] **`PlotTheme.rcparams` read-only.** Wrap in `MappingProxyType` in
   `__post_init__` like `containers.py:218`; `styles.py:219` already copies
   before merging. Annotate as `Mapping[str, Any]`.

Verify: `uv run pytest tests/test_plotting -q` (216 collected before, 260
after; the old "143" counted functions, not parametrized cases),
`./scripts/check.sh types`, then the full suite (2901 passed on 3.14).

## Phase 6 — remaining duplication, dead weight, comments

Budget ~20 files, ~150 added / ~550 deleted. Group commits by bullet.

1. - [ ] **One nan-policy helper.** `diagnostics.py:34-77`
   (`_apply_nan_policy_single`) and `:80-128` (`_apply_nan_policy`) become one
   variadic `_apply_nan_policy(*arrays, nan_policy)`. Derive the
   `("omit", "propagate", "raise")` tuples at `diagnostics.py:47,98`,
   `reductions.py:59`, `comparison.py:65` from `get_args(NanPolicy.__value__)`;
   the CLI already pins its Literal to that alias by test.

2. - [ ] **`spectral.py` radial binning.** `power_spectrum_2d` (:175-202) and
   `power_spectrum_3d` (:282-306) repeat the `k_radial → digitize → bincount`
   block. Extract `_radial_bin(power, k_radial, n_bins)`. Parseval tests in
   `tests/test_invariants` guard it.

3. - [ ] **`derived.py` pressure decomposition.** The `bhat_1**2 * p11 + ...`
   projection is inlined at `:1685`, `:1755`, `:1829` (and the perpendicular
   companions right after each); call `parallel_pressure` (:1486) and
   `perpendicular_pressure` (:1548) instead. Keep public
   `relativistic_enthalpy` (:504); drop its registry entry in `_recipes.py:166`
   only if nothing outside tests resolves `h_rel` (grep docs/equations.md
   first: it is documented, so more likely keep both and note why).

4. - [ ] **`io/metadata.py` decoders.** `dict_to_grid` (:82),
   `dict_to_normalization` (:116), `dict_to_physics` (:212),
   `dict_to_transforms` (:261), `transforms_to_dict` (:254) have no caller
   outside `tests/test_zarr_io.py` (16 references). Delete them with their
   tests; keep `grid_to_dict` / `normalization_to_dict` / `physics_to_dict`
   (server wire format). `io/zarr.py:100-127` re-implements the
   `schema.version` major check that `decode_pypic_attrs` (:557) owns; delegate.
   `_PYPIC_ROOT_ATTR_KEYS` (`zarr.py:45`) is a hand-kept mirror of the encoder's
   keys: derive it from `encode_pypic_attrs` or delete it.

5. - [ ] **`traces/_tracing.py` closed-loop detector.** The proximity check
   inside `_trace_single_direction` (:204) and
   `_trace_single_direction_adaptive` (:256, allocation at :289-294, use at
   :323) must not disagree. Extract `_closed_loop_hit(points, arclens, i, *,
   loop_tol, loop_min_arclen) -> bool`; keep both integrator loops (they are
   numerically distinct).

6. - [ ] **Dead code.** `numerics/_step_control.py` `err_prev` params (:38,
   :105) plus the "queued PI" docstring (:5-6, :49, :66, :129, `del err_prev`
   :86); `compute.py:254-258` the `passes_geometry` branch is a no-op while
   the 3D-Cartesian raise above it stands (delete the branch and the field,
   or keep with the comment trimmed to one line); `dataset.py:1030`
   fall-through in `in_si` after the `quantity_type` branch (check whether
   any field reaches it); `openggcm/_grid.py:16-18` dangling comment above
   `OpenGGCMGrid`; `coordinates/geometry.py:139` `GEOMETRY_BY_NAME` →
   `MappingProxyType`; `fields.py:92,128` module-level asserts → one
   aggregated test in `tests/test_registry_consistency.py`. Keep
   `_stagger.py` and `openggcm/_wrn2.decode_rle` (named future consumers in
   TASKS.md Steps 35/42) with a one-line note each.

7. - [ ] **Comments that narrate history.** `readers/config.py:457`
   ("previously read"), `io/zarr.py:212` ("used to silently"),
   `io/metadata.py:502` ("previously split"), `cli/inspect.py:364`
   ("used to raise"), `io/_virtual.py:401` ("previously-saved"). Rewrite each
   to describe present behaviour or delete. Wrong docs: `dataset.py:86` and
   `containers.py:144` claim breadth-first chain resolution;
   `coordinates/transforms.py` `resolve_transform` is one-hop, so either
   implement BFS (schema.md §2 promises it) or fix the two docstrings and
   schema.md. `comparison.py:3` "only place" claim: verify against
   `reductions`/`regrid` before keeping.

8. - [ ] **Boundaries.** HTTP `kind` / `status_code` classvars leave
   `pypic/exceptions.py:41-95` for a mapping in `server/exceptions.py` (the
   core exception module should not know the wire format).
   `schema/cli.py` imports typer + rich, contradicting the "stdlib + pydantic
   only" promise in architecture.md; move it to `pypic/_schema_cli.py` and
   point the `pypic schema` sub-app at it (mirror `_codegen_cli.py`).

Verify: full suite, `./scripts/check.sh docs` (moved `schema/cli.py`).

## Phase 7 — tests and tooling

Budget ~45 files, ~600 added / ~400 deleted, four config files. Item 1 can
land first and will guard everything else.

1. - [ ] **pytest hardening in `pyproject.toml`.** Today only
   `addopts = "--doctest-modules --strict-markers -ra"`. Add
   `markers = ["slow", "integration", "fixture_data"]`, `xfail_strict = true`,
   `filterwarnings = ["error", ...]`. Run the suite with `-W error` first and
   list what fires; each ignore names the upstream issue. Expect 3–5.

2. - [ ] **Tolerances.** 280 of 794 `assert_allclose` calls carry no
   `rtol`/`atol`. Sweep file by file: `test_derived.py` 39,
   `test_ipic3d_synthetic.py` 38, `test_transforms.py` 25, `test_compute.py`
   22, `test_readers_base.py` 18, `test_traces.py` 16, `test_geometry.py` 15,
   `test_virtual_io.py` 11, `test_icechunk_io.py` 11, `test_numerics.py` 10.
   Byte-exact round-trips become `assert_array_equal`; the rest get an
   explicit `rtol`. For cancelling sums scale `atol` by the sum of magnitudes
   (the fix in `test_reduction_identities.py`, commit 7f043bc, is the model).
   Then add a ruff-free guard: a test that greps `tests/` for bare
   `assert_allclose(` and fails with the file list.

3. - [ ] **Fixtures.** Adopt `conftest.py` `cartesian_3d` / `spherical_3d`
   and `tests/_helpers.py` in `test_comparison.py` (58 inline `from_arrays`),
   `test_regrid.py` (33), `test_zarr_io.py` (27), and the 24 files building
   `GridInfo(...)` inline. Add fixtures to `conftest.py` only when a second
   file needs the same shape.

4. - [ ] **Banners and headers.** 66 `# ---` / `# ===` banner lines across
   8 test files (`test_arrow_parquet_io`, `test_traces`, `test_schema`,
   `test_comparison`, `test_regrid`, `test_derived`, `test_transforms`,
   `test_registry_consistency`): delete, use classes or module split if the
   grouping mattered. Normalize the 8 `# Claims:` headers in
   `tests/test_invariants/` to `# Claim:`; add a test that every module there
   carries both `# Source:` and `# Claim:`.

5. - [ ] **Gating.** Drop `importorskip("pydantic")` at
   `tests/test_server_exceptions.py:95` (pydantic is core). Replace the 8
   hand-rolled `pytest.skip(...)` in `test_examples_smoke.py:167-246` with one
   `fixture_data` marker plus a `skipif` on missing `tests/data`. Delete
   `tests/data/.DS_Store` and add it to `.gitignore` if not already.

6. - [ ] **Doctests for non-exempt gaps.** `dataset.py`: `transform_to`
   (:374), `sel` (:750), `in_si` (:987), `in_units` (:1040), `isel` (:1069),
   `where` (:1093), `reduce` (:1117). `compute.py` has 4 doctest lines over
   18 defs; cover `compute_field`, `available_quantities`,
   `field_dependencies`, `register_recipe`, `display_unit_factor`,
   `field_si_factor`. `codegen.py` (4 public exports, 0 doctests) and
   `traces/_sampling.py` (0 doctests) get one each on synthetic arrays.

7. - [ ] **Coverage.** `pytest-cov` in the dev group, `[tool.coverage.run]`
   with `source = ["pypic"]`, `fail_under` = measured minus 2, a `cov`
   subcommand in `scripts/check.sh`, one CI job. No upload service.

8. - [ ] **Pre-commit.** `.pre-commit-config.yaml` running
   `scripts/check.sh lint` and `format` only (local hooks, no mirrors, so the
   pinned ruff in `uv.lock` is the one that runs).

9. - [ ] **Ruff families, one commit each.** PLW, BLE, SLF (with a
   per-file-ignores list that doubles as the seam inventory), PTH, PERF,
   FURB, C4, PIE, RET, LOG. Known one-offs: PLW1510 at `cli/plot.py:101`
   (`subprocess.run` without `check=`), S608 at `io/_duckdb.py:92` (the path
   is escaped; add a same-line reason). Not PLR / TRY / EM / FBT / COM /
   PLC0415 / ISC.

10. - [ ] **CI matrix (optional).** Currently 3.13 + 3.14 on ubuntu. macOS
    adds little (dev platform); Windows would add h5py path signal if wanted.

Verify: full `./scripts/check.sh` after every item; the invariant is
2854+ tests green on 3.13 and 3.14, mypy strict clean, doctests green.

## Phase 8 — pickling, typing, import time

Budget ~14 files, ~130 added / ~45 deleted, one new module. Independent of
Phases 6 and 7.

1. - [x] **Pickle and deepcopy** (a `fix:` commit). A frozen dataclass that
   stores a `MappingProxyType` can be neither pickled nor deep-copied:
   confirmed for `PhysicsParams`, `SimulationConfig`, `TabularData`,
   `FieldLine` and `ParticleTrace`, and `FieldDataset` inherits it through
   its `PhysicsParams`. Measured 2026-09-11, that fails stdlib `pickle` and
   everything built on it (process pools from `concurrent.futures` or
   `multiprocessing`, sending or returning a dataset; `joblib.dump`, which
   `joblib.Memory` uses), plus `copy.deepcopy` and `dataclasses.asdict`.
   cloudpickle paths already work: `joblib.Parallel` with its default loky
   backend, and dask (inferred, not run). Stored proxies: `units.py:677`
   (`PhysicsParams.extra`), `containers.py:102,114` (`StaggerInfo`,
   conditional), `:218-220` (`SimulationConfig`), `:283-284`
   (`TabularData`), `:427` (`ParticleData`), `traces/_fieldline.py:96-97`,
   `traces/_particletrace.py:98-99`, `traces/_poincare.py:278`,
   `ipic3d/_config.py:120`, and the proxies `openggcm/_grid.py:133-134` hands
   to `OpenGGCMGrid` from outside.

   Fix the type, not its holders. A stdlib-only leaf module (e.g.
   `pypic/_pickling.py`) registers one reducer,
   `copyreg.pickle(MappingProxyType, lambda m: (_mappingproxy, (dict(m),)))`,
   where `_mappingproxy` is a named module-level function returning
   `MappingProxyType(mapping)`: pickle cannot reference the type itself,
   which is not importable as `builtins.mappingproxy`. `pypic/__init__.py`
   imports the module first, for its side effect; the package init runs
   before any submodule, so every class is covered, and the import stays
   eager when item 3 lands. `copy.deepcopy` and the process-pool pickler
   read the same `copyreg.dispatch_table`: in the measurement every failing
   path above passed, round-trips compared equal and the fields stayed
   read-only. The registration is process-wide, so after `import pypic` any
   `mappingproxy` pickles as a snapshot instead of raising; nothing that
   pickles today changes. Delete `PlotTheme.__getstate__` / `__setstate__`
   (`plotting/styles.py`), which the reducer makes redundant; the theme
   clone test keeps guarding it. Test: one aggregated invariant that every
   class above round-trips through `pickle` and `copy.deepcopy` equal and
   still read-only, with one `FieldDataset` sent through
   `multiprocessing.reduction.ForkingPickler`, the pickler process pools
   use, so no worker process is needed.

2. - [ ] **Typing.** `containers.py:333,410` and `dataset.py:1093` bare
   `np.ndarray` → `IntArray` / `FloatArray` / `BoolArray` from `pypic.types`;
   `containers.py:200,210,266` annotate `Mapping[...]` to match the
   `MappingProxyType` substitution in `__post_init__` (`units.py` is the
   model); `:198` `default_factory=PhysicsParams`; the four
   `type: ignore[type-arg]` in `traces/_tracing.py:545,671,867,1075` →
   `dict[str, Any]`; named constants for magic indices at
   `batsrus/_hdf5.py:63-67` (`ipm[2]`, `ipm[1]`, `rpm[0]`),
   `ipic3d/_conserved.py` column indices, `openggcm/_wrn2.py` header offsets,
   and `ipic3d/_config.py` `reference_density`.

3. - [ ] **Import time.** `import pypic` takes 0.73 s warm (best of 7 fresh
   interpreters, 2026-09-11). The avoidable part is `scipy.interpolate`, which
   `regrid.py:37` and `traces/_tracing.py:19` import at module scope and
   `__init__.py` reaches through `comparison` (:6), `reconnection` (:174,
   imports `traces`), `regrid` (:176) and `traces` (:180). Fix: export those
   four modules' 21 names lazily through a PEP 562 `__getattr__` and
   `__dir__` in `__init__.py`, driven by one name → module table, with the
   real imports kept under `if TYPE_CHECKING:` so mypy and IDEs still see
   the types. The four modules keep their module-scope imports; add the
   lazy table to architecture.md's import rule as its second sanctioned
   case. Stubbing all four out of `sys.modules` measured 0.52 s with
   `scipy.interpolate` never loaded; stubbing only `comparison` and `regrid`
   measured no saving, because `traces` loads it anyway. `codegen` stays
   eager: `__init__.py:177` imports `pypic.schema` (pydantic) directly, and
   deferring `codegen` measured no saving. Guard: a subprocess test that
   `import pypic` leaves `scipy.interpolate` out of `sys.modules`, rather
   than a wall-clock ceiling that would flake on CI. Target below 0.6 s.

Verify: `./scripts/check.sh types`, the full suite, and
`uv run python -X importtime -c "import pypic"` before and after item 3
(`scipy.interpolate` must drop out).

## Non-goals

No 2D operator support (TASKS.md Step 50). No destaggering of BATSRUS or
OpenGGCM. No new core dependencies. No plotting rewrite around option
objects, no public kwarg or function renames. No schema semantics or JSON
artifact changes. No `derived.py` or `schema/_models.py` split. No big-bang
commits.
