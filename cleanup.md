# Code-quality cleanup — remaining phases

Checklist for the last four phases of the September 2026 code-quality audit.
Phases 0–4 landed on `main` (bug fixes, vector triplets from the field
registry, `ReaderBase`, alias/layering consolidation, the `_recipes` /
`_field_table` / `cli/` / `tests/test_plotting/` splits and the `reduce`
decomposition; see `git log 499a846..4562923`). Phase 5 landed on 2026-09-11,
Phases 8 and 6 on 2026-09-12. Phases 7 and 9 remain; their anchors date from
2026-09-10. Re-check before editing, they drift — and check the premise, not
just the line number: four of Phase 6's proposed deletions turned out to be
load-bearing.

Working rules for every phase:

- Moves land in separate commits from edits, so `git log --follow` stays useful.
- Commit prefixes `refactor:` / `fix:` / `test:` / `docs:` / `style:` / `chore:`,
  message says why.
- Gates: `./scripts/check.sh lint`, `format`, `types`, then the targeted pytest
  selection, then the full suite. `check.sh` stops at the format gate on untracked
  `examples/*/` scripts, so run the subcommands individually when those exist.
  `docs` gate for anything that moves a module.
- Additive API by default: no kwarg renames, no options dataclasses, no new
  core deps. The few removals of public surface are named per item and each
  carries a CHANGELOG line.
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

Landed 2026-09-12, `git log 9075636..a89c1f2` (12 commits). src: 24 files,
+420 / −430. Deviations, all recorded per item below: item 4 rewrote the
orphaned decoders' tests to pin the surviving *encoders* rather than
deleting them (`grid_to_dict` / `physics_to_dict` are the server wire
format and had no other coverage); item 5 took the documented
alternative — align the detectors' gate and cross-reference them — rather
than one shared helper, because every shape of that helper either copies
on the scalar hot loop or leaves both callers doing the windowing; item 7
left `schema/_models.py` alone, since those docstrings feed the generated
JSON Schema descriptions and regenerating the artifact is a phase
non-goal. Two stray defects surfaced and were fixed in their own commits:
two E501s and one F401 that the `lint` gate does not report (see Phase 7
item 1).

Budget was ~18 files, ~200 added / ~420 deleted. Group commits by bullet.
Anchors re-verified 2026-09-12 against `9075636`. Several of the
original deletions turned out to be live code — `h_rel`,
`passes_geometry`, the `in_si` fall-through, three of the five
`io/metadata` decoders, and `decode_rle`. Each stays struck below with
the reason, so a later reader does not re-propose it. Three items remove
public surface (item 6's `err_prev`, item 6's `GEOMETRY_BY_NAME`
annotation, item 8's `pypic.schema.cli` path) against the additive-API
rule above — each is defensible at 0.1.x, and each wants a CHANGELOG
line rather than a silent `refactor:`.

1. - [x] **One nan-policy helper.** `diagnostics.py:34-77`
   (`_apply_nan_policy_single`) and `:80-128` (`_apply_nan_policy`) are
   line-for-line identical apart from mask construction
   (`np.isnan(field)` :54 versus the joint `np.isnan(computed) |
   np.isnan(reference)` :105, deliberate so relative norms compare one
   point set) and return arity; both error and both warning strings
   already match. One variadic `_apply_nan_policy(*arrays, nan_policy)`
   over `np.logical_or.reduce` covers them. It needs `@overload` on
   arity to stay mypy-strict clean: `l2_relative_error` (:172) and
   `linf_error` (:221) destructure a 2-tuple, and the five single-array
   callers (:325, :439, :531, :570, :606) grow a `(masked,) = ...`.
   Derive the `("omit", "propagate", "raise")` tuples at
   `diagnostics.py:47,98`, `reductions.py:59` and `comparison.py:65`
   from `get_args(NanPolicy.__value__)` — `reductions.py:58` already
   does exactly that for `Reduction`. Two constraints: the CLI Literal
   at `cli/_options.py:57-60` stays hand-written (typer introspects it,
   and `tests/test_cli.py:337` pins it to the alias), and
   `reductions.py:41` imports `NanPolicy` only under `TYPE_CHECKING`,
   so deriving adds a runtime `reductions → diagnostics` edge — safe
   (no cycle, and `__init__.py:106` already imports diagnostics
   eagerly, so `import pypic` does not grow). Keep the check at
   `comparison.py:325-327`: its comment says it duplicates the inner
   one deliberately, to fail before the expensive regrid.

2. - [x] **`spectral.py` radial binning.** `power_spectrum_2d`
   (:177-202) and `power_spectrum_3d` (:284-306) are token-identical
   apart from one comment word — same `linspace(0, k_max, n_bins + 1)`
   edges, same `digitize` and `clip`, same count-average, same
   empty-bin NaN-then-filter, same midpoint `k_centers`, and no
   annulus-versus-shell weighting difference (geometric weighting lives
   in the isotropy of the k-grid sampling). Extract
   `_radial_bin(power, k_radial, n_bins)`; the window, FFT and
   normalization above each block legitimately differ and stay put.
   Two corrections to the original text: there is no `bincount`
   anywhere — both blocks use `np.add.at`, so keep it or justify the
   `counts` dtype shift and the mandatory `minlength` separately — and
   the Parseval tests do **not** guard this.
   `tests/test_invariants/test_parseval_identity.py` is 1-D only, and
   the 2-D/3-D coverage in `tests/test_spectral.py` is shape and
   sanity, so an off-by-one in the helper's `digitize` or `k_centers`
   would pass the suite. Land a 2-D absolute-scale or Parseval
   assertion first, then refactor under it.

3. - [x] **`derived.py` pressure decomposition.** The
   `bhat_1**2 * p11 + ...` projection is inlined three times, but only
   two of them are worth delegating and only two carry a perpendicular
   companion. `agyrotropy` (:1683-1689) and `aunai_nongyrotropy`
   (:1752-1761) reproduce `parallel_pressure` (:1486) — and, at the
   second site, `perpendicular_pressure` (:1548) — token for token, and
   neither touches `bhat_*` again afterwards, so delegating drops a
   whole `_unit_vector` call at each. Delegate `p_par` only and keep
   `p_perp = (trace_p - p_par) / 2.0` inline: `perpendicular_pressure`
   re-calls `parallel_pressure` internally, so delegating both would
   recompute `bhat`. `scudder_agyrotropy` (:1826-1834) stays inline —
   it reads `bhat_1/2/3` heavily at :1836-1851, so delegating there
   trades the dedup for a second `_unit_vector` over the full grid.

   `h_rel` stays, both the function and the registry entry. The
   original text made dropping `_recipes.py:165` conditional on nothing
   outside tests resolving it; the condition fails. It is documented at
   `docs/equations.md:101` with footnote `:116` and again at
   `docs/schema.md:1090`, exported from `__init__.py:94`, aliased from
   `_aliases.py:103`, and pinned by
   `tests/test_registry_consistency.py:115`. That
   `relativistic_enthalpy` (:504) is a one-line forward to
   `enthalpy(..., c=c)` is the point of it — a named entry point for a
   documented quantity, not duplication.

4. - [x] **`io/metadata.py` and `io/zarr.py` attrs plumbing.** Three of
   the five decoders the original text called dead are live on the
   `from_zarr` path: `dict_to_normalization` (:116) is called from
   `decode_pypic_attrs` at `metadata.py:615`, `dict_to_transforms`
   (:261) at `:644`, and `transforms_to_dict` (:254) at `:793` inside
   the encoder. Only `dict_to_grid` (:82) and `dict_to_physics` (:212)
   are orphaned, because the v1.0 reshape replaced them with
   `_attrs_to_grid` (:691) and `_attrs_to_physics` (:826). Delete those
   two with their `tests/test_zarr_io.py` references; neither is
   exported from `pypic.io` nor reachable from the docs. Keep
   `grid_to_dict` / `normalization_to_dict` / `physics_to_dict` — all
   three are the server wire format (`server/arrow.py:28-29,271-272`,
   `server/routes.py:25-27,85-88`).

   `_PYPIC_ROOT_ATTR_KEYS` (`zarr.py:45`) and `_strip_pypic_attrs`
   (:59) are dead, not stale. `_open_store` returns
   `tree["fields"].to_dataset()` (:127), whose attrs are the `/fields`
   group's own; the section keys live on the root group and reach
   `_ds_to_field_dataset` as the separate `root_attrs` dict. Measured
   2026-09-12: a store carrying `boundary_conditions`, `coordinates`,
   `grid`, `metadata`, `model`, `normalization`, `physics`, `schema`
   and `species` on root has `[]` on `/fields` and `[]` on the returned
   dataset. Delete both and pin the no-leak guarantee with a
   `from_zarr(...).xr.attrs` assertion, instead of a hand-kept mirror
   of the encoder's keys that has already drifted six keys out of date.

   `zarr.py:110-123` and `decode_pypic_attrs` (`metadata.py:596-609`)
   run the same pair of `schema.version` checks — strict equality
   against `SCHEMA_VERSION`, not the major-component check the original
   text described. Delegating is right, but it must keep the
   `source_label` prefix in the message and keep failing ahead of the
   `"fields" not in tree.children` check at :124.

5. - [x] **`traces/_tracing.py` closed-loop detector.** The original
   text put one detector in `_trace_single_direction` (:204); there is
   none there — that path is fixed-step classical RK4, and
   `trace_field_line` (:589) never exposes loop detection at all. The
   two detectors are both on adaptive paths: scalar at :321-336
   (monotone `arclen` prefix allocated :289-297, `searchsorted`-bounded
   tail scan) and batched at :462-501 (per-seed `arclen` allocated
   :409-416, rectangular `(M, max_n)` eligibility mask). Tolerance
   source, arc-length cutoff, `<=` eligibility, past-only scan and
   trigger all agree today; the one divergence is the cheap gate,
   `j_end > 0` versus `cutoffs.max() > 0.0`, which disagree when
   `arclen == loop_min_arclen` exactly — the scalar path tests the seed
   point, the batched path skips the scan.

   So the proposed scalar-bool `_closed_loop_hit(...)` is the wrong
   shape: it fits only the scalar site, where it removes no
   duplication, and routing the batched path through it needs a
   per-seed Python loop that undoes the vectorization the comment at
   :465-472 exists to justify. Either extract a vectorized
   `_closed_loop_hits(points, arclens, cur_idx, *, loop_tol,
   loop_min_arclen) -> BoolArray` that the scalar path calls with
   `M == 1`, resolving the gate explicitly, or drop the item and leave
   a one-line note on each detector pointing at the other. Step 44f
   (`return_endpoints_only`) reworks this function and would rather
   inherit one detector than two.

6. - [x] **Dead code.** Two of the six original deletions here are live
   code, one is a type change rather than dead code, and the closing
   keep-note was right about the outcome but wrong about the reason.
   All four stay recorded so they are not re-proposed.

   *Delete* `numerics/_step_control.py` `err_prev`: params :38 and
   :105, prose :5-6, :49-50, :66-70 and :129-132, and **two**
   `del err_prev` — :86 and :151, the second missed the first time. No
   caller passes it (`traces/_tracing.py:314`, :446-448) and
   `tests/test_numerics.py:291` asserts only that it is ignored, which
   is worse than not offering the kwarg. It takes three more edits with
   it: that test, `docs/api/numerics.md:9-11`, and
   `docs/references.bib:265` — removing the last `[@Gustafsson1988]`
   citations orphans the entry and fails
   `tests/test_bibliography.py::test_no_orphan_bib_entries`. While
   there, fix `TASKS.md:37` and Step 44e: both describe "PI step
   control" that the shipped elementary (I) controller does not
   implement.

   *Delete* `openggcm/_grid.py:16-18`, a comment whose constant is
   gone; `parse_grid_file`'s docstring (:75-78) already carries the
   same facts.

   *Delete* the `fields.py:92-94` assert — `tests/test_fields.py:503-505`
   is already the identical set equality. The `:128-130` one needs its
   counterpart tightened first: `test_fields.py:425-429` checks one
   direction only, so it would miss an orphan key left in
   `_QUANTITY_DIMENSIONS` after a `QuantityType` member is renamed.
   Both belong in `TestQuantityTypeCoverage` (`test_fields.py:319`),
   not in `tests/test_registry_consistency.py`, which never references
   either table.

   *Type change, not dead code:* `coordinates/geometry.py:139`
   `GEOMETRY_BY_NAME` → `MappingProxyType`. All nine callers read
   (`io/_virtual.py:73`, `io/metadata.py:85,721`,
   `readers/_simple.py:117`, `readers/config.py:280`,
   `readers/batsrus/_reader.py:211,288,335`,
   `readers/batsrus/_config.py:244`), so the swap is safe in-repo, but
   it narrows a `coordinates.__all__` export and its annotation.

   *Keep* `compute.py:253-257`. The `passes_geometry` branch is
   value-wise inert behind the Cartesian raise, but it is not free to
   delete: `codegen.py:53` serializes the field as `"passesGeometry"`
   into the bundle webpic consumes, with no test pinning that key, so
   dropping it is a cross-repo break invisible to CI, and
   `tests/test_registry_consistency.py:538-545` validates the call
   shape the branch produces. Deferred Step 31 is the work that makes
   it live. Trim the comment to one line; leave branch and field.

   *Keep* `dataset.py:1030`. The fall-through is the registry-resolution
   path for fields with no `quantity_type` attr, which
   `from_arrays(..., strict_fields=False)` produces —
   `readers/_base.py:118`, `regrid.py:324`, `comparison.py:574` and
   `cli/convert.py:125` all take it, and `open_virtual(...).in_si(...)`
   resolves only through it, since VirtualiZarr hands HDF5 attrs
   straight through. The duplication actually present is the
   three-line factor-and-`length_axes` tail repeated at :1026-1029 and
   :1035-1038; collapse that instead, and document the `ValueError`
   that `field_si_factor` raises in the `Raises` section `in_si` lacks.

   *Keep* `_stagger.py` and `openggcm/_wrn2.py` `decode_rle`, and add
   no notes — both already carry one (`_stagger.py:19-20`,
   `_wrn2.py:46-48`). `decode_rle` is not unwired:
   `decompress_field` calls it at :158 and :164, and it is the readable
   oracle that `decompress_field_vectorized` (:261) is cross-validated
   against in `tests/test_openggcm_wrn2.py`. Its citation of TASKS.md
   Steps 35/42 was wrong — those name `_stagger.py`, whose only
   importer is `tests/test_destagger.py:8`.

7. - [x] **Comments that narrate history.** Four of the five hold, at
   slightly different lines: `readers/config.py:457-458` (a migration
   note on a private helper; :455-456 above it describes present
   behaviour and stays), `io/zarr.py:211-214` ("the writer used to
   silently flatten...", which doubles as the guard's rationale — recast
   it in the conditional rather than delete), `io/metadata.py:501-503`
   (only the parenthetical narrates the old on-disk layout), and
   `cli/inspect.py:363-364` (keep the guard's rationale at :361-363,
   drop the trailing "used to raise TypeError here and take the whole
   command down").
   `io/_virtual.py:401` is a false positive and stays: "previously-saved
   containers" and "earlier commits" are live icechunk repo state, not
   pypic's own history.

   The original sweep was incomplete. Same class, same treatment:
   `plotting/_format.py:3` ("previously duplicated across..."),
   `regrid.py:349` ("matches the historical behavior", in a *public*
   docstring), `server/app.py:96-98` ("stays back-compat with the
   previous ... shape"), `exceptions.py:58-59` ("the previous workaround
   silently mangled..."). Forward references are the same rule read the
   other way: `comparison.py:131-134` says the helper "forwards" an
   `epoch` kwarg that `transform_to` does not have — a false claim, not
   merely a roadmap note — plus `schema/_models.py:276,317,1402` ("a
   future v1.1 may...").

   `comparison.py:3` overclaims. `reductions.py:77` runs integrate,
   mean, median, std and argmax on `FieldDataset`-held arrays, and its
   own docstring (:153) points at
   `pypic.diagnostics.l2_relative_error` as a sibling. Narrow the
   sentence to what holds: comparison.py is the only place the pure
   `pypic.diagnostics` norms are wrapped as a `FieldDataset`-level
   public API.

   The breadth-first claim is not a comment fix and leaves this phase.
   `dataset.py:83-86` and `containers.py:144-146` promise BFS chain
   resolution; `resolve_transform` (`coordinates/transforms.py:262-325`)
   is hard-capped at two edges, tie-breaks on dict insertion order, and
   has no cycle detection — and the same promise is repeated in
   `docs/schema.md:493-496`, `schema/_export.py:197` and the generated
   `simulation.schema.v1.0.json`. Correcting the docs would mean
   regenerating the JSON Schema, which this phase's non-goals forbid,
   and implementing BFS is feature work. Tracked as TASKS.md Step 15b.

8. - [x] **Boundaries.** HTTP `kind` / `status_code` classvars leave
   `pypic/exceptions.py` (base :48-49, then :75-76, :87-88, :94-95,
   :106-107) for a mapping in `server/exceptions.py`, which already
   exists and already defines the sixth carrier, `ValidationFailedError`
   (:39-51). Only two readers, `server/app.py:103-106` and
   `server/stream.py:97-98`, and `exceptions.py:11-17` already concedes
   the classvars are inert for everyone else. One trap: the base
   class's `internal` / 500 is a deliberate inherited fallback
   (`exceptions.py:41-46`) and `ErrorFrame.kind`
   (`server/protocol.py:195-202`) is a closed six-value Literal, so a
   plain dict lookup would emit an invalid frame for an unmapped
   subclass. Walk the MRO or carry a default, and add the
   `PypicError.__subclasses__()` exhaustiveness test the suite lacks —
   `tests/test_server_exceptions.py:35-60` is a hardcoded five-tuple
   parametrize that a new subclass cannot fail.

   `schema/cli.py` imports typer at module scope (:27) and rich lazily
   (:326), so the "only stdlib and pydantic" promise at
   `docs/architecture.md:75-77` and `schema/__init__.py:3-5` is false
   as written — lifting the subpackage would carry a typer dependency.
   It is not re-exported from `schema/__init__.py` (which is what keeps
   `import pypic.schema` typer-free); the sub-app is mounted from
   `cli/__init__.py:14,79`. Move it to `pypic/_schema_cli.py`,
   mirroring `_codegen_cli.py`, which already cites the split it copies
   (`codegen.py:10`, `_codegen_cli.py:3,45-46`). `pypic.schema.cli` is
   an importable path today, so this is a removal — CHANGELOG line.

Verify: full suite, `./scripts/check.sh docs` (moved `schema/cli.py`).
Items 1, 2 and 4 want their guard test landed before the refactor, not
after.

## Phase 7 — tests and tooling

Landed 2026-09-12. Budget was ~45 files, ~600 added / ~400 deleted, four
config files. Item 1 landed first and guarded the rest.

Three anchors were measured wrong in the text below and are corrected
per item: the bare-`assert_allclose` count (280 by grep, 194 by parse —
a grep reports a multi-line call as bare when its `rtol` sits on a later
line), item 3's fixture-adoption premise (the three named files already
import `tests/_helpers`), and item 5's `test_examples_smoke.py` premise
(that file is `--sim-data`-gated and never reads `tests/data`). Item 1's
lint-gate diagnosis was also wrong in an interesting way — see below.

1. - [x] **pytest hardening in `pyproject.toml`.** Today only
   `addopts = "--doctest-modules --strict-markers -ra"`. Add
   `markers = ["slow", "integration", "fixture_data"]`, `xfail_strict = true`,
   `filterwarnings = ["error", ...]`. Run the suite with `-W error` first and
   list what fires; each ignore names the upstream issue. Expect 3–5.

   While here, fix the **lint gate's blind spot**, measured 2026-09-12:
   `scripts/check.sh lint` runs `ruff check src tests scripts benchmarks
   examples` and reported "All checks passed!" on a tree where
   `ruff check src` alone reported two E501s in `diagnostics.py` and an
   F401 in `io/zarr.py` — all three real, all three introduced and then
   committed during Phase 6 before `ruff format --check` caught the
   E501s. Adding paths suppressed the findings; `--no-cache` did not
   change it, and no nested ruff config exists under the extra
   directories. So the gate CI trusts under-reports. Reproduce, then
   either pin the invocation (per-path loop, or `--no-cache`) or file it
   upstream with the ruff version from `uv.lock`. Until it is fixed,
   `format` is the gate that actually catches long lines.

   Landed. Only **two** warning sources fired under `-W error`, not the
   3-5 expected: zarr-python's consolidated-metadata warning (71 of the
   73, and deliberate on our side — schema.md § 4.2 makes consolidated
   metadata part of the layout), which takes the one suite-wide ignore;
   and pypic's own NaN-omit warning, which reaches exactly one test, the
   one checking that `-q` routes it to logging. That test opts out
   locally with `@pytest.mark.filterwarnings`, so an unexpected NaN
   warning anywhere else still fails. `xfail_strict` went in (no xfail
   exists yet; it is a guard on the first one). The three **markers did
   not**: `--strict-markers` already fails an undeclared marker at first
   use, which is the moment to declare it, and none of `slow` /
   `integration` / `fixture_data` has a user — item 5's `fixture_data`
   candidate turned out not to be one. Verified on 3.13 and 3.14: 2912
   passed, 2 warnings, both the allowed ones.

   The lint-gate diagnosis above is wrong. It is not that adding paths
   suppresses findings — it is that ruff 0.15.5's **multi-root walk is
   nondeterministic**. Same tree, same command, repeated: 264 files
   walked, or 253, or 69 with `src/` dropped whole. A planted F401 in
   `src/` was caught **1 run in 10**. `--no-cache` makes it worse, not
   better: it pins the walk to the 69-file truncation. Some runs also
   ignored `.gitignore` and linted `examples/*/` scratch directories.
   Fix: give ruff a single root (`ruff check .`), which is deterministic
   across 8 runs, catches the planted F401 8/8, and walks a strict
   superset of the five paths — the only extra file is `pyproject.toml`.
   mypy keeps an explicit list; it has no `.gitignore` awareness. Worth
   reporting upstream, with this reproduction.

2. - [x] **Tolerances.** 280 of 794 `assert_allclose` calls carry no
   `rtol`/`atol`. Sweep file by file: `test_derived.py` 39,
   `test_ipic3d_synthetic.py` 38, `test_transforms.py` 25, `test_compute.py`
   22, `test_readers_base.py` 18, `test_traces.py` 16, `test_geometry.py` 15,
   `test_virtual_io.py` 11, `test_icechunk_io.py` 11, `test_numerics.py` 10.
   Byte-exact round-trips become `assert_array_equal`; the rest get an
   explicit `rtol`. For cancelling sums scale `atol` by the sum of magnitudes
   (the fix in `test_reduction_identities.py`, commit 7f043bc, is the model).
   Then add a ruff-free guard: a test that greps `tests/` for bare
   `assert_allclose(` and fails with the file list.

   Landed. **194**, not 280 — the count above is grep-derived and a grep
   calls a multi-line invocation bare when its `rtol` sits on a later
   line. `test_derived.py`'s 39 and `test_compute.py`'s 22 were almost
   entirely that: 0 and 5 respectively once parsed. The guard is
   therefore AST-based too, in the new `tests/test_suite_conventions.py`,
   and was checked against a planted multi-line bare call that a grep
   guard would have passed.
   183 of the 194 were never approximations — an I/O round-trip, a
   second reader path over the same bytes, an alias resolving to the
   same array, a signed-permutation rotation, a literal read back out of
   a fixture — and became `assert_array_equal`. Every one passed first
   run; no exactness claim had to be walked back. The remaining 11 got
   `rtol=1e-15`, a tightening from numpy's 1e-7 default. No cancelling
   sum needed the `atol` treatment the text anticipated. Two findings:
   the iPIC3D `rho_c`-sums-over-species tests compare against an
   identically-zero field, where a bare `assert_allclose` asserts
   nothing at all (rtol scales the desired value); and
   `test_numerics.py`'s Dormand-Prince docstring already promised the
   batched and scalar kernels agree bit-for-bit while the assertion
   claimed only "close".

3. - [x] **Fixtures.** Adopt `conftest.py` `cartesian_3d` / `spherical_3d`
   and `tests/_helpers.py` in `test_comparison.py` (58 inline `from_arrays`),
   `test_regrid.py` (33), `test_zarr_io.py` (27), and the 24 files building
   `GridInfo(...)` inline. Add fixtures to `conftest.py` only when a second
   file needs the same shape.

   Premise stale on the first half: `test_comparison.py`, `test_regrid.py`
   and `test_zarr_io.py` already import `tests/_helpers`, and their
   `from_arrays` calls are not boilerplate — each carries the analytic
   field that test is about (`np.sin(x)`, `x + 2y + offset`), which is
   exactly what a shared random-normal fixture would destroy.
   `cartesian_3d` is an 8x6x4 standard-normal dataset; a comparison or
   regrid test needs known interpolants, not noise. Nothing adopted
   there, and no new `conftest.py` fixture earned a second caller.
   The real residual was the `GridInfo(...)` half, measured by AST: 93
   inline constructions, of which 58 pass `geometry`, `dt`, `boundary`
   or `surviving_axes` and cannot be expressed by `make_uniform_grid`.
   31 could and now do (the other 4 are inside `_helpers.py` itself,
   where routing them through a sibling helper adds indirection for no
   reader). Seven of the 31 also carried a function-local `from
   pypic.grid import GridInfo` purely to build a unit grid. Net -77
   lines, 405 tests unchanged.

4. - [x] **Banners and headers.** 66 `# ---` / `# ===` banner lines across
   8 test files (`test_arrow_parquet_io`, `test_traces`, `test_schema`,
   `test_comparison`, `test_regrid`, `test_derived`, `test_transforms`,
   `test_registry_consistency`): delete, use classes or module split if the
   grouping mattered. Normalize the 8 `# Claims:` headers in
   `tests/test_invariants/` to `# Claim:`; add a test that every module there
   carries both `# Source:` and `# Claim:`.

   Landed. 35 of the 36 banner blocks were pure restatement of the class
   or docstring directly beneath (`# compare_fields` over
   `class TestCompareFields`; `# Test 2 — per-species prefixes resolve
   for both static (s0/s1) and dynamic (s5+)` over a docstring saying
   that), and three named work batches rather than code, which CLAUDE.md
   rules out on its own. One survivor labels a module-level data table
   and carries a schema.md pointer found nowhere else; it keeps the text
   and loses the rules. No grouping was load-bearing, so the class-wrap
   escape hatch went unused: 678 tests over the eight files, before and
   after.
   The header half was worse than the plural-label count suggested:
   eight *further* modules had no `# Claim:` label at all, their claim
   sitting as unlabelled prose inside the `# Source:` block. Those are
   split apart; two (`test_norm_roundtrip`, `test_compute_plasma_params`)
   had no claim written anywhere and got one derived from their
   assertions. The split also caught `test_rotation_roundtrip` promising
   "two consequences" of orthogonality while testing three.

5. - [x] **Gating.** Drop `importorskip("pydantic")` at
   `tests/test_server_exceptions.py:95` (pydantic is core). Replace the 8
   hand-rolled `pytest.skip(...)` in `test_examples_smoke.py:167-246` with one
   `fixture_data` marker plus a `skipif` on missing `tests/data`. Delete
   `tests/data/.DS_Store` and add it to `.gitignore` if not already.

   The `importorskip` is gone (it had drifted to :146). The `.DS_Store`
   half was already done: both files are untracked and `.gitignore:109`
   already re-excludes `tests/data/.DS_Store` past the `!tests/data/**`
   negation. Local copies deleted.
   The `test_examples_smoke.py` premise **fails**. That module collects
   zero tests without `--sim-data` and never reads `tests/data` — it
   scans a user-supplied directory of real simulation output. Its eight
   skips are per-discovered-directory dispatch ("no reader for model in
   X", "not an MHDUCLA simulation", "no B_1 field"), which a single
   `fixture_data` marker plus a `skipif` on a path cannot express, and
   replacing them would lose the dispatch while gating on a directory
   the file does not use. Left alone; this is also why no `fixture_data`
   marker was declared in item 1.

6. - [x] **Doctests for non-exempt gaps.** `dataset.py`: `transform_to`
   (:374), `sel` (:750), `in_si` (:987), `in_units` (:1040), `isel` (:1069),
   `where` (:1093), `reduce` (:1117). `compute.py` has 4 doctest lines over
   18 defs; cover `compute_field`, `available_quantities`,
   `field_dependencies`, `register_recipe`, `display_unit_factor`,
   `field_si_factor`. `codegen.py` (4 public exports, 0 doctests) and
   `traces/_sampling.py` (0 doctests) get one each on synthetic arrays.

   Landed: 15 new Examples blocks. `register_recipe` already had one, so
   `compute.py` got five not six. `codegen.py`'s four double as the first
   pin on the camelCase wire keys webpic reads — Phase 6 item 6 found
   `passesGeometry` crossing repos with no test behind it, and
   `export_recipes`'s doctest now asserts it. `traces/_sampling.py` got
   two (`sample_field`, `sample_fields`), both showing the
   out-of-domain-returns-NaN behaviour that the prose only asserts.

7. - [x] **Coverage.** `pytest-cov` in the dev group, `[tool.coverage.run]`
   with `source = ["pypic"]`, `fail_under` = measured minus 2, a `cov`
   subcommand in `scripts/check.sh`, one CI job. No upload service.

   Measured 88.19%; `fail_under = 86`. Two `exclude_also` entries for
   code that cannot execute: `if TYPE_CHECKING:` blocks, and the
   `assert_never` guard architecture.md mandates after an exhaustive
   enum match (reaching it means the enum grew a member, which mypy
   rejects first). `cov` sits outside the default `check.sh` chain — it
   re-runs the suite `test` already ran, instrumented, for ~60% more
   wall clock — and is one CI job rather than a flag on the 3.13/3.14
   matrix, which would buy the same number twice.

8. - [x] **Pre-commit.** `.pre-commit-config.yaml` running
   `scripts/check.sh lint` and `format` only (local hooks, no mirrors, so the
   pinned ruff in `uv.lock` is the one that runs).

   Landed as specified, plus a CONTRIBUTING.md pointer. Verified with
   `uvx pre-commit run --all-files`; both hooks pass. `pre-commit` is
   deliberately not a dev-group dependency — CI gates on `check.sh`
   directly, so the hook is a local convenience, not a build input.

9. - [ ] **Ruff families, one commit each.** PLW, BLE, SLF (with a
   per-file-ignores list that doubles as the seam inventory), PTH, PERF,
   FURB, C4, PIE, RET, LOG. Known one-offs: PLW1510 at `cli/plot.py:101`
   (`subprocess.run` without `check=`), S608 at `io/_duckdb.py:92` (the path
   is escaped; add a same-line reason). Not PLR / TRY / EM / FBT / COM /
   PLC0415 / ISC.

   Nine of the ten enabled, in five commits rather than ten. FURB+LOG,
   then PERF+C4+PIE+PTH together, then PLW, BLE and SLF separately. The
   four in the middle share files — `readers/ipic3d/_conserved.py` alone
   carries PTH123, PIE810, C401 and PERF401, two of them in one
   function — so a commit per family would have meant staging
   overlapping hunks, which serves bisection worse than one commit that
   names all four. The last three each carry a distinct judgement and
   are worth reading separately.
   **RET is not enabled.** It finds eight things here and is wrong about
   five: `_morton.py`'s bit-interleave and `fields.py`'s LaTeX
   substitutions are ladders of parallel transformations where promoting
   the last rung to a `return` makes it look special when it isn't;
   `test_operators` names its `max(...)` `error` because that is what
   the number means; and RET501's single finding is a `find_spec`
   returning None as the meta-path protocol's "not mine" signal. A
   family needing five suppressions to buy three line deletions is not
   worth the noqa noise, so the three genuine ones landed as a hand
   edit (`c45a2fa`) and the rule stayed off.
   Suppressions, all reasoned: PLW0603 per-file for `plotting/styles.py`
   (the active theme is process-wide because `use_theme()` wraps
   matplotlib's own global rcParams); BLE001 and SLF001 per-file for
   `tests/*` (aggregated sweeps where the exception type is the finding;
   private attributes that are the thing under test) and SLF001 for
   `scripts/visual/*`; and three same-line SLF001 plus three same-line
   BLE001 in src, each naming its seam. PERF, C4, PIE, PTH, FURB and LOG
   needed none.
   The S608 one-off above is moot: `S` is not in the family list and
   enabling it would report 3078 findings, 3000-odd of them S101
   "use of assert" in the test suite. With `S` off, a `# noqa: S608`
   would itself be flagged by RUF100. The escaping is already visible on
   the line above (`escaped = glob_pattern.replace("'", "''")`).

10. - [ ] **CI matrix (optional).** Currently 3.13 + 3.14 on ubuntu. macOS
    adds little (dev platform); Windows would add h5py path signal if wanted.

    Not done, deliberately. macOS is the development platform, so a job
    there mostly re-runs what is already run before every push. Windows
    would be real signal — `pathlib` is mandated everywhere precisely so
    paths stay portable, and nothing checks that claim — but no Windows
    support is claimed beyond the `Operating System :: OS Independent`
    classifier, and no Windows user has reported anything. Adding a
    platform to catch a hypothetical is the kind of speculative
    maintenance this audit removes elsewhere. Revisit on the first
    Windows bug report, or the first time the project claims Windows in
    prose rather than in a classifier.

Verify: full `./scripts/check.sh` after every item; the invariant is
2854+ tests green on 3.13 and 3.14, mypy strict clean, doctests green.

Verified on landing: 2930 passed / 15 skipped on both 3.13 and 3.14
(2912 before the doctests and guards this phase added), mypy strict
clean over 148 files, `ruff check .` and `ruff format --check .` clean,
coverage 88.19% against a floor of 86.

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

2. - [x] **Typing.** `containers.py:333,410` and `dataset.py:1093` bare
   `np.ndarray` → `IntArray` / `FloatArray` / `BoolArray` from `pypic.types`;
   `containers.py:200,210,266` annotate `Mapping[...]` to match the
   `MappingProxyType` substitution in `__post_init__` (`units.py` is the
   model); `:198` `default_factory=PhysicsParams`; the four
   `type: ignore[type-arg]` in `traces/_tracing.py:545,671,867,1075` →
   `dict[str, Any]`; named constants for magic indices at
   `batsrus/_hdf5.py:63-67` (`ipm[2]`, `ipm[1]`, `rpm[0]`),
   `ipic3d/_conserved.py` column indices, `openggcm/_wrn2.py` header offsets,
   and `ipic3d/_config.py` `reference_density`.

   Landed: `Mapping` went on every proxied field in `containers.py`, not just
   the three anchors, and mypy then pulled `FieldDataset.__init__` /
   `from_arrays` (`metadata`, `aliases`, `transforms` — all three already
   copy defensively) and two `io/_virtual.py` locals along with it. The
   `reference_density` anchor was stale: no such symbol in
   `ipic3d/_config.py`, and what sits behind that absence is now Phase 9.
   `_conserved.py` got prefixed constants per layout
   (`_A_` / `_B_` / `_SQ_`) because Format B's mapping is spelled out in two
   functions and its species stride in three; `_wrn2.py` got the encoding
   constants both the scalar and vectorized path read. No new test — item 1's
   invariant already asserts the runtime read-only half, and `check.sh types`
   is the static half.

3. - [x] **Import time.** `import pypic` takes 0.73 s warm (best of 7 fresh
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

   Landed: 0.73 s → 0.50 s (best of 7 fresh interpreters, 2026-09-12), with
   `scipy.interpolate` absent from `-X importtime` entirely. One flat
   name → module table, as planned. Resolving any one name binds that
   module's whole export set rather than the single name asked for:
   importing `pypic.regrid` binds the submodule as `pypic.regrid`, where the
   eager surface had the function of that name, so a one-at-a-time binding
   would make `pypic.regrid` callable or not depending on access order. Two
   guard tests, not one — the second pins that ordering.

Verify: `./scripts/check.sh types`, the full suite, and
`uv run python -X importtime -c "import pypic"` before and after item 3
(`scipy.interpolate` must drop out).

## Phase 9 — an unset normalization reads as SI

Budget ~6 files, ~70 added / ~5 deleted. Independent of Phases 6-8, and a
behaviour change rather than a cleanup, so it wants its own decision.

1. - [ ] **`in_si` cannot tell "already SI" from "no normalization known".**
   Phase 8 item 2 went looking for a `reference_density` constant in
   `ipic3d/_config.py`; the anchor was stale, and the reason it was stale is
   the finding. iPIC3D's `.inp` carries no *physical* reference density
   (`rhoINIT` is code units, typically 1.0, and lands as
   `SpeciesInfo.density`), so `to_simulation_config` hands the config
   `Normalization.identity()` (`_config.py:482`) and only a
   `simulation.toml` merge can replace it. `batsrus/_config.py:294`,
   `openggcm/__init__.py:103` and `_simple.py:222,505,682` fall back the
   same way. Identity is the honest choice for a dimensionless code — the
   problem is that it is silent.

   `Normalization` (`units.py:81`) carries no provenance field, so that
   fallback is indistinguishable from the identity a declared
   `system = "SI"` produces (`readers/config.py:317`). `in_si` then gets
   `factor == 1.0` and returns the array unchanged
   (`dataset.py:1026-1029`, same short-circuit at `:1038` for
   `in_units`), so `in_si("B_1")` or `in_units("B_1", "nT")` on an
   un-normalized iPIC3D run returns code units labelled tesla, with no way
   for the caller to notice. It is the one path where
   "normalized internally, converted at boundaries" can be violated without
   an error.

   Options: (a) a provenance field on `Normalization` that `identity()`
   leaves unset and every `[units]` path sets, with `in_si` / `in_units`
   raising a new `pypic.exceptions` subclass unless the caller opts in; (b)
   leave the arithmetic alone and stamp the fallback in metadata so
   `describe()` and `pypic info` can say "code units, no `[units]`
   section"; (c) both, (b) as the diagnostic and (a) as the guard. (a)
   changes behaviour for every reader that currently falls back, so it
   needs a release note and probably a 0.2 landing; (b) is additive and
   could ship first.

   Tests: a declared `system = "SI"` dataset still round-trips `in_si`
   unchanged; an iPIC3D fixture with no `simulation.toml` reports (or
   raises) instead of returning code units; the existing
   identity-normalization tests pass under whichever default lands.

## Non-goals

No 2D operator support (TASKS.md Step 50). No destaggering of BATSRUS or
OpenGGCM. No new core dependencies. No plotting rewrite around option
objects, no public kwarg or function renames. No schema semantics or JSON
artifact changes. No `derived.py` or `schema/_models.py` split. No big-bang
commits.
