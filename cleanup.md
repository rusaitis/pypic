# Code-quality cleanup

Record of the September 2026 code-quality audit. Phases 0-9 have landed;
Phase 10 is open.

Kept because each phase found something other than what its anchors
predicted — check the premise, not just the line number. Four of Phase 6's
proposed deletions turned out to be load-bearing; Phase 8's stale
`reference_density` anchor turned out to be Phase 9's entire finding; Phase
9's own reproduction case needed a change the plan had ruled out.

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

## Completed

### Phases 0-4 — bug fixes and structure (`git log 499a846..4562923`)

- Vector triplets sourced from the field registry rather than hand-listed.
- `ReaderBase` extracted; alias and layering consolidation.
- Splits: `_recipes`, `_field_table`, `cli/`, `tests/test_plotting/`.
- `reduce` decomposed.

### Phase 5 — plotting dedup (2026-09-11, `4562923..f3eb6b8`, 11 commits, 15 files, +917 / −1104)

- `resolve_norm` in `plotting/_colormaps.py` replaces three copies; returns
  one `Normalize`, not the planned tuple.
- `finish_axes` in `plotting/_resolve.py` replaces six epilogues. Leaves
  saving to the caller — inside the theme context `savefig` would pick up the
  theme's dpi/bbox — and rounds every plot, not only owned ones.
- `_theme_io.py` table-driven: one `_THEME_FIELDS` table drives load and save,
  replacing 53 hand-written `lines.append` calls. The `[webpic]` block
  round-trips untouched.
- `_vector_prelude` shared by `plot_streamlines` / `plot_quiver`.
- pyvista singular line/trajectory helpers wrap the plural.
- Additive kwargs: `ax=`, `vmin`/`vmax`, `units=`, `save=`; `add_contours`
  returns its `QuadContourSet`. `PlotTheme.rcparams` read-only.
- **Four latent bugs surfaced first**, each its own `fix:` commit: theme
  colormaps ignored when `cmap=None`, `load_theme` crashing on omitted colors,
  quiver `units=` coloring in code units, pyvista line/trajectory divergences.

### Phase 6 — duplication, dead weight, comments (2026-09-12, `9075636..a89c1f2`, 12 commits, 24 files, +420 / −430)

- One variadic `_apply_nan_policy`, `@overload`ed on arity.
- `_radial_bin` shared by `power_spectrum_2d/3d` — landed *after* a 2-D
  absolute-scale assertion, since the existing tests would not have caught a
  `digitize` off-by-one.
- `derived.py` delegates `p_par` in `agyrotropy` / `aunai_nongyrotropy`;
  `scudder_agyrotropy` stays inline to avoid a second `_unit_vector` pass.
- `io/metadata.py` attrs plumbing; tests rewritten to pin the surviving
  *encoders* (the server wire format, otherwise uncovered).
- Traces closed-loop detectors aligned and cross-referenced rather than
  merged — every shared-helper shape either copied on the scalar hot loop or
  left both callers windowing.
- Dead code removed. `err_prev`, the `GEOMETRY_BY_NAME` annotation and the
  `pypic.schema.cli` path each remove public surface and each carry a
  CHANGELOG line. History-narrating comments removed; `schema/_models.py`
  left alone, since those docstrings feed the generated JSON Schema.
- **Five proposed deletions were live code** and stay struck with reasons so
  nobody re-proposes them: `h_rel`, `passes_geometry`, the `in_si`
  fall-through, three of five `io/metadata` decoders, `decode_rle`.

### Phase 7 — tests and tooling (2026-09-12)

- pytest hardening: `markers`, `xfail_strict = true`, `filterwarnings = ["error", ...]`.
- **Lint gate blind spot fixed**: multi-root `ruff check` dropped `src/`
  nondeterministically, reporting "All checks passed" over two E501s and an
  F401. `check.sh` now uses one root.
- Tolerances added to bare `assert_allclose` — 194 by parse, not the 280 a
  grep reported (multi-line calls read as bare).
- conftest fixtures adopted, 66 banner comments removed, the
  `importorskip("pydantic")` gate dropped, doctests filled for non-exempt gaps.
- `pytest-cov` with a coverage floor; `.pre-commit-config.yaml`; ruff families
  PLW, BLE, SLF, one commit each.
- CI matrix (item 10) **declined with reasons**, not deferred — the one item
  left open.

### Phase 8 — pickling, typing, import time

- `pypic/_pickling.py` registers one `copyreg` reducer for `MappingProxyType`.
  A frozen dataclass storing one could previously be neither pickled nor
  deep-copied, which broke `pickle`, process pools, `joblib.dump`,
  `copy.deepcopy` and `dataclasses.asdict` across ten-odd classes including
  `FieldDataset`. Fixed the type, not its holders.
- Bare-`Any` typing cleanups in `containers.py` / `dataset.py`.
- `import pypic` 0.73 s → 0.50 s: PEP 562 lazy export of `comparison`,
  `reconnection`, `regrid`, `traces`, with `scipy.interpolate` gone from
  `-X importtime` entirely. `codegen` stays eager — deferring it measured no
  saving, since `__init__` imports pydantic through `pypic.schema` anyway.
- **Two guard tests, not one.** Resolving any lazy name binds that module's
  whole export set, so `pypic.regrid` is the submodule or the function
  depending on access order; the second test pins that.

### Phase 9 — an unset normalization reads as SI (2026-09-12)

- `Normalization.system: UnitSystem | None` plus `undeclared()`; the four
  reader fallbacks switched to it. `identity()` keeps meaning "declared SI".
- Guard placed in `si_factor` — the single chokepoint under `in_si`,
  `in_units` and `field_si_factor` — not in `dataset.py`.
- The bit round-trips through Zarr/HDF5 stores and the server wire; a store
  written without the key decodes as declared-`custom`.
- `pypic info` stopped asserting the lie; docs and
  `examples/ipic3d-double-harris.toml` corrected — the anchor is *chosen*,
  not reconstructed from `qom`/`B0`/`rhoINIT`.
- **The finding**: 0.1 code units of B reported as 10⁸ nT, silently. The
  comparison family defaults to `units = "si"`, so three library entry points
  and two CLI commands changed behaviour; two internal `identity()` sites
  (diff datasets, `--to-si` output) are deliberately exempt.
- Landed: 2957 passed / 15 skipped, coverage 88.23%, `units.py` at 100%.

### Follow-on — new-user contradictions (2026-09-12, separate plan)

Not a cleanup phase, but Phase 10 builds on it. Closed nine contradictions a
first-time user hits: the custom-`[units]` `0.0` sentinel, unreachable
`compute("psi")`, missing-input errors reported as unknown names, massless
fluid electrons, 2D differential operators (TASKS Step 50), `reference_velocity`
for hybrid anchors. Its lasting artifact is
**`tests/test_schema_parity.py`'s validator↔loader invariant**: whatever
`validate_simulation_toml` accepts, `load_config` must accept. Phase 10 item 1
is the first case where that invariant needs a documented exemption.

## Phase 10 — what pypic silently assumes

`FieldDataset` and `GridInfo` hard-code four assumptions that appear in no
docstring and no test: the grid is **uniform**, the arrays are **real**, there
are **at most three dimensions**, and **boundaries are open**. All four are
defensible. What is not defensible is how they fail — three of the four return
plausible wrong numbers, and the remaining one raises from inside a `zip`.

This matters beyond tidiness because it is the ceiling on which codes pypic
can ever read. Measured against `986cd68` on 2026-09-13, walking two cases
that the schema already claims to describe: a spectral code (Fourier in
configuration space, or Hermite-Laguerre in velocity space) and a
region-varying fluid/kinetic model (MHD-EPIC, FLEKS, MHD-AEPIC).

Two things checked out better than expected and need no work:

- **`[model].type` drives zero dispatch.** Every use is display, server JSON,
  or the one validator cross-check that `[physics.pic]` matches `type = "PIC"`.
  So `"hybrid"` / `"vlasov"` / `"gyrokinetic"` cost nothing and break nothing.
- **The eight references carry no `v = l/t` constraint** — `__post_init__`
  checks positivity only. That is what makes gyro-Bohm expressible, where the
  length unit ($\rho_i$) and time unit ($L_{ref}/c_s$) are deliberately not
  related by the velocity unit. A stricter constructor would have locked it out.

**The phase has two halves, and the second one was not planned.** Items 1-5
are the container assumptions above. Items 6-9 are the *normalization*
assumptions, and they turned out to share a single root cause worth stating
plainly, because it reframes what item 7 is for:

> Nothing in pypic ever checks that a `Normalization` is consistent with the
> code-unit convention `derived.py` computes in.

Every EM quantity in `derived.py` hardcodes SI-rationalized units — `e_B` is
$B^2/2$, `e_E` is $E^2/2$, `div_E` is $\rho_c$ — which holds exactly when
$B_{ref}^2/(\mu_0 n_{ref} m_{ref} v_{ref}^2) = 1$. That ratio is never
computed anywhere. Both bugs below are it going unnoticed:

| Item | Symptom | Ratio |
|---|---|---|
| 6 | `si_factor("poynting_flux")` off by $\mu_0$ | 1 (the factor itself was wrong) |
| 7 | `speed_of_light = c/10` corrupts every EM conversion | $(c_{SI}/c_{ref})^2 = 100$ |
| 8 | `system = "SI"` is wrong by $\mu_0$ for every EM quantity | $1/\mu_0 = 795775$ |

**Eight references are enough; the taxonomy is not the problem.** Surveyed
2026-09-13 — the three consistency relations across the code families pypic
claims to serve:

| Family | $v = l/t$ | $E = vB$ | $B^2/(\mu_0 n m v^2)$ |
|---|---|---|---|
| PIC, electron-normalized | 1 | 1 | 1 |
| PIC, ion-normalized | 1 | 1 | 1 |
| MHD, Alfvénic | 1 | 1 | 1 |
| hybrid ($d_i$, $v_A$) | 1 | 1 | 1 |
| gyrokinetic (gyro-Bohm) | $\rho_* = 0.0016$ | 1 | $2/\beta$ |
| declared SI | 1 | 1 | $1/\mu_0$ |

The four basic families are exactly consistent and already expressible —
`pic_standard` with `reference_species`, `mhd_standard`, and the
`reference_velocity` branch cover them with no new machinery. Gyrokinetics
goes through `custom` and is *deliberately* inconsistent on two of the three:
$\rho_*$ is the gyrokinetic ordering parameter, and the ratio is $2/\beta$
because GK normalizes $B$ to the equilibrium field rather than to the field
at which $v_A = v_{ref}$. Both are physics, not error — which is why the
ratio must be a *reported diagnostic* rather than a hard gate: its **value**
names the convention.

So the answer to "do we need more `system` values" is no. What the eight
references cannot express is the EM convention, and that is a separate axis
from the anchor — `[model].type` is the code family, `system` is the input
form, and neither says whether $\mu_0 = 1$.

So item 9's consistency checks are not tidying after the fixes — they are the
thing that makes this family of bug impossible rather than invisible, and they
would have caught both. Land them even if the exponent table slips.

Budget ~12 files, ~220 added / ~70 deleted. Items 1-3 are guards and are
independent. A separate finding, recorded because it is the same shape as
item 1: `FieldInfo.unit_dimension` is validated on registration and `None`
for every built-in field, so the openPMD metadata pypic advertises comes
entirely from a fallback — documented, validated, never populated.

1. - [ ] **Refuse `[grid.stretched]` at the reader boundary.** The section
   validates, reaches `SimulationSchema.grid.stretched`, and `_build_grid`
   (`readers/config.py:293-315`) drops it — `GridInfo` has one `spacing` float
   per axis and `coordinate_arrays()` computes `origin + (i+0.5)*dx`.
   Measured, with `axis_widths = [0.5, 0.6, 0.8, 1.0, 1.3]`:

   ```
   GridInfo.spacing = (0.84, 1.0, 1.0)
   coords           = [0.42 1.26 2.10 2.94 3.78]
   truth            = [0.25 0.80 1.50 2.40 3.55]
   ```

   Every gradient, divergence, curl and integral on that axis is then wrong by
   a position-dependent factor, silently. This is the `[units]` `0.0` sentinel
   again, except that one raised. Raise in `_build_grid` naming the section and
   the axes that carry widths.

   **This breaks the validator↔loader parity invariant**, deliberately, and
   the test needs a named exemption rather than a weakened assertion: the
   Pydantic model is the *cross-tool* contract (rustpic and webpic may well
   support stretched axes), while `load_config` is pypic's reader. Record the
   distinction in the test's failure message — "pypic does not read this, and
   here is why" is a different statement from "this document is invalid".

   Honouring stretched axes is a TASKS item, not a cleanup one, but it is
   smaller than it looks: `np.gradient` already accepts coordinate arrays in
   place of a scalar spacing, so the work is an optional
   `GridInfo.axis_coords` plus threading it through `coordinates/operators.py`.
   Note that pointer where the raise lands, so the next reader does not
   re-derive it.

2. - [x] **Reject complex field arrays in `FieldDataset.from_arrays`.**
   Complex input is accepted today and propagates into physics that assumes
   real. Measured: `compute("|B|")` on complex `B_1/B_2/B_3` returns
   `complex128` — $\sqrt{B_1^2+B_2^2+B_3^2}$ under complex arithmetic, not
   $\sqrt{|B_1|^2+|B_2|^2+|B_3|^2}$ — with no warning. A pseudo-spectral code
   dumping k-space, or any reader that forgets an inverse transform, gets
   numbers that look like fields.

   Raise on non-real dtype at `dataset.py:165`, naming the field and saying
   that spectral coefficients want an inverse transform at the reader
   boundary. One guard, one test; the alternative (complex-aware magnitudes
   throughout `derived.py`) is a much larger change for a use case no reader
   currently produces.

   **Landed one level down, in `__init__` rather than `from_arrays`.** The
   plan named `from_arrays` because `ReaderBase._finish` routes through it, so
   every reader is covered — but `io/zarr.py` and `io/_virtual.py` build a
   `FieldDataset` directly, and a foreign Zarr store or a VirtualiZarr view
   over legacy HDF5 is exactly where k-space data arrives. `__init__` is the
   one chokepoint under all three, plus `with_field` and the slicing path;
   same Phase 9 reasoning that put the undeclared-normalization guard in
   `si_factor` instead of at its three callers. Cost is an
   `np.issubdtype` per variable per construction.

3. - [x] **A domain error for grids above three dimensions.** `GridInfo`
   accepts a 5-tuple without complaint — `__post_init__` (`grid.py:61-90`)
   never compares `len(dimensions)` against the geometry's three
   `axis_names` — so `surviving_axis_names` silently truncates to three and
   the failure surfaces two calls later:

   ```
   File "src/pypic/dataset.py", line 253, in from_arrays
       axis_coords = dict(zip(dim_names, coord_arrays, strict=True))
   ValueError: zip() argument 2 is longer than argument 1
   ```

   Check in `GridInfo.__post_init__` against `len(geometry.axis_names)`,
   naming the dimensionality and pointing at `[phase_space]` — which already
   rides through to `SimulationConfig.phase_space` as typed metadata, so a 5D
   gyrokinetic run can *describe* itself today even though no container holds
   its distribution function.

4. - [ ] **Periodic domains get open-boundary stencils, and pypic already
   knows better.** `GridInfo.boundary` records `("periodic", ...)` per axis,
   and `grep` finds it **never read by any numerical code** — only validated
   (`grid.py:79`), sliced (`:165`), and serialized (`io/metadata.py`). Every
   operator goes through `np.gradient`, which applies one-sided stencils at
   the first and last grid point instead of wrapping.

   Measured on an analytically divergence-free periodic field
   $\mathbf{B} = (\sin x \cos y,\; -\cos x \sin y,\; 0)$ on a $32^3$ box:

   ```
   interior  max |div B| = 1.887e-15    (O(h^2), at roundoff)
   full grid max |div B| = 9.546e-03    <- boundary faces
   cells on a boundary face: 5768 of 32768 (18%)
   ```

   Thirteen orders of magnitude, across a fifth of the domain. It bites
   spectral and turbulence codes hardest because those runs are periodic *by
   construction*, and their headline diagnostics — $\nabla\cdot\mathbf{B}$
   drift, vorticity statistics, spectra — are box-wide reductions where 18%
   of cells is not a boundary detail.

   `docs/conventions.md` § *Boundary treatment for finite differences* does
   document it ("periodic domains should pad ghost cells before calling"), so
   this is a known limitation rather than a surprise. What is new is that the
   information needed to do better is already in the container and ignored.

   Phase 10's part is the guard, matching item 1: when `grid.boundary` marks
   an axis periodic and the operator cannot wrap, say so once rather than
   returning a silently degraded edge. Actually wrapping the stencil is TASKS
   Step 52 — `np.gradient` has no periodic mode, so it means `np.roll`-based
   central differences on flagged axes, which is a small kernel but a real
   numerical change that wants its own convergence tests.

5. - [ ] **Write the four invariants down.** They belong in
   `docs/architecture.md` next to "Readers produce `FieldDataset`", as one
   short block: uniform structured grid, real-valued arrays, at most three
   dimensions, open boundaries — each with the one-line reason and the escape
   hatch. Items 1-4 make the code enforce them; this makes a contributor able
   to find them before writing a reader that violates one.

6. - [x] **Fix the Poynting SI factor** (a `fix:`, lands before item 9).
   `si_factor("poynting_flux")` returns $E_{ref}B_{ref}$, which is wrong by
   $\mu_0$ — results are $\sim 10^6$ too small. Measured on
   `pic_electron(1e18)` with $E = 0.1$, $B = 0.2$ in code units:

   ```
   pypic in_si(S)  = 6.168662e+05 W/m^2
   hand-computed   = 4.908865e+11 W/m^2
   ratio           = 1.256637e-06   == mu_0
   ```

   The derivation: pypic's code units are SI-rationalized, so
   `derived.poynting_flux` returns $\mathbf{E}\times\mathbf{B}$ with no
   $\mu_0$ (checked: `magnetic_energy_density(2) == 2.0`, i.e. $B^2/2$, not
   the Gaussian $B^2/8\pi$). Then
   $S_{SI} = S_{code}\,E_{ref}B_{ref}/\mu_0$ — the $1/\mu_0$ is missing.

   **The correct factor is `e_field_ref * b_field_ref / mu_0`.** Write it
   that way and not as $n\,m\,v^3$: the two are equal *only* when the
   reference set itself satisfies $B_{ref}^2 = \mu_0 n_{ref}m_{ref}v_{ref}^2$,
   which `pic_standard` and `mhd_standard` do by construction (verified to 1
   part in $10^{12}$) but a hand-built `custom` set need not — measured, they
   diverge by eight orders on a custom set with an independently chosen
   `b_field`. The $E_{ref}B_{ref}/\mu_0$ form follows from
   $\mathbf{S} = \mathbf{E}\times\mathbf{B}$ alone and assumes nothing about
   the relationship among references. That the two forms *coincide* for the
   built-in constructors is a useful invariant to assert separately, not the
   definition to implement.

   The registered dimension $(0,1,-3,0)$ = W/m² has been right all along, and
   `e_B` converts correctly today, so this is the one EM entry that drifted.

   Why nothing caught it: `tests/test_units.py:172` pins the factor as
   `lambda n: n.e_field_ref * n.b_field_ref` — the implementation restated,
   so the test is a tautology. Replace it with a physics assertion
   ($E\times B/\mu_0$ from hand-computed SI inputs), which is the form that
   would have failed. The `_COMPOUND_FACTORS` comment claiming the two fluxes
   "normalize differently" is the reasoning that produced the bug and goes
   with it.

7. - [x] **`[units].speed_of_light` below $c$ silently breaks $\mu_0 = 1$**
   (a second `fix:`; same root as item 5, different symptom). Reduced-$c$ runs
   are routine — semi-implicit PIC relaxing the CFL, Boris-corrected MHD — and
   the key that looks like the place to record it is a trap.

   `pic_standard` scales `length_ref` and `velocity_ref` with the supplied
   $c$ but pins $B_{ref} = m\,\omega_{ref}/q$, which is $c$-independent. So
   the rationalization ratio moves as $(c_{SI}/c_{ref})^2$:

   ```
   c = c_SI       ratio = 1.0        l=5.3141e-03  v=2.9979e+08  B=3.2075e-01
   c = c_SI/10    ratio = 100.0      l=5.3141e-04  v=2.9979e+07  B=3.2075e-01
   c = c_SI/100   ratio = 10000.0    l=5.3141e-05  v=2.9979e+06  B=3.2075e-01
   ```

   Every EM conversion is then wrong by that factor. Magnetic energy density
   on a $c/10$ deck, measured: pypic `1.023388e+02`, true `1.023388e+04` J/m³
   — a factor of 100, silent.

   **The coherent knob already exists.** `reference_velocity` — added for
   hybrid anchors — takes the generalized
   $B_{ref} = v_{ref}\sqrt{\mu_0 n_{ref} m_{ref}}$ branch and holds the ratio
   at 1:

   ```
   speed_of_light     = c/10  ->  ratio 100.0
   reference_velocity = c/10  ->  ratio   1.0
   ```

   So the fix is not new machinery. Reject or warn when `speed_of_light`
   drives the ratio away from 1, and say which of the two things the deck
   meant:

   - *A dimensionless modelling choice* ($c/v_A$ reduced, mass ratio
     lowered): leave `speed_of_light` at $c$ and record it in
     `scaling_factor` / `scaling_description`, which the schema already
     defines as informational with **no effect on computation** — the
     reference template's own example is `"c/v_A reduced by 10x"`.
   - *A genuinely different velocity unit*: use `reference_velocity`.

   **Why no third option.** SI *defines* $c$. A run that really solves
   Maxwell with $c' \neq c$ is not a rescaled plasma but a different vacuum,
   where $c'^2 = 1/(\mu_0'\epsilon_0')$ forces at least one vacuum constant
   off its SI value. There is no unique SI mapping to pick, so the library's
   job is to say so rather than choose one silently.

   Two smaller things in the same pass: `speed_of_light` is declared on
   `UnitsPIC` only (`_models.py:593`), so MHD decks — exactly where the Boris
   / reduced-speed-of-light approximation lives — cannot express it at all,
   while `docs/schema.md` says it "stays at top-level `[units].speed_of_light`
   regardless of approach". And `PhysicsParams.c` stays 1.0 regardless
   (`readers/config.py:432-439` already documents this), so no relativistic
   branch can see a reduced $c$ either.

8. - [x] **`system = "SI"` is unsafe for every EM quantity** — decide the
   semantics, then warn. The most user-facing of the three, because "my data
   is already in SI" is where a newcomer starts and `UnitsSI` is a first-class
   schema variant.

   `derived.py` computes in SI-*rationalized* units ($\mu_0 = 1$). SI data has
   $\mu_0 = 1.2566\times10^{-6}$. The two cannot both hold, so every EM
   quantity comes out wrong by a power of $\mu_0$. Measured on solar-wind
   values ($B = 5$ nT, $n = 5\times10^6$ m⁻³), declared `system = "SI"`:

   ```
   v_A   pypic = 5.4675e+01 m/s     true = 4.8773e+04 m/s    ratio = sqrt(mu_0)
   e_B   pypic = 1.2500e-17 J/m^3   true = 9.9472e-12 J/m^3  ratio = mu_0
   ```

   `docs/schema.md:451` says "If `system` is 'SI', all data is already in SI
   and no conversion is needed (all reference values = 1.0)" — true for the
   *conversion*, false for the *computation*, and the doc does not say so.
   Dimensionless quantities ($\beta$, Mach numbers, agyrotropy) and non-EM
   ones (density, pressure, temperature, $|V|$) are unaffected; the damage is
   confined to quantities where $\mu_0$ or $\epsilon_0$ appears.

   Three ways out, in increasing cost — **this one needs a decision, not a
   default**:

   - *Warn and document.* `system = "SI"` keeps meaning "conversion is a
     no-op", and EM quantities raise or warn. Cheapest, honest, leaves the
     user to pre-normalize.
   - *Normalize on load.* Treat SI as an input encoding, not a code-unit
     system: rescale to a $\mu_0 = 1$ set at the reader boundary. The
     machinery exists — `normalize_fields` (`_config_helpers.py:97`) does
     exactly this — but it **short-circuits on `is_identity`**, which is
     precisely the case that needs it. Correct, and the largest change.
   - *Carry $\mu_0$ into the EM functions.* Parameterize `derived.py` the way
     the relativistic `c=None` kwarg already works. Most general, most
     invasive, and only worth it if non-rationalized code units become a real
     requirement.

   Whichever is chosen, item 8's ratio check reports it as $1/\mu_0$, so the
   diagnosis is free either way.

9. - [x] **`_COMPOUND_FACTORS` as exponent vectors.** `units.py:583-607` is 16
   lambdas, and every one is a *monomial* in the eight references.
   `pressure` is $n\,m\,v^2$, `current_density` is $q\,n\,v$, `frequency` is
   $t^{-1}$, `b_field_per_length` is $B/l$. Replace the lambda table with
   eight-integer exponent tuples and derive the factor by `math.prod`. Same
   entry count, less code, and it pays three times.

   Three tables are keyed by the same `QuantityType` strings today:
   `_QUANTITY_UNITS` (display label), `_QUANTITY_DIMENSIONS` (openPMD SI
   7-tuple) and `_COMPOUND_FACTORS` (factor over the eight references). The
   last two encode the *same physics in two notations*, independently
   maintained, **with no test relating them** — which is exactly how item 5's
   bug survived. Composing the 8-exponent vector with each reference's own SI
   dimension reproduces `_QUANTITY_DIMENSIONS` for 21 of 22 entries today;
   the 22nd is `poynting_flux`, and once item 6 lands it is 22 of 22 with no
   exception. `_QUANTITY_UNITS` stays hand-written — it carries information
   the exponents do not (`pressure` and `energy_density` have *identical*
   exponents and differ only as "Pa" versus "J/m³", which is why the enum
   cannot collapse).

   **Two invariants, and they are the real product of this item** — see the
   phase preamble for why they, not the table, are the point.

   - *Dimension-of-factor equals registered-dimension*, for every quantity
     type, as one aggregate test. Catches item 5.
   - *The rationalization ratio $B_{ref}^2/(\mu_0 n_{ref} m_{ref} v_{ref}^2)$
     is 1.* Catches item 6, and every future deck that reaches the same state
     by another route. **Not advisory — warn on construction**, naming the
     ratio and the two legitimate ways to express a non-$c$ velocity unit
     (`reference_velocity`, or `scaling_factor` for a dimensionless
     modelling choice). A set that violates it is announcing that its code
     units are not SI-rationalized, and for such data the whole EM surface is
     wrong, not one conversion — that is too large a failure to whisper.

   Warn rather than raise: `Normalization` is constructible by hand and the
   eight-positional-argument form is used throughout the suite, so a raise
   would be a breaking change for data that may never touch an EM quantity.
   The advisory-versus-fatal line sits where the phase's other guards sit —
   refuse at the *reader* boundary (item 1), warn on a hand-built object.

   **Scoping note — pypic mandates one code-unit convention, deliberately.**
   `docs/conventions.md` § *Gaussian vs SI-Rationalized* already fixes it:
   code units are SI-rationalized ($\mu_0 = \epsilon_0 = 1$, no $4\pi$), and
   Gaussian codes such as iPIC3D are converted **at the reader boundary**,
   once. Item 5 does not add that constraint; it makes the conversion agree
   with the constraint that already exists. Supporting a user-chosen
   $\mu_0 \neq 1$ would mean parameterizing every pure function in
   `derived.py`, not changing a factor — and `PhysicsConstants` already
   stores the code-unit $\mu_0 / \epsilon_0 / c$ that such a design would
   read, while `Normalization.si_factor` cannot see it. Worth knowing the
   seam exists; not worth opening in this phase.

   The other two payoffs:

   - **TASKS Step 43c falls out.** After `reduce(reduction="integrate")` along
     $n$ axes the SI unit shifts by $n$ length factors, and today
     `quantity_type` is preserved unchanged so `in_si()` is off by
     `length_ref**n`. With exponents it is a vector subtraction.
   - **New quantity types get a safe path.** `register_recipe` rejects an
     unknown `quantity_type` eagerly, with the valid list — good behaviour,
     keep it. But a genuinely new dimensional group (gyro-Bohm flux, say)
     currently needs a new lambda; with exponents it is a tuple a user can
     supply without being able to express nonsense.

   Two notes for whoever lands it:

   - **Not bit-identical.** `math.prod` associates left-to-right in reference
     order, the lambdas associate as written, so multi-factor entries
     (`pressure`, `current_density`, `energy_flux`, `power_density`) differ by
     **at most 3 ULP, relative 4.3e-16** (measured over 2000 random
     normalizations). Pin the equivalence test with `rtol=1e-15`, not `==`,
     and say in the commit that the change is deliberate and sub-ULP-scale.
   - **`FieldInfo.unit_dimension` is not the parallel table** — it is a
     per-field *override*, and it is `None` for every built-in field.
     `dataset.py:272` and `:938` already fall back to
     `quantity_dimension(quantity_type)`, which is where the real table
     lives. Leave the override alone.

   Incidental fix in the same pass: `_QUANTITIES` is six names, not eight —
   `mass` and `charge` are references with no `si_factor`, so
   `SpeciesInfo.mass` has no route to SI. The exponent table covers all eight
   for free.

   Do **not** open `QuantityType` to arbitrary strings. Its being closed and
   eagerly validated is what makes `in_si()` trustworthy; the exponent table
   is the extension point, the enum stays the vocabulary.

10. - [x] **Teach the explicit form the third relation** — landed as part of
   the schema 2.0 `[units]` collapse, not as a standalone item.

11. - [x] **A velocity anchor for the MHD form** — subsumed by the same
    collapse. `UnitsMHD` no longer exists to add a key to.

**What actually landed, and why it was bigger than items 10-11.** Asked
whether the deferred `system` -> anchor-form rename was worth doing, given
that breaking compatibility is now free (no users, 0.2 ahead). The answer
turned out to be that the *rename* is cosmetic but the *collapse* it enables
is not, and item 10's third relation is what enables it:

- **`UnitsMHD` is a strict special case of `UnitsCustom`** once
  $B = v\sqrt{\mu_0 n m}$ is known. Verified: `mhd_standard(L, rho, B)` is
  reproduced **bit-exactly on all eight references** by an explicit deck. So
  adding item 10's relation does not extend a form, it deletes one.
- Four `[units]` variants plus a nested `[units.reference]` sub-table became
  **three flat anchor forms** — `from_species` (length derived from a
  species' plasma frequency), `explicit` (length given), `si`.
- `explicit` takes `reference_length` plus **any two** of a velocity, a
  density and a field; the third follows. That rule is the rank of the
  relation set, not a convention, and it covers MHD, PLUTO/Athena++,
  hand-built PIC and gyrokinetics without a fourth form.
- **Over-supply is legal, duplicate spellings are not.** Item 11 had said the
  `b_field` + `velocity` pair must be rejected; that is wrong, because the
  gyrokinetic case needs exactly that pair accepted and no tolerance gate can
  tell a deliberate $2/\beta$ from a typo. The rule that resolves both:
  reject two names for *one* primitive (both density keys), accept redundant
  *distinct* primitives. Recorded so nobody re-derives item 11's version.
- Three defects died with it: `reference_density` meaning m⁻³ under `PIC` and
  kg/m³ under `MHD`; `reference_length` versus the nested `.length` as two
  spellings of one concept; and PLUTO-class decks being **rejected outright**
  for a $B_{ref}$ the deck already implied.

**Item 8 resolved as *normalize on load*.** `system = "SI"` was conflating
two things — an anchor form and an input encoding — and they separate into
`anchor` plus `data_in_si`, because rescaling SI data needs an anchor to
rescale *against*. A Vlasiator-shaped deck now returns hand-checked SI for
$v_A$ and $e_B$; it was wrong by $\sqrt{\mu_0}$ and $\mu_0$.

**Two measurements worth keeping.** The "bit-exact" claim above holds only
because the loader derives the velocity from $\rho_m$ directly:
$(\rho/m_p)\cdot m_p \neq \rho$ for **3.2% of random double mass densities**,
so routing through $n$ and back is not exact in general — the repo's own MHD
deck round-trips by luck. And the `MHD` label never carried the information
it looked like it carried: $v_{ref}$ equals the Alfvén speed of the reference
state in **every** rationalized form, PIC included, so nothing was lost by
dropping it.

**What did not move: `[model].type`.** It was never the duplicate. The two
fields simply shared a vocabulary while meaning different things, so a hybrid
deck read as the contradiction `type = "hybrid"`, `system = "PIC"`. Once
`[units]` stops using code-type words the collision is gone from both sides.


12. - [ ] **Document how exotic codes actually land.** A short
   `docs/architecture.md` addition, because the answer is counter-intuitive
   and currently lives nowhere: **extend readers, not containers.**

   - *Spectral in velocity space* (Hermite-Laguerre, gyrokinetic): the first
     three coefficients of a Hermite hierarchy **are** `n_s0`, `V_s0_i`,
     `P_s0_ij`. A reader that maps moments 0-2 onto canonical names inherits
     the entire derived surface for free. The high-order tail is a different
     kind of object and wants a different container, not a 6-D `FieldDataset`.
   - *Region-varying fluid/kinetic*: one normalization per run is **correct**,
     not a limitation — coupled codes must agree on units at the interface.
     The embedded patch is a second grid, and two `FieldDataset`s plus
     `align_grids` / `compare_fields` is already the supported answer. What is
     genuinely missing is only a *relationship* — nothing records "same run,
     same step, different region" — and the sharper trap is that `beta` on the
     fluid side and `beta_s0` on the kinetic side mean different things with
     nothing saying so.

### Pressure test — four codes outside the reader matrix

`TASKS-schema-extension.md` § *Reader → blocker matrix* tracks what each
pending reader needs from the **schema vocabulary**. It says nothing about
normalization form or the runtime container, which is what this phase is
about, so four codes were walked against those instead (2026-09-13). Marked
*(unverified)* where the claim comes from the code's documentation rather than
from a file on disk — confirm before anyone builds a reader on it.

| Code | Units form (v2.0) | Blocked by |
|---|---|---|
| **Entity** (SRPIC/GRPIC, GPU) | `from_species` — skin-depth anchored, $v_{ref} = c$ *(unverified)* | item 1: QSpherical is a log-radial stretched grid. GRPIC also needs an off-diagonal metric, which `CoordinateGeometry` cannot express at all — out of scope, not a Phase 10 item |
| **Zeltron** (relativistic PIC) | `from_species` *(unverified)*; Gaussian CGS converts at the reader, iPIC3D precedent | nothing in this phase. QED / radiation reaction already deferred in the matrix |
| **PLUTO** (MHD/RMHD) | ✅ `explicit` — `reference_length` / `reference_mass_density` / `reference_velocity`, one-to-one with `UNIT_LENGTH` / `UNIT_DENSITY` / `UNIT_VELOCITY` | item 1 only (logarithmic grid patches). Gaussian $4\pi$ converts at the reader |
| **Vlasiator** (hybrid-Vlasov) | ✅ `explicit` + `data_in_si = true` | nothing in this phase. Dual FSgrid/DCCRG regridding is already TASKS Step 23 |

**Resolved.** PLUTO and Vlasiator were the two mis-served codes and both are
now expressible: PLUTO under an accurate label rather than `custom`, and
Vlasiator's SI arrays rescaled on load instead of run through the
$\mu_0 = 1$ EM functions unconverted. Item 1 is the only Phase 10 item still
load-bearing for any of the four, and it is a container guard, not a
normalization one.

**One stale entry found next door.** The matrix marks `[grid.stretched]` ✅
as unblocking ARMS (Step 36, line 267), but item 1 measures that the section
validates and is then dropped by `load_config` — so ARMS spherical-$r$ is
unblocked in the *schema* and silently wrong at *runtime*. Fix the ✅ when
item 1 lands, or ARMS gets built on it.

Verify: `./scripts/check.sh` in full. Items 1 and 3 touch the schema path, so
`./scripts/check.sh schema` must stay green — none of this changes the JSON
artifact, and if it does, the change was not additive.

Items 6-8 are the phase's behaviour changes and each wants a CHANGELOG line:
`in_si` / `in_units` on Poynting flux move by $1/\mu_0$, which is a
correction, not a migration. Item 9 wants two tests landed *before* the
lambdas are deleted — the exponent table reproduces every current `si_factor`
to `rtol=1e-15` over a lopsided normalization, and dimension-of-factor equals
`_QUANTITY_DIMENSIONS` for all 22 entries once item 6 has landed.

## Non-goals

No N-D phase-space `FieldDataset` — it pays once and costs the whole
container; `[phase_space]` metadata already carries the description, which is
most of the value. No `system = "gyrokinetic"` or other new `UnitSystem`
member; the anchor is parameterized by `reference_velocity` and the custom
reference table, and `[model].type` already holds code identity. No complex
arithmetic through `derived.py`. No honouring of stretched axes in this phase
(item 1 refuses them; honouring is a TASKS item). No destaggering of BATSRUS
or OpenGGCM. No new core dependencies. No plotting rewrite around option
objects, no public kwarg or function renames. No schema semantics or JSON
artifact changes. No `derived.py` or `schema/_models.py` split. No big-bang
commits.
