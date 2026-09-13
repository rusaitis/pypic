# pypic — Implementation Roadmap

Each step produces something testable. No step starts until the previous step's tests pass.

## Completed

- [x] **Step 1:** Project skeleton (pyproject.toml, uv, ruff, pytest, mypy, CI)
- [x] **Step 2:** Normalization class (pic_standard, pic_electron, mhd_standard, identity)
- [x] **Step 3:** PhysicsConstants, SpeciesInfo
- [x] **Step 4:** CoordinateGeometry, GeometryType, metric_factors
- [x] **Step 5:** FieldDataset, GridInfo, SimulationReader, SimulationConfig
- [x] **Step 6:** simulation.toml loader
- [x] **Step 7:** Derived part 1 — |B|, |E|, |J|, |V|, beta, v_A, Poynting, energies, entropy
- [x] **Step 8:** Derived part 2 — characteristic scales (v_th, omega_p, d_i, r_i, lambda_D, c_s, M_A)
- [x] **Step 8b:** Per-species pressure decomposition in compute registry
- [x] **Step 9:** Diagnostics — L2/Linf error, div B/E, field energy
- [x] **Step 10:** Differential operators — curl, div, grad (Cartesian)
- [x] **Step 11:** PlaneSelection, BoxSelection
- [x] **Step 12:** Readers — iPIC3D (parallel/serial/H5hut), BATSRUS (IDL/HDF5), OpenGGCM, SimpleReader + registry + auto-detection
- [x] **Step 13:** compute(), in_si(), in_units(), QuantityType, field metadata registry
- [x] **Step 14:** 2D plotting — plot_field_slice, plot_comparison, publication styles
- [x] **Step 15:** Frame transforms — ReferenceFrame, FrameTransform, chaining, transform_to
- [x] **Step 16:** SphereSelection (NaN masking) + FieldDataset.where()
- [x] **Step 17:** MkDocs documentation site
- [x] **Step 18:** Relativistic — lorentz_factor, magnetization, ~9 functions with rel. corrections
- [x] **Step 19:** Regrid — regrid(), align_grids(), common_grid() (Cartesian)
- [x] **Step 20:** Cross-grid comparison — compare_fields(), field_comparison_report(), field_difference_dataset()
- [x] **Step 21:** CLI — info, fields, stats, compare, validate (typer + rich)
- [x] **Step 21b:** available_fields() + available_fields_mapping()
- [x] **Step 22:** CLI — plot, plot-compare (themes, contours, animate, batch)
- [x] **Step 30:** Reduced geometry after slicing
- [x] **Step 32:** Separate four_velocity quantity type
- [x] **Step 33:** specific_energy quantity type for enthalpy (fixed dimensional bug)
- [x] **Step 34:** StaggerInfo provenance metadata
- [x] **Step 45:** Spectral analysis — `power_spectrum_1d/2d/3d` in `pypic.spectral`, Parseval-consistent, Hann-windowed, radially/spherically averaged
- [x] **Step 46:** Reconnection diagnostics — `pypic.reconnection` (`find_saddle_points`, `reconnection_rate`, `schindler_xi`) plus the `D_e` / `R_recon` / `agyrotropy` / `D_ng` / `A_phi` registry entries
- [x] **Step 47:** Field-line tracing — `trace_field_line`, `trace_field_line_adaptive` (Dormand-Prince 5(4) with elementary (I) step control), the batched path, and `pypic.numerics`
- [x] **Step 48:** Poincaré sections — `PoincareSurface`, `poincare_section`, `plane_crossings`, `plot_poincare_section`
- [x] **Step 49:** Cross-language codegen — `pypic.codegen` exports aliases, recipes, species templates and field metadata as one JSON bundle; CLI `pypic export`
- [x] **M0-prep:** webpic API readiness — public `pypic.aliases`, `Recipe`, `RECIPES`, a `[webpic]` block in the bundled themes, and public-API invariants tests

### Deferred

- [ ] **Step 31:** Remove default geometry from operators — revisit when non-Cartesian operators land.

**Milestone: daily-use tool** — load data → compute derived quantities → compare runs → select subregions → convert units → make paper figures. ✅

## Phase 8: Modern I/O Formats

- [x] **Step 24:** Zarr v3 export/import — `to_zarr`, `to_zarr_timeseries`, `from_zarr`; canonical names on disk, bitshuffle+zstd, `dtype="float32"` and `shards=`
- [x] **Step 24b:** VirtualiZarr — `open_virtual(path)` builds a virtual Zarr view over legacy HDF5 with no conversion
- [x] **Step 24c:** Icechunk backend — versioned ACID storage over Zarr v3, written via `to_zarr(..., backend="icechunk")` and auto-detected on read
- [x] **Step 25:** Parquet/Arrow particles — in-memory `particles_to_arrow` and Hive-partitioned `particles_to_dataset`, Morton-sorted for spatial pushdown, optional DuckDB `query_sql`
- [x] **Step 25b:** Canonical ParticleData — per-particle `weight` plus scalar `species_charge` / `species_mass`; combined-storage codes split at the reader boundary
- [x] **Step 26:** `pypic convert` CLI — `fields`, `particles`, `all`; single-step writes `to_zarr`, multi-step `to_zarr_timeseries`, Icechunk tags supported

## Phase 9: Additional Readers

- [ ] **Step 23: `pypic.readers.vlasiator` — VLSV reader via analysator**
  `VLasiatorReader`. Two-grid strategy: FSgrid fields (`fg_b`, `fg_e`) read as uniform arrays; DCCRG fields (`proton/vg_rho`, `proton/vg_v`, `proton/vg_p`) regridded to uniform at `target_resolution` (default: FSgrid resolution). DCCRG cell IDs encode position + refinement level — decode then block-average/NN-repeat (BATSRUS AMR pattern, not `pypic.regrid`). Field map: `fg_b` → `B_*`, `fg_e` → `E_*`, `proton/vg_rho` → `n_s0`, `proton/vg_v` → `V_*`, `proton/vg_p` (6 components) → pressure tensor. Species auto-detected from VLSV population names. Auto-detect via `.vlsv` extension + signature. Optional dep: `analysator` under `vlasiator` extra. Tests mock analysator.

- [ ] **Step 35: `pypic.readers.vpic` — VPIC reader**
  `VPICReader`. VPIC writes per-rank binary (band-interleaved by field) or HDF5 via `vpic_decks`. Map: `cbx/cby/cbz` → `B_*` (cell-centered), `ex/ey/ez` → `E_*` (Yee edge), `jfx/jfy/jfz` → `J_*`, `rhob` → `rho_c`, per-species hydro → density/velocity/pressure tensor. Yee mesh destaggering (linear interp, `StaggerInfo(convention="staggered")`) — the Cartesian-Yee destagger primitive is already implemented at `src/pypic/_stagger.py` (tested, currently unwired); consume it here. Metadata from `info` dumps or deck header. Auto-detect: `global.vpc` or `info` file. Tests use synthetic fixtures.

- [ ] **Step 36: `pypic.readers.arms` — ARMS reader**
  `ARMSReader`. ARMS (Adaptively Refined MHD Solver) outputs HDF5 block-structured AMR. Regrid to uniform at `target_resolution` (BATSRUS pattern). Spherical geometry support (ARMS commonly run in spherical for coronal/heliospheric). `StaggerInfo(convention="staggered")` — CT for divergence-free B. Auto-detect via ARMS-specific HDF5 group structure. Tests use synthetic fixtures.

- [ ] **Step 42: `pypic.readers.openpmd` — openPMD reader (WarpX, PIConGPU, Smilei, FBPIC)**
  `OpenPMDReader`. One reader covers four major modern PIC codes (all emit openPMD natively, HDF5 + ADIOS2). High leverage vs per-code readers.
  **Iteration encoding:** `groupBased` (single file, `/data/<step>/`) and `fileBased` (one file per step, `%T` pattern). `variableBased` (ADIOS2 streaming) out of scope for v1.
  **Field map:** `meshes/B/{x,y,z}` → `B_*`, `meshes/E/{x,y,z}` → `E_*`, `meshes/J/{x,y,z}` → `J_*`, `meshes/rho` → `rho_c`. Per-species moments where emitted.
  **Stagger:** read per-record `position` array (0.0–1.0 offset) into `StaggerInfo` (Tier 2/3 per-component). Destagger to co-located grid via the Cartesian-Yee primitive at `src/pypic/_stagger.py` (tested, currently unwired) — wire it into the per-record `position` handling here.
  **Units:** read `unitDimension` 7-tuple + `unitSI` per record. Reconstruct `[units]` from ED-PIC particle records (`charge`, `mass`, `weighting`) + reference density from species moments. Fall back to `Normalization.identity()` (treat as SI) when insufficient metadata.
  **Particles:** `position/positionOffset/momentum/charge/mass/weighting/id` → canonical `ParticleData` (Step 25b form). Honor `macroWeighted` + `weightingPower` semantics.
  **Geometry:** `cartesian` → ours; `thetaMode` (FBPIC RZ-mode) needs azimuthal-mode reconstruction before destagger — punt to Phase 2.
  **Code dispatch:** `software` attribute drives a small table for code-specific quirks (path conventions, mass/charge unit drift from ED-PIC). Auto-detect via `openPMD` root attribute.
  Optional dep: `openpmd-api>=0.17` under `openpmd` extra. Tests use synthetic openPMD via openpmd-api. **Depends on:** Tier 1 ED-PIC vocabulary, Tier 2 per-component stagger.

## Extensions

- [ ] **Step 51: non-uniform / log-radial grids at runtime**
  `[grid.stretched]` has validated since the v1.0.x additive batch, but nothing downstream reads it: `GridInfo` carries one scalar `spacing` per axis and `coordinate_arrays()` returns `origin + (i+0.5)*dx`. A stretched deck would therefore load with *wrong cell positions*, every derivative, integral and slice on that axis silently off — measured with widths `[0.5, 0.6, 0.8, 1.0, 1.3]`, coords came out `[0.42 1.26 2.10 2.94 3.78]` against a truth of `[0.25 0.80 1.50 2.40 3.55]`. `_build_grid` (`readers/config.py`) now refuses such a deck with `UnsupportedGridError` instead of dropping the section, so the wrong numbers are unreachable; what remains is to honour the widths. Blocks ARMS spherical-$r$ (Step 36, whose blocker matrix entry is marked cleared on the schema side only), PLUTO's logarithmic grid patches, and Entity's QSpherical.
  The numerics are nearly free: `np.gradient(f, coords, axis=i)` already accepts a coordinate array where `operators.py` passes a scalar today, and is exact at interior points on a non-uniform mesh. The work is plumbing and invariants — an optional `GridInfo.axis_coords`, threading it through `divergence`/`curl`/`gradient` and the trapezoidal `integrate` branch, and reworking `_build_grid_from_dataset` (`grid.py`), which reconstructs the origin by inverting the uniform cell-centre formula and is the one place that genuinely assumes uniformity. The refusal that stands in for it landed as cleanup.md Phase 10 item 1; delete the `_READER_UNSUPPORTED` entry in `tests/test_schema_parity.py` when this lands, which that test will demand.
  Tests: cell centres match the declared widths; $d/dx$ of $x^2$ on a stretched axis is exact in the interior; a uniform deck is bit-identical to today; slicing a stretched axis preserves coordinates.
  **Depends on:** Step 5. **Unblocks:** Step 36 (ARMS), PLUTO and Entity readers.

- [ ] **Step 52: periodic stencils for the differential operators**
  `GridInfo.boundary` already records `("periodic", ...)` per axis and is read by nothing numerical — only validated, sliced on a selection, and serialized. Every operator calls `np.gradient`, which has no periodic mode and falls back to one-sided stencils at the first and last point. On an analytically divergence-free periodic field $\mathbf{B} = (\sin x \cos y, -\cos x \sin y, 0)$ over a $32^3$ box, interior $\max|\nabla\cdot\mathbf{B}|$ is `1.887e-15` while the full-grid figure is `9.546e-03`, and 18% of cells sit on a boundary face. Spectral and turbulence codes take this worst — they are periodic by construction, and their headline diagnostics are box-wide reductions.
  Implementation: `np.roll`-based central differences on axes `grid.boundary` marks periodic, selected per axis inside `divergence` / `curl` / `gradient`, leaving every non-periodic axis on the current `np.gradient` path bit-for-bit. `docs/conventions.md` § *Boundary treatment for finite differences* documents the ghost-cell workaround today and should point here instead once this lands.
  Tests: $\nabla\cdot\mathbf{B}$ of the field above is at roundoff over the *whole* grid, not only the interior; second-order convergence across the seam at three resolutions; a non-periodic grid reproduces today's values exactly; a mixed deck (periodic $x$, open $z$) wraps only the flagged axis.
  **Depends on:** Step 10. **Pairs with:** Step 51 — both replace a scalar-spacing assumption inside the same three functions, so landing them together avoids touching that code twice.

- [ ] **Step 53: non-orthogonal metrics — what GRPIC would take**
  Scoping entry, not yet scheduled. `CoordinateGeometry.metric_factors` returns three diagonal scale factors for $ds^2 = h_1^2dx_1^2 + h_2^2dx_2^2 + h_3^2dx_3^2$ — orthogonal coordinates only. Kerr-Schild and other GR metrics carry off-diagonal $\gamma_{ij}$, so Entity's GRPIC mode (and any Kerr-Schild MHD output) has no representation today.
  **pypic is not fundamentally incompatible, if the boundary is drawn at the reader.** Projecting onto a local orthonormal tetrad at load time makes $|B| = \sqrt{B_1^2+B_2^2+B_3^2}$ true again, and with it every pointwise quantity — magnitudes, beta, field-aligned decomposition, the pressure tensor, Mach numbers — works unchanged. That is the same shape as the existing rule that readers destagger to co-located grids: the reader absorbs the representation problem so downstream stays simple. What tetrad projection does *not* fix is the differential operators, which need $\nabla\cdot\mathbf{A} = (1/\sqrt{\gamma})\,\partial_i(\sqrt{\gamma}A^i)$ and a Levi-Civita curl, plus the contravariant/covariant distinction the canonical names do not currently carry.
  So the staging is: (a) reader-side tetrad projection plus a `StaggerInfo`-style provenance record of the metric, which buys Entity SRPIC and the pointwise surface of GRPIC; (b) a general-metric branch in `coordinates/operators.py`, generalizing `metric_factors()` from three scale factors to a metric tensor — the same seam Steps 19b and 43b already widen for spherical and cylindrical; (c) a `[coordinates].metric` schema block naming the metric and its parameters, which is where TASKS-schema-extension.md would pick it up. Nothing before a concrete reader needs it.
  **Depends on:** Step 51 (Entity's QSpherical is stretched before it is curved), Step 19b.

- [ ] **Step 19b: spherical regridding for `pypic.regrid`**
  Extend `regrid()` / `common_grid()` / `align_grids()` to handle `GeometryType.SPHERICAL`. Metric-factor-aware interpolation on $(r, \theta, \phi)$ — $\sin\theta$ Jacobian matters near the poles. Pole handling: clamp $\theta \in [\epsilon, \pi - \epsilon]$ or local Cartesian chart near each pole. Intersection: $r$ like Cartesian; $\theta$ in $[0, \pi]$; $\phi$ modulo $2\pi$ with wrap. Primary use: comparing ARMS runs at different angular resolutions (Step 36); unblocks Step 20b.
  Tests: spherical harmonic round-trip ($Y_\ell^m$, residual bounded by truncation order); pole fidelity ($\cos\theta$ field, zero error at poles within tolerance); $\phi$-wrap correctness at the $2\pi$ seam.
  **Depends on:** Step 19.

- [ ] **Step 20b: volume-weighted comparison norms**
  Add metric-factor integration to `compare_fields()` / `field_comparison_report()` so L2/L∞ weight each cell by $\sqrt{|g|}\,d^n x$. Step 20 is correct on uniform Cartesian ($\Delta V$ cancels) but wrong on spherical (poles, $r=0$). API: `compare_fields(..., weighted: bool = False)` — defaults preserve Cartesian behavior. L∞ unaffected (pointwise). Tests: radial shell L2 = analytic shell volume; Cartesian regression unchanged.
  **Depends on:** Step 19b.

- [x] **Step 43:** `pypic.reductions` — `reduce(axis, reduction=..., *, weight=None)` over ten ops, `attrs["reduction"]` provenance, composes with selections; CLI `pypic reduce apply`. Carry-overs: Steps 43b, 43c

- [x] **Step 50:** 2D support in the differential operators — `divergence` /
  `curl` / `gradient` take `d3: float | None = None`, dropping the terms that
  differentiate along $x_3$ where $\partial/\partial x_3 \equiv 0$; `div_b`,
  `max_div_b` and `compute("div_B")` follow, and `pypic validate` reports
  `max |div B|` on 2D grids instead of skipping. Resurrected `compute("psi")`,
  which the 3D-only gate had made unreachable on every grid — 2D was refused
  by the gate, 3D by the flux function's own 2D requirement.

- [ ] **Step 43b: spherical / cylindrical Jacobian-aware `reduce(integrate)`**
  Wire `CoordinateGeometry.metric_factors(...)` into the `integrate` branch so reductions on spherical/cylindrical return $\int f \, h_i \, dx^i$ instead of raising `NotImplementedError`. Build Jacobian $J = \prod_i h_i$ over reduced axes (function of *surviving* coords — $r$ for spherical $\theta$-integration, $r\sin\theta$ for $\phi$), broadcast, multiply field by $J$ before trapezoidal. Cartesian unchanged ($h_i=1$). Tests: spherical shell volume = $\frac{4}{3}\pi(r_2^3 - r_1^3)$ at machine precision; cylindrical disc area = $\pi r^2$ to trapezoidal order; Cartesian regression unchanged. `metric_factors()` is already at `coordinates/geometry.py:73-116` for all three geometries, so no longer blocked on 19b.
  **Depends on:** Step 5, Step 43.

- [ ] **Step 43c: unit-aware `reduce()`**
  After `reduce(reduction="integrate")` along $n$ axes, the SI unit shifts by $n$ length factors (m⁻³ → m⁻² → m⁻¹). Today `quantity_type`/`si_unit` are preserved unchanged, so `in_si()` is off by one length factor per reduced axis (workaround: multiply by `normalization.length_si**n`). Options: (a) generalize the `unit_dimension` 7-tuple arithmetic so attrs carry correct post-reduction dimensions; (b) add shifted canonical names (`column_density`, `surface_brightness`); (c) hybrid. Pairs with Step 20b's volume-weighted-norms work (same unit-dim concerns).
  **Depends on:** Step 43. **Pairs with:** Step 20b (shares the `unit_dimension` 7-tuple arithmetic concerns; ships independently).

- [ ] **Step 15b: breadth-first frame-chain resolution**
  `resolve_transform` (`coordinates/transforms.py`) builds the directed edge set (each declared transform plus its inverse), then tries a direct lookup and a single intermediate hop — a hard ceiling of two edges. `transform_to` (`dataset.py`) calls it once, so it inherits that ceiling: a three-hop request (simulation→GSE→GSM→SM) raises `ValueError("No transform path ...")` even though every edge is declared. Tie-breaking is dict insertion order, where each transform's inverse edge is inserted right behind its forward edge, and there is no cycle detection anywhere.
  schema.md § 2 (*Chain resolution*) promises something stricter — BFS from `[coordinates].frame` returning the **shortest** path, ties broken on TOML declaration order, cycles raising at apply time — and the promise is repeated in `dataset.py` / `containers.py` docstrings, in `schema/_export.py`'s runtime-enforced list, and in the generated `simulation.schema.v1.0.json`. So the documented contract, not the code, is the reference; implement it rather than weaken four documents.
  BFS over the edge set is small (a queue, a visited set, path reconstruction, then `compose_transforms` along the path). Two things it must preserve: the inverse-edge synthesis, and the signed-permutation restriction `transform_to` enforces on the composed rotation. Tests: three-hop chain resolves and round-trips; a shorter path wins over a longer one regardless of declaration order; two equal-length paths pick the earlier-declared one; a cycle raises instead of looping. Regenerate the JSON Schema afterwards only if the wording there changes.
  **Depends on:** Step 15.

- [ ] **Step 40: time-dependent frame transforms**
  Extend `FrameTransform` to per-timestep rotations. Primary use: GSE↔GSM via dipole tilt angle. Two paths:
  - **Parameter-driven:** `parameter = "dipole_tilt"` in `[coordinates.transforms]` names a time-varying quantity; rotation recomputed each step.
  - **SPICE:** `from_spice(frame_a, frame_b, epoch)` builds a transform from NAIF kernels (GSE↔HEE↔RTN, planetary frames). Optional dep: `spiceypy` under `spice` extra. Useful for spacecraft-observation comparison.
  `FieldDataset.transform_to(frame, *, epoch=None)` gains optional epoch. Static transforms unchanged. Tests: round-trip GSE→GSM→GSE at known tilt angles against published rotation matrices.
  **Depends on:** Step 15.

- [ ] **Step 44: field-line tracer & mapping infrastructure — symplectic integrator, periodic tricubic, curvature step control, squashing factor $Q$**
  Additive opt-in upgrades to `pypic.traces` + new `pypic.maps` module; existing `trace_field_line_adaptive` (Dormand-Prince 5(4) + trilinear + elementary (I) step control) stays default. Motivates: fusion/Poincaré topology (44a, 44b), solar coronal mapping (44d–44g), magnetospheric X-line detection (44e–44g). 44e/44f are direct lifts from Predictive Science's MapFL Fortran tracer; 44g ($Q$) builds on both. Reference: Hairer-Lubich-Wanner *Geometric Numerical Integration* (2006), Ch. II.1 + V. No new mandatory deps — scipy ships cubic `RegularGridInterpolator` and periodic `CubicSpline`.

  **44a — implicit midpoint integrator.** New `trace_field_line_symplectic` (fixed `step_size`, 1-stage Gauss-Legendre RK solved by 2-3 fixed-point iterations). Preserves discrete symplectic 2-form: invariants like $r^2$ on closed orbits stay *bounded* instead of secularly drifting. Take 5-10× larger steps for same picture quality on Hamiltonian-like flows → net wall-clock 1.2-1.5×. **Tests:** $r^2$ conservation on $\mathbf{B}=(-y,x,0)$ over $10^4$ steps; $\psi$-conservation on 2D analytic flux; closed-circle Poincaré collapses to a single ring.

  **44b — tricubic interpolation kwarg.** Promote `VectorFieldInterpolator.from_dataset(..., method="cubic")`. Error $O(h^2) \to O(h^4)$ at 3-5× per-eval cost; preserves $\nabla\hat{\mathbf{B}}$ smoothness across cell faces (44a is more sensitive to this than DP). **Tests:** convergence on $\mathbf{B}=(-y,x,0)$ at three resolutions; Harris ($\tanh(y/L)$) smoothness — no spurious $\hat{\mathbf{B}}$ flips at cell faces.

  **44c — $\mathbf{A}$-based reconstruction (optional).** When a reader exposes `A_1/A_2/A_3`, interpolate $\mathbf{A}$ cubically and compute $\mathbf{B}=\nabla\times\mathbf{A}$ analytically. Guarantees $\nabla\cdot\mathbf{B}=0$ at every interior point — matters near nulls/separatrices. Punt until a reader writes `A_*` (BATSRUS HDF5 does, iPIC3D doesn't).

  **44d — periodic tricubic splines for $\phi$/$\theta$.** Knob: `periodic_axes: tuple[int,...] = ()` on `VectorFieldInterpolator.from_dataset()` — flagged axes use `CubicSpline(..., bc_type="periodic")` per spline line, accumulated into a tensor-product cubic block. Fixes the C⁰ discontinuity in $\hat{\mathbf{B}}$ at the $\phi=2\pi$ seam that visibly bends field lines. **Tests:** seamless across $\phi=2\pi$ on $\mathbf{B}=(-\sin\phi,\cos\phi,0)$; regression vs 44b in seam-free interior.
  **Depends on:** Step 19b (gates the geometry), 44b.

  **44e — curvature-based step control (non-default).** `step_control: Literal["error","curvature"] = "error"` kwarg alongside the elementary (I) error controller. Curvature path keeps $\|\hat{\mathbf{B}}_{n+1}-\hat{\mathbf{B}}_n\|\cdot h/\Delta s \approx$ `over_rc` (default 0.0025), clamped by `local_mesh_factor × min(\Delta x_i)`. Wins 2-5× on long quasi-laminar traces (PFSS, dipole magnetospheres, tokamak equilibria); auto-tightens in sharp-gradient regions. **Depends on:** `pypic.traces._tracing`.

  **44f — endpoint-only mode for `trace_field_lines_adaptive`.** `return_endpoints_only: bool = False` kwarg skips materializing `FieldLine.points`/`arc_lengths`, returns a `FieldLineEndpoint` namedtuple `(start, end, arc_length, termination_reason)`. Memory: a 1000×1000 map at 20k steps is $\mathcal{O}(\text{TB})$ in full-trace mode. Kernel of 44g ($Q$ = 5 endpoint traces per seed). **Depends on:** `pypic.traces._tracing`.

  **44g — `pypic.maps` module + squashing factor $Q$.** Open/closed classification, footpoint mapping $\mathbf{r}(\theta_0,\phi_0)\mapsto\mathbf{r}(\theta_1,\phi_1)$, the Titov-Démoulin squashing factor $Q$, and the Pariat-Démoulin signed-log $\text{slog}(Q) = \mathrm{sign}(B_r)\cdot\log_{10}(Q/2 + \sqrt{(Q/2)^2-1})$. Flagship: `squashing_factor(ds, seeds, *, h=1e-4, ...)` — at each seed, trace 1 central + 4 perp-pair neighbors, central-difference for the 2×2 Jacobian $D$, return $Q = \|D\|_F^2/|\det D|$. Verbatim MapFL `getq` recipe, including the $\sin\theta < 5\times10^{-3}$ pole switch to Cartesian basis. **Use cases:** solar coronal connectivity (open-field maps, coronal-hole boundaries), magnetospheric QSL mapping, pre-flare $Q$-line detection. **Tests:** $Q\approx 2$ on uniform field; $Q\to\infty$ across analytic separatrix; numerical comparison vs MapFL on Titov-Démoulin flux rope.
  **Depends on:** 44f; benefits from 44d on spherical PFSS.

  **Not in scope:** variational integrators; 2-stage order-4 Gauss-Legendre (until 1-stage benchmarked); Boozer/Hamada flux-coordinate Hamiltonians; full MapFL diagnostic surface beyond $Q$ (expansion factor, $K$-factor, magnetic dips — add piecewise when a concrete use case appears).

## Phase 10: Ecosystem Integration

- [ ] **Step 27: `pypic.interop` — yt, PlasmaPy, SpacePy adapters**
  `pypic.interop.yt`: `to_yt_dataset(fds) -> yt.StreamDataset` — maps canonical fields to yt field tuples, sets domain from GridInfo. Cartesian only.
  `pypic.interop.plasmpy`: `to_plasmpy_plasma(fds, species_index) -> dict` — extracts density, temperature, |B| as `astropy.units.Quantity` (SI via `normalization.to_si()`). Dict, not PlasmaPy Plasma object (API unstable). `validate_against_plasmpy()` cross-validates derived quantities.
  `pypic.interop.spacepy`: `from_spacepy_dm(dm, grid, normalization) -> FieldDataset` — converts SpacePy DataModel (CDF/ISTP) with user-supplied grid and field map.
  Optional deps: `yt>=4.3`, `plasmapy>=2024.7`+`astropy>=6.0`, `spacepy>=0.6` — each under its own extra. Import-guarded with helpful install message. Tests mock external libraries.

- [ ] **Step 28: ecosystem documentation page**
  `docs/ecosystem.md` — positioning: pypic (multi-code reader + normalization + derived), PlasmaPy (reference formulas + constants), SpacePy (spacecraft/CDF), yt (AMR + volume rendering). Code examples for each adapter. "When to use which tool" decision guide.

- [ ] **Step 29: `pypic.interop.spase` — SPASE XML metadata export**
  `to_spase_xml(fds, *, resource_id, contact, description) -> str` generates a SPASE `NumericalData` XML. Maps `simulation.toml` sections to SPASE elements: `[model]` → `SimulationRun`, `[grid]` → `SpatialDescription`, `[units]` → `Units` on each Parameter, `[[species]]` → `Particle`, canonical fields → `Parameter` (`ParameterKey`/`Name`/`Description`/`Units`). `to_spase_file(fds, path, **kwargs)` writes to disk. No external deps (stdlib `xml.etree.ElementTree`). Enables publishing to CDAWEB/VHO/CCMC. Tests use synthetic FieldDatasets.

## Phase 11: Virtual Probes & Spacecraft

- [ ] **Step 41: `pypic.probes` — virtual probe sampling**
  `Probe` frozen dataclass: named point `(x, y, z)`. `ProbeArray`: collection (detector arrays, satellite constellations). `ProbeTrajectory`: time-varying `(t, x, y, z)` — spacecraft orbit / moving detector. A fixed probe is a degenerate trajectory.
  Core functions:
  - `sample(probe, dataset) -> dict[str, float]` — interpolate all fields at the probe position for one timestep. Reuses `RegularGridInterpolator` from `traces/_sampling.py`.
  - `sample_timeseries(probe, simulation, steps) -> TabularData` — sample across timesteps (time, B_1, B_2, B_3, ...).
  - `sample_trajectory(trajectory, simulation) -> TabularData` — sample along a moving path.
  - `sample_array(probes, dataset) -> TabularData` — all probes at one timestep, one row per probe.
  Schema: `[[probes]]` section (schema.md § 2 — `[[probes]]`). Probes from config available via `Simulation.probes`. CLI: `pypic probe <path> --name NAME --step all --field FIELD`.
  **iPIC3D integration:** iPIC3D outputs native virtual satellite data at fixed locations (high cadence). `AuxiliaryDataReader` already loads as `TabularData`. Probes can: (a) define new and resample, (b) load native, (c) compare (native = higher time res; resampled = all derived fields).
  **Depends on:** `traces/_sampling.py`, `TabularData`.

- [ ] **Step 41b: SPICE-driven probe trajectories**
  `ProbeTrajectory.from_spice(target, observer, frame, epochs)` builds a trajectory from NAIF SPICE kernels via `spiceypy`. Direct comparison: load sim, define trajectory matching MMS/Cluster/PSP orbit, sample fields along real path, compare with CDF observations (SpacePy adapter, Step 27). Optional dep: `spiceypy` under `spice` extra.
  **Depends on:** Step 41, Step 40.

## Phase 12: Cross-Project Integration

> **Tier-3 canonical names (locked pre-v1.0).** Cross-tool work below uses `<field>[_s<N>][_<i>]` with species qualifier between field name and index (`B_1`, `V_s0_1`, `P_s0_11`, `q_s0_1`). HDF5 §4.1 and Zarr §4.2 stores must use these — `B1`, `V1_s0`, `P11_s0` are not emitted by any pypic-aware tool. rustpic and webpic wire directly to Tier-3; no migration shim since neither has shipped.

- [x] **Step 37:** `pypic.server` — Arrow IPC over WebSocket (`app` / `routes` / `arrow` / `stream` / `protocol`), the `server` extra, and the `pypic serve` command

- [x] **Step 37a:** Typed server exceptions — `PypicError` and its subclasses in `pypic.exceptions`, one dispatcher unifying HTTP status codes and WebSocket `ErrorFrame` kinds

- [ ] **Step 37b: selection provenance — `attrs.selections` round-trip**
  Symmetric counterpart to `attrs["reduction"]`. Today `BoxSelection`/`PlaneSelection`/`SphereSelection` produce datasets with no recorded region — box/plane partly recoverable from post-slice `grid.lower/upper/dimensions`, but `SphereSelection` loses centre, radius, keep-direction once NaN mask lands. Webpic's `{selection, axis, reduction}` wire format round-trippable only when selection state survives `to_zarr`/`from_zarr`. Design: (a) **list** at root (`attrs.selections`), not per-field — selections apply globally and chained `Box → Sphere → reduce` needs ordered composition; (b) entries typed by `kind` (`"box"`|`"plane"`|`"sphere"`) with parameters in code units + pointer to active normalization (radii interpretable across normalizations); (c) Pydantic `[[selection]]` records in `pypic.schema._models`, JSON Schema regen + drift test; (d) replay via `Selection.from_attrs(record)` classmethods. Defer until 37 forces the wire-format contract.
  **Depends on:** Step 37.

- [ ] **Step 38: `pypic.readers.rustpic` — Rust PIC code reader**
  Reader for rustpic's schema.md-conformant HDF5 output. Rust writes the canonical layout (Section 4) directly, so essentially `SimpleReader` + rustpic-specific metadata extraction + validation. pypic serves as the **reference implementation** — validate Rust-computed derived quantities against Python on the same problem. Cross-project integration tests: identical ICs, compare via `field_comparison_report()`. Auto-detect via HDF5 `model` attribute = `"rustpic"`.
  **Depends on:** Step 20.

- [ ] **Step 39: webpic data pipeline documentation**
  End-to-end guide: rustpic (Rust sim) → HDF5 → pypic (Python analysis) → Starlette/FastAPI + Arrow IPC → webpic (Three.js/WebGPU). Documents schema.md contract syncing Python, Rust, JS. Selection round-trip: viewer UI → server `Selection` → `FieldDataset` slice → Arrow IPC → GPU buffer. Coordinate transform: viewer requests frame → server `transform_to()` → transformed data streamed. Covers auth model, chunked transfer, WebSocket option for time-series animation.

## Schema extension backlog

Proposed additions to the `simulation.toml` schema — code-readiness notes
for MHD / PIC / hybrid codes not yet covered by the v1.0 validator — live
in [TASKS-schema-extension.md](TASKS-schema-extension.md). Read it before
changing `pypic/schema/_models.py`.

## Dependency Graph

```
Step 19 (regrid)         ←── 19b (spherical) ←── 20b (volume-weighted norms)  pairs-with 43c
Step 15 (transforms)     ←── 40 (time-dependent) ←── 41b (SPICE trajectory)
                         ←── 15b (BFS chain resolution)
Step 5 (FieldDataset)    ←── 24, 25 (Zarr/Arrow)    ←── 26 (convert CLI)
                                                   ←── 24b (VirtualiZarr), 24c (Icechunk)
                                                   ←── 25b (canonical ParticleData)
                         ←── 23, 35, 36, 42 (readers)
                         ←── 27 (interop adapters)
Step 11 + Step 5         ←── 43 (reductions) ←── 43b (Jacobian), 43c (units)
Steps 24, 25             ←── 37 (Arrow IPC server, shipped) ←── 37a (typed exceptions, shipped)
                                                   ←── 37b (selection provenance), 39 (docs)
Step 20                  ←── 38 (rustpic reader)
pypic.traces             ←── 44 (field-line tracer + pypic.maps)
                              44a (implicit midpoint), 44b (tricubic), 44c (A-based)
                              44d (periodic tricubic) ←── 19b, 44b
                              44e (curvature step), 44f (endpoint-only)
                              44g (pypic.maps + Q)    ←── 44f, (44d on spherical)
traces/_sampling.py      ←── 41 (probes) ←── 41b (SPICE)
```

Recommended order: 37b next (37 and 37a shipped); 15b is small and closes a documented-contract gap; 38 needs a rustpic dump; 39 follows 37; 23/35/36/42 anytime; 40 unblocks 41b; 43b/43c anytime after 43; 44a-g modular (44f is the kernel for 44g); 19b precedes 20b and 44d.
