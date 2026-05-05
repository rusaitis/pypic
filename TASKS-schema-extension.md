# simulation.toml schema — extension backlog

Readiness review for popular MHD / PIC / hybrid codes against the v1.0
Pydantic validator (`src/pypic/schema/_models.py`). Organized as a
release plan: a **v1.0.x additive batch** that lands now (no v1.0
documents invalidated), a **v1.1 shape-change batch** bundled with the
readers that demand it, and **Tier 3 deferred** items. Pointers cite
line numbers in `_models.py` after the v1.0 refactor (commit `8030876`).

**Two surfaces, one document.** Items affect either the **TOML schema**
(`pypic.schema._models`, versioned via `schema_version`) or **runtime
metadata** (`FieldDataset.attrs`, `StaggerInfo` — not versioned by the
schema). Runtime-metadata items land whenever; TOML-schema items must
respect the v1.0 contract (no required field added, no existing field
removed or retyped, no field renamed).

## v1.0.x — additive batch (ship now)

Purely additive changes: new optional fields, expanded enum values,
new runtime metadata. Existing v1.0 documents continue to validate;
the `schema_version = "1.0"` string stays valid. Bundle as one
"vocabulary + metadata" pass. The window for cheap additions closes
once external producers (rustpic, third-party adopters) start emitting
v1.0 documents we can't perturb.

### TOML-schema additions

- [ ] **PIC field solver vocabulary — PSATD vs PS.**
  `PICFieldSolver` (`_models.py:43`) exposes `"pseudo-spectral"` but
  conflates PS (FD-in-time, spectral-in-space) with PSATD (analytic in
  time, spectral in space). Add `"psatd"` and `"spectral-azimuthal"`
  (FBPIC's RZ-mode decomposition). Both are dominant production solvers
  in WarpX and FBPIC.

- [ ] **PIC current/charge smoothing knobs.**
  `PICSolver` (`_models.py:354`) lacks `current_smoothing` /
  `charge_smoothing` fields (binomial passes, compensator filters).
  Standard in every production PIC code. Oddly already present on
  `HybridSolver` (`_models.py:391`).

- [ ] **Adopt openPMD ED-PIC vocabulary across PIC literals.**
  The openPMD ED-PIC extension (`EXT_ED-PIC.md`, extension ID 1) defines
  the *de facto* PIC vocabulary used by WarpX, PIConGPU, Smilei, and
  FBPIC. While the openPMD standard itself has been frozen at v1.1.0
  since Feb 2017, the ED-PIC vocabulary is what these production codes
  actually emit on disk and is the natural alignment target for our
  enums. Concrete additions:
  - `PICFieldSolver` (`_models.py:43`) gains `"lehe"` (Lehe stencil),
    `"ck"` / `"ckc"` (Cole-Kärkkäinen compact stencil), `"pstd"`,
    `"gpstd"`. (`"psatd"` already covered above.)
  - `PICPusher` (`_models.py:42`) gains `"llrk4"` and `"free-streaming"`.
  - New literal `ChargeCorrection = "marder" | "langdon" | "boris" |
    "hyperbolic" | "spectral" | "none"` on `PICSolver`.
  - New literal `ParticleShape = "ngp" | "cic" | "tsc" | "pqs"` on
    `Species` or `PICSolver` (NGP=order 0, CIC=1, TSC=2, PQS=3).
  - New literal `CurrentDeposition = "esirkepov" | "zigzag" |
    "villabune" | "direct-boris" | "direct-morse-nielson" | "none"` on
    `PICSolver`.
  All purely additive — existing v1.0 docs continue to validate.

- [ ] **`ModelType` — admit Vlasov / gyrokinetic.**
  `ModelType = "PIC" | "MHD" | "hybrid"` (`_models.py:34`) excludes
  continuum-Vlasov (Vlasiator, Gkeyll Vlasov-Maxwell) and gyrokinetic
  (GENE, GS2, GX, Gkeyll-GK). Add `"vlasov"` and `"gyrokinetic"`. Solver
  vocabularies can land later — the root enum gate is the immediate
  blocker.

- [ ] **Tracer flag on `[[species]]`.**
  Add `tracer: bool = False` to `Species` (`_models.py:500`). PIC codes
  routinely write a tagged subset for trajectory tracking that must not
  contribute to charge/current deposition. OSIRIS, Smilei, TRISTAN-MP
  all support this natively. Open question before locking the API:
  some codes distinguish *test* particles (passive, no back-reaction
  on fields) from *tracer* particles (full dynamics, just tagged for
  output). If the boolean turns out to be insufficient, a follow-up
  `tracer_kind: "test" | "tagged"` field can land additively.

- [ ] **Time-integration vocabulary expansion.**
  `TimeScheme = "fixed" | "adaptive" | "subcycled"` (`_models.py:37`)
  doesn't express RK substages (`vl2`, `rk2`, `rk3`, `ssprk3`),
  Strang/Lie operator splitting, or IMEX-RK schemes for stiff source
  terms (radiation, cooling, chemistry). These are *the*
  time-integration vocabulary across modern grid codes (Athena++,
  PLUTO, FLASH). Additive shape: expand the literal and add an
  optional `splitting: "strang" | "lie" | "godunov" | None` field on
  `[time]`. No restructuring of the section.

- [ ] **Stochasticity / ensemble metadata on `[run]`.**
  Add optional `random_seed: int | None` and
  `ensemble: {member_id, total} | None` to `Run` (`_models.py:147`).
  Required for ensemble runs (cosmological PIC, turbulence
  realizations). Cheap to add now; otherwise readers fall back to the
  `x-` extension namespace.

- [ ] **AMR kind discriminator.**
  `GridAMR` (`_models.py:207`) implicitly assumes block/patch AMR with
  a global `block_size`. Octree codes (RAMSES, MPI-AMRVAC) have no
  fixed block size — each leaf is one cell. Add optional
  `amr_kind: "block" | "patch" | "octree"` defaulting to `"block"`
  (current implicit behavior). Without it, octree codes must use the
  external-mesh escape hatch.

- [ ] **AMR temporal subcycling flag.**
  Athena++ and AMReX-based codes sub-cycle different AMR levels at
  different effective timesteps. Lightest fix lands additively in
  v1.0.x: optional `level_subcycling: bool = False` on `[grid.amr]`.
  A per-level `dt_factor` array would be a future v1.1 add if needed.

- [ ] **Ghost cell counts.**
  Many codes need ghost cell metadata for downstream edge-derivative
  analysis. Optional one-liner: `ghost_cells: AxisInt | None = None`
  on `[grid]`.

- [ ] **Per-rank / multi-file output layout.**
  `_OutputBase` (`_models.py:547`) describes a single `dir`,
  `step_interval`, `format`, `precision`. VPIC writes one
  band-interleaved binary per MPI rank per dump; WarpX/openPMD writes
  per-process files plus an index. Add optional fields:
  `file_pattern: str | None` (e.g. `"step_{step:06d}/rank_{rank:05d}.h5"`),
  `files_per_step: int | None`,
  `partition: "by_rank" | "by_field" | "monolithic"`. All optional —
  existing single-file readers unaffected. Required for VPIC (Step 35),
  nice-to-have for AMReX-based PIC.

- [ ] **Anisotropic closure enum values.**
  `Closure` (`_models.py:40`) is `"isothermal" | "adiabatic" |
  "polytropic" | "braginskii"`. Add `"cgl"` (double-adiabatic),
  `"10moment"`, `"14moment"`. Pure enum expansion — codes can declare
  a multi-moment closure on `[[species]]` even before the schema
  carries the full anisotropic `gamma_eos` tuple (which is v1.1 because
  it changes the existing scalar field type).

- [ ] **Restart granularity (additive fields).**
  `Restart.from_` (`_models.py:492`) is a single path with no partial-
  restart or hot/cold distinction. v1.0.x adds the additive parts:
  optional `restore: list["fields" | "particles" | "auxiliary"]`
  (partial restart — fields-only continuation) and
  optional `mode: "hot" | "cold"`. The `from: str → str | list[str]`
  widening (multi-file VPIC manifest) is type-changing on an existing
  field and lands in v1.1.

- [ ] **`thetaMode` / RZ azimuthal-mode geometry.**
  FBPIC decomposes EM fields into a small number of azimuthal modes
  (m=0 cylindrically symmetric + the lowest few m components) on an
  (r, z) grid, then reconstructs the full 3-D field on demand.
  openPMD records this as `geometry = "thetaMode"` with a
  `geometryParameters = "m=N,..."` string. Currently
  `[coordinates].geometry` is `"cartesian" | "spherical" |
  "cylindrical"` — none of these covers the modal decomposition
  cleanly. Sketch: `[coordinates].geometry += "thetaMode"`; new
  optional `[coordinates.modes]` sub-table with `n_modes: int` and
  `mode_indices: list[int]`. Reader's job is either (a) reconstruct
  full 3-D on read (default), or (b) expose the per-mode arrays as
  separate fields when requested. **Step 42 (openPMD reader) defers
  thetaMode to Phase 2** — landing this entry first lets the reader
  describe the data losslessly when it gets there.

### Runtime-metadata additions (FieldDataset / StaggerInfo)

Not versioned by `schema_version`; land independently of the
TOML-schema additions above.

- [ ] **`unitDimension` per-field metadata.**
  openPMD records each array's dimensional fingerprint as a 7-tuple
  `[L, M, T, I, Θ, N, J]` (powers of SI base units). Adopt this as a
  derivable property on `FieldDataset` per-field metadata, sitting
  alongside the existing `quantity_type` / `si_unit` / `latex` keys.
  Two design questions before landing:
  - **Compute or store?** For canonical fields the 7-tuple is fully
    determined by `quantity_type` (e.g. `magnetic_field` →
    `[0, 1, -2, -1, 0, 0, 0]`). Recommend a static lookup table per
    `QuantityType` enum value; user-registered custom fields supply
    an explicit tuple at registration time.
  - **Serialize through HDF5/Zarr?** Step 24's `to_zarr` already
    persists `dataset.attrs` automatically — recommend compute on
    read, write on output, so an openPMD ↔ pypic round-trip
    preserves the attribute without adding a schema-config-level
    entry.
  Not a replacement for `[units]`: `[units]` encodes the
  *normalization paradigm* (PIC reference species, MHD reference
  quantities); `unitDimension` encodes the *dimensional fingerprint*.
  Both useful, different questions. Prerequisite for any openPMD
  reader that needs to honor per-record `unitDimension` / `unitSI`
  attributes (TASKS.md Step 42).

- [ ] **Per-component stagger via `position` array.**
  `Grid.stagger: "cell" | "node" | "staggered"` (`_models.py:231`) is a
  single coarse enum that loses information for any code with a true
  Yee mesh: `Bx` lives at `[0.5, 0, 0]` (face-centered in x), `By` at
  `[0, 0.5, 0]`, `Ex` at `[0, 0.5, 0.5]` (edge-centered), and so on.
  Our reader already destaggers to a co-located grid, so the metadata
  is provenance-only — but provenance worth recording precisely. Adopt
  openPMD's `position` semantics: a per-record array of length `ndim`
  with values in `[0.0, 1.0)` describing the relative offset on the
  cell. Lives on `StaggerInfo` (per-field), not on `[grid]` (per-
  dataset). Existing enum stays as a top-level shorthand;
  per-component `position` is the new precise form. Lets Yee-mesh
  PIC, BATSRUS face-centered B, and any future co-located write-out
  describe their native stagger losslessly.

## v1.1 — shape-change batch (ship with readers)

Each item restructures an existing model, retypes an existing field,
or introduces a sub-model that reshapes the surface. Once any v1.1
change lands, all v1.0 documents need re-validation against the new
shape, so amortize the disruption: bundle into one cut tied to a
specific reader rollout.

Recommended pairing of shape work to reader rollouts:

| Reader trigger | v1.1 items unlocked |
|---|---|
| Vlasiator (Step 23) — lossless VDFs | `[velocity_mesh]` |
| ARMS spherical (Step 36) | stretched / non-uniform grids |
| Smilei collisional / EPOCH | `[[collisions]]`, `[[ionization_chains]]` |
| EPOCH-QED / Smilei-QED / Zeltron | QED / radiation reaction |
| PIC PML on E + B; thermal-bath particles | per-field BCs + driver linkage |
| Multi-cadence runs in production | repeatable `[[output.fields]]` |
| Gkeyll, Hakim two-fluid 10-moment (full anisotropy) | `gamma_eos` tuple form |
| VPIC (Step 35) per-rank restart manifest | restart `from_` widening |
| GENE, GS2, GX, Gkeyll-GK | high-D grids |

- [ ] **`[velocity_mesh]` for continuum-Vlasov codes.**
  Vlasiator stores per-cell distribution functions on a 3-D Cartesian
  velocity grid with sparse-block storage. The schema currently has no
  way to describe the v-grid extent (`vmin/vmax`), block size, or
  sparsity policy. Blocks `vlasiator` reader (TASKS.md Step 23) from
  ever describing VDF data losslessly — only fluid moments work today.

- [ ] **Stretched / non-uniform grids.**
  `Grid.spacing` (`_models.py:228`) is a single scalar per axis. PLUTO
  log-radial, Athena++ stretched, FLASH per-block-non-uniform, ARMS
  spherical with stretched-r cannot be described losslessly. The
  external-mesh escape hatch (`source = "grids/mesh.h5"`,
  `_models.py:230`) defers but the in-schema fields claim uniform
  spacing the data won't satisfy.
  Sketch: discriminate `kind: "uniform" | "stretched" | "external"`;
  for `stretched`, allow `spacing: list[list[float]]` (per-axis cell
  widths) or a `stretch_factor: float` for geometric grids.
  **Bites ARMS in spherical mode immediately (Step 36).**

- [ ] **`[[collisions]]` table for collisional PIC.**
  Smilei, EPOCH, OSIRIS-collisional, PIConGPU collisions all need
  per-pair Coulomb collision coefficients (and BGK or Monte-Carlo
  variants). No current shape. CLAUDE.md's v1.1 placeholder
  `physics.collisional` is just a flag.
  Sketch: `[[collisions]]` entries keyed by `species_pair = ["e", "i"]`,
  with `model: "coulomb" | "bgk" | "monte-carlo"`, `coulomb_log: float`,
  `temperature_ref: float | None`.

- [ ] **`[[ionization_chains]]` for atomic-physics PIC.**
  EPOCH, Smilei, OSIRIS support BSI / ADK / multi-photon ionization.
  Need to express that He⁺ and He²⁺ are charge states of the same atom
  (the schema currently says "treat as separate `[[species]]`", losing
  the parent linkage). Sketch: `[[ionization_chains]]` with
  `parent: "He"`, `states: ["He0", "He+", "He2+"]`, `model: "adk" | "bsi" | ...`.

- [ ] **QED / radiation-reaction module.**
  EPOCH-QED, Smilei-QED, OSIRIS, Zeltron support synchrotron radiation
  reaction, Breit-Wheeler pair production, Compton scattering. Needs
  photon species + cross-section table identifiers. CLAUDE.md mentions
  `radiative` as a future top-level flag; here is the actual shape
  needed.

- [ ] **Per-field boundary conditions + BC ↔ driver linkage.**
  `BoundaryConditions` (`_models.py:252`) carries one tag per face. PIC
  needs *different* BCs for E (PML), B (PML), and particles (reflecting/
  absorbing/thermal-bath) at the same face. Solar-wind-driven
  magnetosphere runs need an inflow E-profile that an enum tag can't
  describe. Drivers exist (`[[drivers]]` with `coupling="boundary"`,
  `_models.py:444`) but there's no link from a BC face to the driver
  that supplies its values.
  Sketch: `BoundaryConditions` becomes a list of face-keyed entries
  with `field: "E" | "B" | "particles"` and optional
  `driver: "<name>"` foreign key. Restructures the existing type —
  v1.1.

- [ ] **Multi-cadence / region-of-interest output.**
  Each `[output.fields]`-style sub-table carries a single
  `step_interval`. Real runs write moments at 10× the cadence of full
  distributions, or ROI slabs at 10× the cadence of the global volume.
  Sketch: allow `[[output.fields]]` (repeatable) keyed by `name`, each
  with optional `region: BoxSelection | PlaneSelection`,
  `quantities`, `step_interval`. Singleton → repeatable is shape-
  changing — v1.1.

- [ ] **Anisotropic `gamma_eos` tuple form `(γ_par, γ_perp)`.**
  Two-fluid + 10-moment hybrids (Gkeyll, Hakim) need
  `Species.gamma_eos = (γ_par, γ_perp)` instead of the v1.0 scalar.
  Type-changing on an existing field — must wait for v1.1. The
  enum-only additions (CGL / 10-moment / 14-moment as `Closure`
  values) land in v1.0.x.

- [ ] **Restart `from_` type widening.**
  `Restart.from_` (`_models.py:492`) widens from `str` to `str | list[str]`
  to express VPIC-style multi-file restart manifests (one chk per rank).
  Type-changing on an existing field; v1.0.x adds the orthogonal
  `restore` / `mode` fields additively.

- [ ] **High-dimensional grids (>3D).**
  `Grid.dimensions` is `AxisInt` (1..3 entries). Gyrokinetic codes
  (GENE, GS2, GX, Gkeyll-GK) run on 5D grids; continuum-Vlasov
  (Vlasiator, Gkeyll Vlasov-Maxwell) is intrinsically 6D phase space.
  The v1.1 `[velocity_mesh]` entry separates spatial from velocity
  grids cleanly for Vlasiator, but the gyrokinetic case (3 spatial +
  2 velocity, mixed) doesn't decompose the same way. Decision needed:
  lift the `AxisInt` cap to 6, or treat gyrokinetic as a separate
  top-level `[phase_space]` block. Either route is shape-changing.

## Documentation backlog (no model changes)

Tasks against `docs/schema.md` rather than `_models.py`. Tracked
separately so they neither gate nor are gated by v1.1 shape work.

- [ ] **openPMD compatibility statement.**
  Investigation (May 2026) confirms openPMD and the pypic schema sit on
  *different planes*: openPMD is a **data layout + per-array
  self-description** standard (file structure, iteration encoding,
  per-record `unitDimension` / `unitSI`); pypic schema is a
  **simulation run + intent** standard (model identity, normalization
  paradigm, run provenance, frame transforms, drivers, bodies, probes).
  Genuine overlap is small and lives at the layout/semantics boundary.
  Spec versioning: openPMD standard frozen at v1.1.0 (Feb 2017), no
  2.x; the `openPMD-api` library is active (v0.17.0, Jan 2026) with the
  ED-PIC extension carrying the production PIC vocabulary.
  Documented mapping (`docs/schema.md` § 5 Extensibility) should
  enumerate:
  - **Direct overlap (kept disjoint, no model change):** `gridSpacing`
    ↔ `[grid].spacing`; `gridGlobalOffset` ↔ `[grid].lower`;
    `axisLabels` ↔ `[coordinates].axis_labels`; `geometry` (cartesian /
    thetaMode / other) ↔ `[coordinates].geometry`.
  - **Adopt as parallel form:** `unitDimension` 7-tuple — see v1.0.x
    runtime-metadata entry "`unitDimension` per-field metadata" for
    the model-change sketch (compute-on-read, write-on-output,
    round-trip preserves the attribute). Not a replacement for
    `[units]`.
  - **Adopt as precise alternative:** ED-PIC `position` array (per-
    component stagger 0.0–1.0) — see v1.0.x runtime-metadata entry.
  - **Adopt vocabulary:** ED-PIC field solver / pusher / boundary
    condition / current deposition / particle shape literals — see
    v1.0.x TOML-schema entry.
  - **Reader-absorbed (no schema entry):** ED-PIC particle record
    attributes `macroWeighted` and `weightingPower` describe whether
    per-particle quantities are stored as-is or weighted by
    macroparticle count. The Step 25b canonical `ParticleData`
    (per-particle `weight` + scalar `species_charge` /
    `species_mass`) absorbs the distinction at the reader.
    Round-trip pypic → openPMD writer would need to re-emit
    appropriate attrs; out of scope until pypic gains an openPMD
    writer.
  - **Layout we don't replicate:** iteration encoding (`fileBased` /
    `groupBased` / `variableBased`), `basePath`, `meshesPath`,
    `particlesPath`, constant record components. Our HDF5 layout in
    `docs/schema.md` § 4 stays as is.
  - **Backend-agnostic at the schema layer.** openPMD-api supports
    both HDF5 and ADIOS2 backends; the schema is unaffected. ADIOS2
    is a reader-implementation concern (Step 42), not a v1.x model
    concern.
  - **Pypic-only (openPMD doesn't cover):** `[run]` provenance,
    `[units]` normalization paradigms, `[coordinates.transforms]`,
    `[[bodies]]`, `[[drivers]]`, `[initial_conditions]`, `[[probes]]`,
    `[restart]`, `[time]` (openPMD only stores per-iteration `time`
    and `dt`, not run start/end/n_steps), MHD/hybrid/vlasov physics
    blocks.

## Tier 3 — explicitly deferred / out-of-scope until demand

- [ ] **Realized AMR state.** The per-snapshot block tree (count per
  level, parent-child indices, Morton ordering) belongs in the data
  file, not the config. Reader-side responsibility. Track only when an
  analyzer needs cross-snapshot tree continuity.

- [ ] **Lorentz-boosted reference frames.** `FrameTransform`
  (`_models.py:325`) covers rotation/translation/scale. WarpX
  boosted-frame runs need `gamma_boost` + `boost_direction`. Wait for
  the first WarpX boosted-frame user before adding a relativistic frame
  primitive.

- [ ] **SWMF-style multi-component coupling.** `[[drivers]]` covers
  external→domain coupling. It cannot express *internal* coupling
  between simulation regions (MHD shell ↔ embedded PIC patch ↔
  ring-current solver). SWMF/CCMC archives are full of these — but they
  need a separate framework-level schema, not a v1.x extension.

- [ ] **Body orbital kinematics.** `Body` (`_models.py:421`) has
  `rotation_axis`/`rotation_period` but no orbital motion. Multi-body
  systems (Jupiter + Io + Europa) can't describe relative motion. Tied
  to TASKS.md Step 40 (time-dependent frame transforms) — pick up there.

- [ ] **Embedded boundary / cut-cell geometry.** WarpX EB, AMReX cut
  cells, irregular plasma chamber walls. `Body` covers analytic shapes
  (sphere/torus/cuboid/mesh) but not voxelized cut cells. Add only when
  a reader actually needs it.

## Reader → blocker matrix

What each pending reader needs from this backlog before it can ship a
faithful description of its native data:

| Reader (TASKS.md step) | Blockers | Batch |
|---|---|---|
| Vlasiator (Step 23) — fluid moments only | `ModelType += "vlasov"` | v1.0.x |
| Vlasiator (Step 23) — lossless VDFs | `[velocity_mesh]` | v1.1 |
| VPIC (Step 35) — basic | per-rank file layout; restart `restore`/`mode` | v1.0.x |
| VPIC (Step 35) — full per-rank restart | restart `from_` widening | v1.1 |
| ARMS (Step 36) | stretched grids (spherical-r) | v1.1 |
| WarpX, PIConGPU, Smilei (openPMD reader, Phase 1) | ED-PIC vocabulary; per-component stagger; `unitDimension`; openPMD docs mapping | v1.0.x; Docs |
| FBPIC (openPMD reader, Phase 2) | above + `thetaMode` geometry | v1.0.x |
| Smilei, EPOCH, OSIRIS-collisional | `[[collisions]]`; `[[ionization_chains]]`; QED | v1.1 |
| Gkeyll, Hakim two-fluid 10-moment — vocabulary | anisotropic `Closure` enum values | v1.0.x |
| Gkeyll, Hakim — full anisotropic `gamma_eos` | `gamma_eos` tuple form | v1.1 |
| RAMSES, MPI-AMRVAC | AMR-kind discriminator (octree) | v1.0.x |
| GENE, GS2, GX, Gkeyll-GK | `ModelType += "gyrokinetic"`; high-D grids | v1.0.x; v1.1 |
| All production PIC | current/charge smoothing on `PICSolver` | v1.0.x |

Cross-cutting items with no concrete reader on the hook: per-field
BCs, multi-cadence output, tracer-particle flag, ensemble metadata.

**v1.0 contract is intact** for the iPIC3D / BATSRUS / OpenGGCM /
SimpleHDF5 readers shipping today. The v1.0.x batch above is purely
additive — v1.0 documents continue to validate against an upgraded
validator. Only the v1.1 batch breaks the contract; bundle it tied to
a reader rollout to amortize the disruption.
