# simulation.toml schema — extension backlog

Readiness review for popular MHD / PIC / hybrid codes against the v1.0
Pydantic validator (`src/pypic/schema/_models.py`). Organized as a
release plan: two **v1.0.x additive batches** that land without
invalidating any v1.0 document, a **deferred v1.1 set** of two
genuinely-niche items, and **Tier 3 deferred** items. Pointers name
models in `src/pypic/schema/_models.py`.

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

- [x] **PIC field solver vocabulary — PSATD vs PS.** Shipped: `psatd`,
  `spectral-azimuthal` added to `PICFieldSolver` (and the ED-PIC stencils
  below).

- [x] **PIC current/charge smoothing knobs.** Shipped: `current_smoothing`
  and `charge_smoothing` are optional `NonNegativeInt | None` fields on
  `PICSolver` (matching the long-standing `HybridSolver.current_smoothing`).

- [x] **Adopt openPMD ED-PIC vocabulary across PIC literals.** Shipped:
  - `PICFieldSolver` += `lehe`, `ck`, `ckc`, `pstd`, `gpstd`
    (and `psatd` / `spectral-azimuthal` from the entry above).
  - `PICPusher` += `llrk4`, `free-streaming`.
  - New `ChargeCorrection` literal exposed via the optional
    `PICSolver.charge_correction` field.
  - New `CurrentDeposition` literal exposed via the optional
    `PICSolver.current_deposition` field.
  - New `ParticleShape` literal exposed via the optional
    `Species.shape` field (per-species — WarpX and Smilei vary the
    deposition order by species).

- [x] **`ModelType` — admit Vlasov / gyrokinetic.** Shipped:
  `ModelType` += `vlasov`, `gyrokinetic`.
  `_check_physics_matches_model_type` skips the typed-branch
  cross-check for these new types — their physics knobs route through
  the `[physics]` extras namespace until typed sub-tables land in
  v1.1+.

- [x] **Tracer flag on `[[species]]`.** Shipped: `tracer: bool = False`
  on `Species`. Test-particle vs tagged-tracer semantics may need a
  follow-up `tracer_kind` field; the boolean is the additive starting
  point and any later refinement remains additive.

- [x] **Time-integration vocabulary expansion.** Shipped: `TimeScheme`
  += `rk2`, `rk3`, `rk4`, `vl2`, `ssprk2`, `ssprk3`, `imex-rk2`,
  `imex-rk3`. New optional `Time.splitting: 'strang' | 'lie' | 'godunov'
  | None` field. The `dt > 0` requirement now applies to every scheme
  except `adaptive` (the only CFL-driven mode where `dt` is a
  placeholder).

- [x] **Stochasticity / ensemble metadata on `[run]`.** Shipped:
  optional `Run.random_seed: int | None` and `Run.ensemble: Ensemble |
  None` fields. The `Ensemble` model validates `1 <= member_id <= total`.

- [x] **AMR kind discriminator.** Shipped: `GridAMR.amr_kind: 'block' |
  'patch' | 'octree' = 'block'` — preserves current implicit behavior
  while admitting RAMSES / MPI-AMRVAC octree codes.

- [x] **AMR temporal subcycling flag.** Shipped:
  `GridAMR.level_subcycling: bool = False`. A per-level `dt_factor`
  array remains a v1.1 candidate.

- [x] **Ghost cell counts.** Shipped: `Grid.ghost_cells: list[int] |
  None`, axis count enforced against `grid.dimensions` by the root
  validator.

- [x] **Per-rank / multi-file output layout.** Shipped: optional
  `file_pattern`, `files_per_step`, `partition` (`'by_rank' |
  'by_field' | 'monolithic'`) on every `[output.*]` sub-table via
  `_OutputBase`.

- [x] **Anisotropic closure enum values.** Shipped: `Closure` += `cgl`,
  `10moment`, `14moment`. Codes can declare multi-moment closures on
  `[[species]]` even before the v1.1 anisotropic `gamma_eos` tuple.

- [x] **Restart granularity (additive fields).** Shipped:
  `Restart.restore: list['fields' | 'particles' | 'auxiliary'] | None`
  (entries must be distinct and non-empty when present) and
  `Restart.mode: 'hot' | 'cold' | None`. The `from: str → str |
  list[str]` widening remains v1.1.

- [x] **`thetaMode` / RZ azimuthal-mode geometry.** Shipped:
  `Geometry` += `thetaMode`. New required `[coordinates.modes]`
  sub-table (`CoordinatesModes`) with `n_modes: PositiveInt` and
  optional `mode_indices: list[NonNegativeInt]`. The
  `Coordinates._check_modes_geometry` validator enforces
  `geometry = 'thetaMode' ⇔ modes is set`. The reader-side
  `_build_geometry` maps `thetaMode → CYLINDRICAL` (post-reconstruction
  physical grid); the openPMD reader (TASKS.md Step 42 Phase 2) will
  consume the modal metadata when it lands.

### Runtime-metadata additions (FieldDataset / StaggerInfo)

Not versioned by `schema_version`; land independently of the
TOML-schema additions above.

- [x] **`unitDimension` per-field metadata.**
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

- [x] **Per-component stagger via `position` array.** Shipped: optional
  `StaggerInfo.position: dict[str, tuple[float, ...]] | None` carrying
  per-component cell offsets in ``[0.0, 1.0)``. The constructor coerces
  list inputs to `tuple[float, ...]`, freezes the dict via
  `MappingProxyType`, and rejects out-of-range offsets. Round-trips
  through the Zarr serializer (`_serialize._stagger_to_dict` /
  `_dict_to_stagger`). The existing `convention` enum stays as the
  top-level shorthand; the `position` map is the precise form for
  Yee-mesh PIC, BATSRUS face-centered B, and the openPMD reader
  (TASKS.md Step 42).

## v1.0.x — additive batch round 2 (ship now)

Re-evaluation of the original v1.1 batch found that every item could
be expressed in a purely additive form by adding new optional fields
or sub-tables alongside the existing ones rather than retyping or
restructuring in place. This second round pulls 8 of the 10 v1.1
items forward so Vlasiator, VPIC, ARMS, and adjacent codes can
onboard without waiting on a v1.1 cut. The two genuinely-niche items
(ionization chains, QED) stay deferred — neither has a reader on the
immediate roadmap.

Recommended pairing of round-2 work to reader rollouts:

| Reader trigger | Round-2 items unlocked |
|---|---|
| Vlasiator (Step 23) — lossless VDFs | `[velocity_mesh]`, `[phase_space]` |
| ARMS spherical (Step 36) | `[grid.stretched]` |
| VPIC (Step 35) per-rank restart manifest | `Restart.from_files` |
| Gkeyll, Hakim two-fluid 10-moment (full anisotropy) | `Species.gamma_eos_par` / `gamma_eos_perp` |
| Smilei collisional / EPOCH / OSIRIS-collisional | `[[collisions]]` |
| GENE, GS2, GX, Gkeyll-GK | `[phase_space]` (5D guiding-center) |
| PIC PML on E + B; thermal-bath particles | per-field BC overrides + per-face driver foreign keys |
| Multi-cadence / ROI runs (production cross-cutting) | `[[output.streams]]` |

- [x] **`[velocity_mesh]` for continuum-Vlasov codes.** Shipped: new
  optional top-level section. `VelocityMesh` model with
  `dimensions: list[PositiveInt]` (1..3), `extent: list[list[float]]`
  (per-axis `[v_min, v_max]`), optional `block_size` (must evenly
  divide each `dimensions[i]`), optional `sparsity_threshold`,
  `coordinate_system: "cartesian" | "spherical-velocity"`. Validator
  enforces extent shape and block-size divisibility. Unblocks the
  Vlasiator reader's lossless-VDF mode (TASKS.md Step 23).

- [x] **Stretched / non-uniform grids.** Shipped: new optional
  `[grid.stretched]` sub-table. `GridStretched.axis_widths: dict[str,
  list[PositiveFloat]]`, sparse — only stretched axes appear, keyed by
  axis index as a string. The `Grid` validator enforces that listed
  cell widths sum to `upper[i] - lower[i]` (within `sum_rtol`) and
  that the list length equals `dimensions[i]`. Uniform grids
  validate unchanged; metric-aware operators are out of scope until
  Step 19b. Unblocks ARMS spherical-r (TASKS.md Step 36).

- [x] **`Restart.from_files` widening.** Shipped: new optional
  `from_files: list[str] | None` field on `Restart`. Validator:
  distinct & non-empty when present. The existing `from` field stays
  required (typically a manifest path or canonical checkpoint).
  Unblocks VPIC per-rank restart manifest (TASKS.md Step 35).

- [x] **Anisotropic `gamma_eos` on `[[species]]`.** Shipped: new
  optional `Species.gamma_eos_par`, `Species.gamma_eos_perp` fields.
  Validator: par and perp must both be present or both absent;
  cannot mix with the scalar `gamma_eos`. Unblocks Gkeyll / Hakim
  two-fluid 10-moment hybrids without retyping the existing scalar.

- [x] **`[[collisions]]` table for collisional PIC.** Shipped: new
  optional repeatable section. `Collision` model with
  `species_pair: list[str]` (length 2), `model: 'coulomb' | 'bgk' |
  'monte-carlo'`, optional `coulomb_log`, `temperature_ref`,
  `description`. Root validator enforces that both species names
  resolve against `[[species]].name`; self-collisions (same species
  twice) are permitted. Unblocks Smilei collisional, EPOCH,
  OSIRIS-collisional, PIConGPU.

- [x] **Multi-cadence / region-of-interest output.** Shipped: new
  optional repeatable `[[output.streams]]` section, alongside the
  existing singleton `[output.fields]`. `OutputStream` extends
  `_OutputBase` with required `name`, `quantities`, optional
  `region: BoxRegion | PlaneRegion` (discriminated by `kind = "box" |
  "plane"`), and `precision_overrides`. Output validator enforces
  unique stream names; the root validator cross-checks region axis
  counts against `[grid].dimensions`. Cross-cutting production-run
  benefit; works for any reader.

- [x] **Per-field boundary conditions + BC ↔ driver linkage.**
  Shipped: refactored `BoundaryConditions` into
  `BoundaryConditionsBase` (carries `lower`/`upper`/`drivers_lower`/
  `drivers_upper`) and `BoundaryConditions(BoundaryConditionsBase)`
  with optional `field_overrides: dict['E' | 'B' | 'particles',
  BoundaryConditionsBase]`. The driver foreign keys are sparse dicts
  keyed by axis index (TOML can't represent `null` inside lists).
  Root validator enforces that every driver name resolves against
  `[[drivers]].name`. Unblocks PIC PML and solar-wind-driven runs.

- [x] **`[phase_space]` for >3D kinetic codes.** Shipped: new
  optional top-level `PhaseSpace` model with
  `dimensions: list[PositiveInt]` (2..6), optional `axis_labels`,
  `extents`, and `coordinate_system: 'cartesian' | 'guiding-center'
  | 'field-aligned' | 'spherical-velocity'`. Root validator enforces
  that the first `n` dimensions match `[grid].dimensions` exactly —
  the spatial sub-grid stays owned by `[grid]`; phase space extends
  it. Unblocks gyrokinetic codes (GENE, GS2, GX, Gkeyll-GK) and full
  6D Vlasov phase space without lifting the `AxisInt` cap on
  `Grid.dimensions`.

## v1.1 — deferred items

Two genuinely-niche additions remain deferred: each only matters for
*one specific subset* of one reader family, and neither blocks any
reader currently on the TASKS.md roadmap. Both can land additively
later without disturbing v1.0.x consumers.

- [ ] **`[[ionization_chains]]` for atomic-physics PIC.**
  EPOCH, Smilei, OSIRIS support BSI / ADK / multi-photon ionization.
  Need to express that He⁺ and He²⁺ are charge states of the same atom
  (the schema currently says "treat as separate `[[species]]`", losing
  the parent linkage). Sketch: `[[ionization_chains]]` with
  `parent: "He"`, `states: ["He0", "He+", "He2+"]`, `model: "adk" |
  "bsi" | ...`. Defer until an ionization-capable reader lands.

- [ ] **QED / radiation-reaction module.**
  EPOCH-QED, Smilei-QED, OSIRIS, Zeltron support synchrotron radiation
  reaction, Breit-Wheeler pair production, Compton scattering. Needs
  photon species + cross-section table identifiers. CLAUDE.md mentions
  `radiative` as a future top-level flag; here is the actual shape
  needed. Defer until a QED-capable reader lands.

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

- [ ] **Lorentz-boosted reference frames.** The `FrameTransform`
  model covers rotation/translation/scale. WarpX
  boosted-frame runs need `gamma_boost` + `boost_direction`. Wait for
  the first WarpX boosted-frame user before adding a relativistic frame
  primitive.

- [ ] **SWMF-style multi-component coupling.** `[[drivers]]` covers
  external→domain coupling. It cannot express *internal* coupling
  between simulation regions (MHD shell ↔ embedded PIC patch ↔
  ring-current solver). SWMF/CCMC archives are full of these — but they
  need a separate framework-level schema, not a v1.x extension.

- [ ] **Body orbital kinematics.** The `Body` model has
  `rotation_axis`/`rotation_period` but no orbital motion. Multi-body
  systems (Jupiter + Io + Europa) can't describe relative motion. Tied
  to TASKS.md Step 40 (time-dependent frame transforms) — pick up there.

- [ ] **Embedded boundary / cut-cell geometry.** WarpX EB, AMReX cut
  cells, irregular plasma chamber walls. `Body` covers analytic shapes
  (sphere/torus/cuboid/mesh) but not voxelized cut cells. Add only when
  a reader actually needs it.

## Reader → blocker matrix

What each pending reader needs from this backlog before it can ship a
faithful description of its native data. Items shipped in the v1.0.x
batches (round 1 + round 2) are checked (✅). The two deferred v1.1
items are noted; neither blocks a reader currently on TASKS.md.

| Reader (TASKS.md step) | Blockers | Status |
|---|---|---|
| Vlasiator (Step 23) — fluid moments only | ✅ `ModelType += "vlasov"` | v1.0.x ✅ |
| Vlasiator (Step 23) — lossless VDFs | ✅ `[velocity_mesh]`; ✅ `[phase_space]` | v1.0.x ✅ |
| VPIC (Step 35) — basic | ✅ per-rank file layout; ✅ restart `restore`/`mode` | v1.0.x ✅ |
| VPIC (Step 35) — full per-rank restart manifest | ✅ `Restart.from_files` | v1.0.x ✅ |
| ARMS (Step 36) | ✅ `[grid.stretched]` (spherical-r) | v1.0.x ✅ |
| WarpX, PIConGPU, Smilei (openPMD reader, Phase 1) | ✅ ED-PIC vocabulary; ✅ per-component stagger; ✅ `unitDimension`; openPMD docs mapping | v1.0.x ✅; Docs |
| FBPIC (openPMD reader, Phase 2) | above + ✅ `thetaMode` geometry | v1.0.x ✅ |
| Smilei collisional, EPOCH, OSIRIS-collisional, PIConGPU | ✅ `[[collisions]]` | v1.0.x ✅ |
| EPOCH-QED, Smilei-QED, OSIRIS-QED, Zeltron | QED / radiation reaction | v1.1 (deferred) |
| EPOCH, Smilei, OSIRIS — atomic ionization | `[[ionization_chains]]` | v1.1 (deferred) |
| Gkeyll, Hakim two-fluid 10-moment — vocabulary | ✅ anisotropic `Closure` enum values | v1.0.x ✅ |
| Gkeyll, Hakim — full anisotropic `gamma_eos` | ✅ `gamma_eos_par` / `gamma_eos_perp` | v1.0.x ✅ |
| RAMSES, MPI-AMRVAC | ✅ AMR-kind discriminator (octree) | v1.0.x ✅ |
| GENE, GS2, GX, Gkeyll-GK | ✅ `ModelType += "gyrokinetic"`; ✅ `[phase_space]` | v1.0.x ✅ |
| PIC PML, solar-wind-driven runs | ✅ per-field BC overrides + driver foreign keys | v1.0.x ✅ |
| Multi-cadence / ROI output (cross-cutting) | ✅ `[[output.streams]]` | v1.0.x ✅ |
| All production PIC | ✅ current/charge smoothing on `PICSolver` | v1.0.x ✅ |

**v1.0 contract is intact** for the iPIC3D / BATSRUS / OpenGGCM /
SimpleHDF5 readers shipping today. Both v1.0.x batches (round 1 and
round 2) are purely additive — every v1.0 document continues to
validate against the upgraded validator, and `schema_version = "1.0"`
remains the canonical version string. The two deferred v1.1 items
(ionization chains, QED) will land additively later when a concrete
reader needs them.
