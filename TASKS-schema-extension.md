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

- [x] **PIC field solver vocabulary — PSATD vs PS.** `psatd` and
  `spectral-azimuthal` added to `PICFieldSolver`.

- [x] **PIC current/charge smoothing knobs.** Optional `current_smoothing` /
  `charge_smoothing` on `PICSolver`.

- [x] **Adopt openPMD ED-PIC vocabulary across PIC literals.**
  `PICFieldSolver` and `PICPusher` extended, plus new
  `ChargeCorrection`, `CurrentDeposition` and per-species
  `ParticleShape` literals.

- [x] **`ModelType` — admit Vlasov / gyrokinetic.** `ModelType` += `vlasov`,
  `gyrokinetic`; their physics knobs route through the `[physics]`
  extras namespace until typed sub-tables land.

- [x] **Tracer flag on `[[species]]`.** `Species.tracer: bool = False`; a
  finer `tracer_kind` would remain additive.

- [x] **Time-integration vocabulary expansion.** `TimeScheme` += the RK /
  SSPRK / IMEX values, new optional `Time.splitting`, and `dt > 0`
  required for every scheme except `adaptive`.

- [x] **Stochasticity / ensemble metadata on `[run]`.** Optional
  `Run.random_seed` and `Run.ensemble`, the latter validating `1 <=
  member_id <= total`.

- [x] **AMR kind discriminator.** `GridAMR.amr_kind: 'block' | 'patch' |
  'octree' = 'block'`, admitting octree codes.

- [x] **AMR temporal subcycling flag.** `GridAMR.level_subcycling: bool =
  False`; a per-level `dt_factor` array stays a v1.1 candidate.

- [x] **Ghost cell counts.** `Grid.ghost_cells`, axis count enforced against
  `grid.dimensions`.

- [x] **Per-rank / multi-file output layout.** Optional `file_pattern`,
  `files_per_step` and `partition` on every `[output.*]` via
  `_OutputBase`.

- [x] **Anisotropic closure enum values.** `Closure` += `cgl`, `10moment`,
  `14moment`.

- [x] **Restart granularity (additive fields).** `Restart.restore` and
  `Restart.mode`; `from` was widened by the round-2 `from_files` field
  rather than retyped.

- [x] **`thetaMode` / RZ azimuthal-mode geometry.** `Geometry` +=
  `thetaMode` with a required `[coordinates.modes]` sub-table; the
  reader maps it to `CYLINDRICAL`.

### Runtime-metadata additions (FieldDataset / StaggerInfo)

Not versioned by `schema_version`; land independently of the
TOML-schema additions above.

- [x] **`unitDimension` per-field metadata.** The openPMD 7-tuple derives
  from `quantity_type` (explicit at registration for custom fields) and
  round-trips through the Zarr attrs.

- [x] **Per-component stagger via `position` array.** Optional
  `StaggerInfo.position` map of per-component cell offsets, frozen,
  range-checked, and round-tripped by the Zarr serializer.

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

- [x] **`[velocity_mesh]` for continuum-Vlasov codes.** Optional top-level
  section with extent-shape and block-size validation — unblocks
  Vlasiator lossless VDFs (Step 23).

- [x] **Stretched / non-uniform grids.** Optional `[grid.stretched]`, sparse
  per-axis `axis_widths` validated against the extent — unblocks ARMS
  spherical-r (Step 36).

- [x] **`Restart.from_files` widening.** Optional `from_files` list,
  distinct and non-empty — unblocks the VPIC per-rank restart manifest
  (Step 35).

- [x] **Anisotropic `gamma_eos` on `[[species]]`.** Optional `gamma_eos_par`
  / `gamma_eos_perp`, required together and exclusive with the scalar —
  unblocks 10-moment hybrids.

- [x] **`[[collisions]]` table for collisional PIC.** Optional repeatable
  section; both `species_pair` names must resolve against `[[species]]`,
  self-collisions allowed.

- [x] **Multi-cadence / region-of-interest output.** Optional repeatable
  `[[output.streams]]` with box/plane regions, unique names, and
  per-quantity precision overrides.

- [x] **Per-field boundary conditions + BC ↔ driver linkage.**
  `BoundaryConditions.field_overrides` for E / B / particles, plus
  sparse per-face driver foreign keys resolved against `[[drivers]]`.

- [x] **`[phase_space]` for >3D kinetic codes.** Optional top-level section
  whose first `n` dimensions must match `[grid].dimensions` — unblocks
  gyrokinetic and 6D Vlasov codes.

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
  photon species + cross-section table identifiers. `docs/schema.md`
  names `radiative` as an anticipated `[physics]` flag; here is the
  actual shape needed. Defer until a QED-capable reader lands.

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
| ARMS (Step 36) | ✅ `[grid.stretched]` (spherical-r) — **schema only**; `load_config` now refuses such a deck (`UnsupportedGridError`) rather than mis-placing its cells, so the reader is still blocked on TASKS Step 51 | v1.0.x ✅ / runtime ❌ |
| PLUTO, Athena++ (no step yet) | `[units]` velocity anchor for the MHD form (cleanup.md Phase 10 item 10); `[grid.stretched]` runtime (Step 51) | pending |
| Entity SRPIC (no step yet) | `[grid.stretched]` runtime for QSpherical (Step 51) | pending |
| Entity GRPIC (no step yet) | `[coordinates].metric` block + non-orthogonal operators (Step 52) | deferred |
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
