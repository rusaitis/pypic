# Changelog

All notable changes to pypic are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) — with the caveat
that the public API may still change before 1.0.

The distribution is `pypic-plasma` on PyPI; the import name is `pypic`.

## [Unreleased]

### Added

- `pypic.UnsupportedGridError`: raised by `load_config` when a deck is valid
  under the cross-tool schema but declares a grid pypic's containers cannot
  represent — today, `[grid.stretched]`. A `NotImplementedError` (via
  `GeometryUnsupportedError`, whose server routing and `except` clauses it
  inherits), because "pypic does not read this" and "this document is
  invalid" are different statements and the type is how a user tells which
  one they got.
- `[units].data_in_si` — the second, orthogonal axis. `anchor` says how the
  references were closed; `data_in_si` says whether the arrays on disk are
  already SI, and asks the reader to rescale them on load. These were one
  field before 2.0, which is why SI-emitting codes had no way to name an
  anchor and so no way to be normalized correctly: a Vlasiator-shaped deck
  now returns hand-checked SI for `v_A` and `e_B` where it was previously
  wrong by $\sqrt{\mu_0}$ and $\mu_0$. Readers whose format fixes its own
  unit convention (OpenGGCM, BATSRUS, iPIC3D) convert at their own boundary
  and ignore it; it is for the generic path, where the deck is the only thing
  that knows.
- `Normalization.rationalization_ratio`: $B_{ref}^2 / (\mu_0 n_{ref} m_{ref}
  v_{ref}^2)$, the number that says which code-unit convention a reference set
  implies. `pypic.derived` computes in SI-rationalized units throughout, which
  holds exactly when the ratio is 1; other values name other conventions
  ($1/\mu_0$ for data already in SI, $(c_{SI}/c_{ref})^2$ for a reduced speed
  of light, $2/\beta$ for gyrokinetic gyro-Bohm). Reported, not enforced — the
  last two are deliberate physics.
- `[units].speed_of_light` set below $c$ now warns. It scales the velocity unit
  but not $B_{ref} = m\omega/q$, so it silently moved the ratio to
  $(c_{SI}/c_{ref})^2$ and every EM conversion with it — magnetic energy
  density on a $c/10$ deck was wrong by 100×. The warning names the two honest
  spellings: `reference_velocity` for a genuinely different velocity unit,
  `scaling_factor` / `scaling_description` for a dimensionless modelling choice.
- `pypic info --json` reports all eight storage primitives plus
  `rationalization_ratio`. It carried six, omitting `e_field_ref`,
  `mass_ref` and `charge_ref`, so a JSON consumer could neither read the
  ratio the text output prints nor recompute it — or any SI factor needing
  those three.
- `Normalization.si_factor` resolves all eight storage primitives, so
  `"mass"` and `"charge"` — and with them `SpeciesInfo.mass` — have a route to
  SI for the first time. `normalize` and `to_si` take the same eight.
- `pypic.UnitSystem` and `Normalization.system`: how a `Normalization`'s
  references were anchored (`from_species` / `explicit` / `si`), or `None`
  when no `[units]` section declared one. `Normalization.undeclared()`
  constructs that undeclared state. The value round-trips through Zarr
  stores and the Arrow wire as `attrs.normalization.system`; a store written
  before the key existed decodes as `explicit`, the honest reading of a
  reference set someone stated in full.
- `pypic.UndeclaredNormalizationError`: raised by `Normalization.si_factor`
  (and so by `in_si`, `in_units` and `field_si_factor`) when a dimensional
  quantity is converted under an undeclared normalization. A subclass of
  `ValueError`; the server maps it to `undeclared_normalization` / 400.
- `pypic.vector_component(name)` splits a registered Tier-3 vector component
  name into ``(base, component)`` — the single source `transform_to` now uses
  to decide what rotates.
- `pypic.readers.ReaderBase`: the base every built-in reader now shares.
  `available_fields`, the auxiliary-data defaults and the one `_finish` exit
  that turns arrays into a `FieldDataset` live there, so a new reader
  implements listing, timestep discovery and the read itself, nothing else.
- `FieldDataset.time`: the snapshot time, from `metadata["time"]` when the
  file records one, else `step * grid.dt`. The Zarr time-series writer and
  the CLI use it instead of recomputing `step * dt` at each site.
- `FieldDataset.from_arrays(coords=...)` keeps true coordinate arrays for
  non-uniform meshes; OpenGGCM datasets now carry per-field registry attrs
  like every other reader's.
- OpenGGCM lists fields from the ``.3df`` record headers without decoding
  any WRN2 payload, so `pypic fields` no longer reads the whole file.
- `pypic.compute_with_siblings(name, dataset)`: one call evaluates a
  vector recipe and returns every component. `FieldDataset.with_derived`
  uses it, and per-species components synthesized from the templates
  (`V_s2_perp_1`, `KEF_s3_2`, ...) now batch like registered ones.
- `ax=` on `plot_comparison` (the three panels), `plot_cross_section` (the
  2D/1D pair) and `plot_field_grid` (one axes per field), so they draw into
  caller-built figures; `vmin`/`vmax` on `plot_quiver` and `plot_scatter`;
  `units=` on `plot_kymograph`, appended to the colorbar label; `save=` on
  `plot_poincare_section`; `color=` on the pyvista `add_field_lines` and
  `add_trajectories`.

### Fixed

- `Normalization.summary()` — and so `pypic info` and
  `Simulation.describe()` — reports the rationalization ratio on the `si`
  anchor, which is the case that needed it most and the one it skipped.
  `docs/schema.md` says the ratio is shown "whenever it is not 1" and lists
  $1/\mu_0$ as the "data already in SI" convention, but the summary returned
  `"SI (identity)"` before reaching the check, so the newcomer's first
  anchor was the only one that never announced that its EM surface is off by
  a power of $\mu_0$. The line now names the ratio and points at
  `[units].data_in_si`, which is the fix rather than the symptom.
- BATSRUS datasets carry their periodic axes into `GridInfo.boundary`, so the
  periodic-stencil warning below actually reaches them. `#PERIODIC` was parsed
  out of the `.h` header into a field nothing ever read, `#OUTERBOUNDARY` was
  not parsed at all, and every BATSRUS grid therefore loaded with
  `boundary=None` — including all six committed fixtures, which declare four
  periodic faces each. The two sources are combined rather than ranked
  blindly: the run's `#PERIODIC` decides whether an axis wraps, the deck's
  `#OUTERBOUNDARY` supplies the richer tag otherwise (`outflow`, `float`,
  `inflow`, ...), and a deck claiming periodic against a header that says
  otherwise resolves to `"open"`. Faces collapse two-to-one per axis because
  `GridInfo` holds one tag per axis; an asymmetric pair becomes `"mixed"`, and
  the lossless per-face list survives in `metadata["outer_boundary"]`.
- A deliberate refusal now reaches the caller of `open_simulation` as itself,
  instead of being buried in the "All candidate readers failed"
  `ExceptionGroup`. The probe loop caught every failure alike, so a
  `UnsupportedGridError` on a `[grid.stretched]` deck was demoted to a log
  line and the CLI printed only `All candidate readers failed for <path>
  (1 sub-exception)` — a headline about reader detection, with none of the
  reason, for a deck where every candidate resolves the same
  `simulation.toml` through the same `load_config` and would fail
  identically. Any `PypicError` now propagates unwrapped and stops the loop;
  untyped failures, which are reader-specific, still fall through and group
  as before. `pypic.server` gains the same correction for free: the refusal
  routes as `geometry_unsupported` / 400 by MRO where the group had landed on
  `internal` / 500.
- `GridInfo` refuses more dimensions than its geometry has axes, instead of
  truncating. A 5-tuple constructed without complaint, `surviving_axis_names`
  silently dropped everything past the third axis, and the mismatch surfaced
  two calls later as `zip() argument 2 is longer than argument 1` from inside
  `FieldDataset.from_arrays`. The refusal names the dimensionality and points
  at `[phase_space]`, which carries a gyrokinetic 5D or continuum-Vlasov 6D
  description as typed metadata even though no container holds the
  distribution function. 1D, 2D and 3D grids are unaffected.
- Complex field arrays are refused at `FieldDataset` construction instead of
  propagating into physics that assumes real. `compute("|B|")` on complex
  components returned `complex128` — $\sqrt{B_1^2+B_2^2+B_3^2}$ under complex
  arithmetic, which is not $\sqrt{|B_1|^2+|B_2|^2+|B_3|^2}$ — with nothing
  said, so a pseudo-spectral code dumping k-space, or a reader that forgot an
  inverse transform, got numbers that looked like fields. The guard sits in
  `__init__` rather than `from_arrays` so it also covers `from_zarr` and
  `open_virtual`, which construct directly; the error names every complex
  field and points at the reader boundary.
- `in_si` / `in_units` on Poynting flux were too small by a factor of $\mu_0$
  (~$1.3\times10^{-6}$). pypic's code units are SI-rationalized, so
  `derived.poynting_flux` returns a bare $\mathbf{E}\times\mathbf{B}$ and the
  SI factor owed a $1/\mu_0$ that it did not carry. A correction, not a
  migration: previously published numbers for `S_1` / `S_2` / `S_3` in SI or
  display units are wrong by that factor. Code units are unaffected, as are
  all other quantities. The test that should have caught it pinned the factor
  as the implementation restated; it now asserts $\mathbf{E}\times\mathbf{B}/\mu_0$
  against hand-computed SI inputs.
- 3D field lines coloured by a compound quantity (pressure, temperature,
  energy density, ...) with `units=` set raised `Unknown quantity`: the
  pyvista path converted through `Normalization.to_si`, which resolves only
  the six base quantities. It uses `si_factor` now, like every other
  conversion site.
- Derived quantities that divide (`temperature`, `alfven_speed`, `plasma_beta`,
  ...) crashed on integer arrays because the output was allocated with the
  input dtype; inputs now promote to float. `entropy` and `gyrotropic_entropy`
  no longer emit `RuntimeWarning` on non-positive ratios.
- `compute("div_B")` — and `div_E`, `curl_B_*`, `vort_*` — on 2D data died
  with a bare `TypeError` from the operator signature. It now raises
  `GeometryUnsupportedError` naming the quantity and the grid dimensionality.
- `v_th_s2` and higher skipped the relativistic cap that `v_th_s0` and
  `v_th_s1` applied: the species-0/1 characteristic-scale recipes are now
  generated from the same templates as every other species index, and
  `SpeciesTemplate` gained `supports_relativistic`.
- A missing dependency deep in a recipe chain was reported against the wrong
  quantity and lost its "did you mean" suggestions; the error now names both
  the requested quantity and the missing leaf.
- `Simulation.read(fields=["EFe"])` — any vector-group alias — loaded the
  three components and then raised `UnknownFieldError`, because the post-read
  check resolved compute aliases but not group aliases. One expansion funnel
  now serves both the read and the check.
- `open_simulation(path, reader="simple")` raised `TypeError`: the simple
  reader's registered factory returned a `Simulation` where every other
  factory returns `(reader, config)`.
- BATSRUS assigned its merged `simulation.toml` to a field it never read and
  OpenGGCM ignored its own, so `frame`, `normalization`, `species` and
  `physics` never reached their datasets. Both now build datasets from the
  merged config, normalizing SI-valued arrays by the declared references.
- `merge_simulation_toml` copied a hand-maintained field list that omitted
  `run`, `probes`, `collisions` and `phase_space`; it now walks
  `SimulationConfig`'s fields, so `attrs.run` provenance is written for
  reader-opened simulations.
- `FieldDataset.transform_to` rotated only eight hard-coded vector prefixes
  (`B`, `B0`, `E`, `EF`, `J`, `V`, `S`, `u`). Every derived vector stored in a
  dataset — `E_prime_*`, `E_ideal_*`, `E_Hall_*`, `curl_B_*`, `vort_*`,
  `V_perp_*`, `J_perp_*`, `KEF_s0_*`, `q_s0_*`, ... — was reoriented in space
  but kept its old-frame components. Vector triplets are now detected through
  the field registry (`pypic.fields.vector_component`), so anything registered
  as a vector rotates; an invariants test pins every registered component to
  a complete, detectable triplet.
- `plot_field_slice`, `plot_streamlines` and `plot_quiver` drew in
  matplotlib's default colormap whenever `cmap` was omitted, ignoring the
  theme's diverging and sequential maps, so signed fields got a sequential
  one. They now use the theme's, as `plot_comparison` did.
- `load_theme` raised `AttributeError` on a theme file that omits any
  optional color instead of falling back to the `PlotTheme` default.
- `plot_quiver(units=...)` labeled its colorbar in those units but colored
  the arrows by the magnitude in code units.
- The pyvista field-line and trajectory helpers: signed field lines got the
  sequential colormap under symmetric limits; a single line or trajectory
  kept pyvista's own limits and white tubes where a batch used a shared scale
  and theme colors; an unknown trajectory scalar drew white tubes instead of
  raising `KeyError`; and a trace's own `speed` scalar lost to the derived one.
- `FieldDataset`, `SimulationConfig`, `PhysicsParams`, `TabularData`,
  `ParticleData`, `FieldLine`, `ParticleTrace` and the other frozen containers
  that hold a read-only mapping raised `TypeError` under `pickle`,
  `copy.deepcopy` and `dataclasses.asdict`, so they could not cross a
  `ProcessPoolExecutor` or land in a `joblib.Memory` cache. Importing pypic
  registers a pickle reducer for `MappingProxyType`: round-trips come back
  equal and still read-only. The registration is process-wide, so any
  mappingproxy then pickles as a snapshot of its mapping.

### Changed

- `pypic.regrid` the *module* is now `pypic.regridding`, so `pypic.regrid` is
  unambiguously the function. Importing a submodule binds `pypic.<name>` to the
  module, and PEP 562 `__getattr__` runs only for names that fail to resolve —
  so with a module and a function sharing one name, `pypic.regrid` was the
  function on a clean import and the *module* after anything did
  `from pypic.regrid import align_grids`, where calling it raised
  `TypeError: 'module' object is not callable`. The module is now a noun like
  every one of its siblings, mirroring `reduce()` in `reductions.py`.
  `from pypic import regrid, align_grids, common_grid` is unchanged, which is
  the documented import path; only `from pypic.regrid import ...` moves, to
  `pypic.regridding`. A parity test in `tests/test_public_api.py` now fails if
  any exported name is shadowed by a submodule of the same name.
- `compute()` warns when a grid-dependent quantity (`div_B`, `div_E`,
  `curl_B_*`, `vort_*`) is asked for on an axis `[boundary_conditions]` marks
  `periodic`. `GridInfo.boundary` was recorded, serialized, and read by no
  numerical code, while every operator went through `np.gradient` — which has
  no periodic mode and falls back to a one-sided stencil at each end plane.
  On an analytically divergence-free periodic field over a $32^3$ box,
  interior $\max|\nabla\cdot\mathbf{B}|$ is $1.9\times10^{-15}$ against a
  full-grid $9.5\times10^{-3}$, with 18% of cells on a boundary face — so
  box-wide reductions were reporting the stencil, not the physics. The
  interior is unchanged and still second-order; the warning names the axes.
  Wrapping the stencil is TASKS Step 52.
- `load_config` refuses a `[grid.stretched]` deck instead of dropping the
  section. The widths validated, reached `SimulationSchema.grid.stretched`,
  and were then discarded — so the deck loaded with one scalar spacing per
  axis and every cell on a stretched axis at the wrong position. Measured
  with widths `[0.5, 0.6, 0.8, 1.0, 1.3]`, coordinates came out
  `[0.42 1.26 2.10 2.94 3.78]` against a truth of `[0.25 0.80 1.50 2.40 3.55]`,
  making every derivative, integral and slice along it wrong by a
  position-dependent factor, silently. Honouring the widths is TASKS Step 51;
  until then the refusal names the section and the axes. This is the one
  sanctioned divergence from the validator↔loader parity invariant, and
  `tests/test_schema_parity.py` pins it by type so it deletes itself when
  Step 51 lands.
- **`simulation.toml` is now schema 2.0, and `[units]` is the whole break.**
  `[units].system` becomes `[units].anchor`, and the four code-type-shaped
  values collapse to three anchor forms named for how the eight SI references
  are actually closed:
- `compute()` warns when a grid-dependent quantity (`div_B`, `div_E`,
  `curl_B_*`, `vort_*`) is asked for on an axis `[boundary_conditions]` marks
  `periodic`. `GridInfo.boundary` was recorded, serialized, and read by no
  numerical code, while every operator went through `np.gradient` — which has
  no periodic mode and falls back to a one-sided stencil at each end plane.
  On an analytically divergence-free periodic field over a $32^3$ box,
  interior $\max|\nabla\cdot\mathbf{B}|$ is $1.9\times10^{-15}$ against a
  full-grid $9.5\times10^{-3}$, with 18% of cells on a boundary face — so
  box-wide reductions were reporting the stencil, not the physics. The
  interior is unchanged and still second-order; the warning names the axes.
  Wrapping the stencil is TASKS Step 52.
- `load_config` refuses a `[grid.stretched]` deck instead of dropping the
  section. The widths validated, reached `SimulationSchema.grid.stretched`,
  and were then discarded — so the deck loaded with one scalar spacing per
  axis and every cell on a stretched axis at the wrong position. Measured
  with widths `[0.5, 0.6, 0.8, 1.0, 1.3]`, coordinates came out
  `[0.42 1.26 2.10 2.94 3.78]` against a truth of `[0.25 0.80 1.50 2.40 3.55]`,
  making every derivative, integral and slice along it wrong by a
  position-dependent factor, silently. Honouring the widths is TASKS Step 51;
  until then the refusal names the section and the axes. This is the one
  sanctioned divergence from the validator↔loader parity invariant, and
  `tests/test_schema_parity.py` pins it by type so it deletes itself when
  Step 51 lands.
- **`simulation.toml` is now schema 2.0, and `[units]` is the whole break.**
  `[units].system` becomes `[units].anchor`, and the four code-type-shaped
  values collapse to three anchor forms named for how the eight SI references
  are actually closed:

  | v1.0 | v2.0 |
  |---|---|
  | `system = "PIC"` | `anchor = "from_species"` — length derived from a species' plasma frequency |
  | `system = "MHD"` | `anchor = "explicit"` — length given |
  | `system = "custom"` + `[units.reference]` sub-table | `anchor = "explicit"` with flat `reference_*` keys |
  | `system = "SI"` | `anchor = "si"` |

  The `MHD` form was a strict special case of `custom` once the third
  relation $B = v\sqrt{\mu_0 n m}$ is known, so it is gone rather than
  renamed: `Normalization.mhd_standard` is reproduced bit-for-bit by an
  `explicit` deck. `explicit` takes `reference_length` plus **any two** of a
  velocity, a density and a field; the third follows. Over-supplying is legal
  and is how gyrokinetic decks state a deliberately inconsistent set.

  Three defects go with it. `reference_density` meant m⁻³ under `PIC` and
  kg/m³ under `MHD` — one key, two units — and is now
  `reference_number_density` / `reference_mass_density`. The same concept had
  two spellings (`reference_length` versus the nested `[units.reference].length`)
  and now has one. And PLUTO-class decks (`UNIT_LENGTH` / `UNIT_DENSITY` /
  `UNIT_VELOCITY`) could not validate at all — they were rejected as
  underdetermined for a $B_{ref}$ that the deck already implied.

  `[model].type` is untouched. It was never the duplicate: the two fields
  simply shared a vocabulary while meaning different things, so a hybrid deck
  read as the contradiction `type = "hybrid"`, `system = "PIC"`.

  On-disk stores bump with the validator. The store key stays `system` — it
  names the Python attribute, not the TOML key — but its *values* are the new
  vocabulary, so pre-2.0 stores are not read. Regenerate them.
- `UnitSystem` members are `FROM_SPECIES` / `EXPLICIT` / `SI`, values
  `"from_species"` / `"explicit"` / `"si"`. `Normalization.pic_standard`
  stamps `FROM_SPECIES`, `mhd_standard` and the hand-built default stamp
  `EXPLICIT`. The Python constructors themselves are unchanged.
- `Normalization.summary()` reports the rationalization ratio when it is not
  1, so `pypic info` says when a deck is not SI-rationalized.
- SI conversion factors are declared as integer exponents over the eight
  references plus an explicit power of $\mu_0$, replacing sixteen hand-written
  lambdas. Every factor pypic needs was already a monomial in those references,
  so this loses nothing and buys the invariant that was missing: composing the
  exponents with each reference's own SI dimension must reproduce the openPMD
  `unitDimension` 7-tuple registered for that quantity. The two tables encoded
  the same physics in two notations with nothing relating them, which is how
  the Poynting factor above stayed wrong. Recording $\mu_0$ as a power rather
  than a bare number is what makes the check possible. Values are unchanged to
  within 4.3e-16 relative (~2 ULP over 2000 random normalizations) — `math.prod`
  associates in reference order where the lambdas associated as written.
- **SI conversion now fails loud when no unit system was declared.** A reader
  that finds no `simulation.toml` used to hand back `Normalization.identity()`,
  which is indistinguishable from a declared `[units] system = "SI"`: `in_si`
  saw a factor of 1.0 and returned the array unchanged, so
  `in_units("B_1", "nT")` reported 0.1 code units of B as 1e8 nT. The
  reference density is not recoverable from a PIC deck — it fixes only
  dimensionless ratios — so the anchor stays a `simulation.toml` concern and
  only the silence is fixed. Affects `pypic compare`, `pypic plot-compare`,
  and `compare_fields` / `field_comparison_report` /
  `field_difference_dataset`, which all default to `units="si"`; and
  `FieldDataset.from_arrays` called without a `normalization` argument.
  Dimensionless quantities (`beta`, `M_A`, `agyrotropy`, ...) are exempt and
  keep converting. To restore the old reading, pass
  `Normalization.identity()` explicitly — it now means "these arrays already
  are SI" — or request `units="code"`.
- `pypic info` reports three unit states rather than two: declared SI,
  declared non-trivial (named by system), and undeclared. It previously
  printed `identity (SI)` for the undeclared case, which was backwards.
- `pypic.coordinates.GEOMETRY_BY_NAME` is a read-only `Mapping`
  (`MappingProxyType`), not a `dict`: the three geometries are singletons, and
  a caller that mutated the table would change what every later dataset
  resolves to. Reads are unaffected.
- Fail loud instead of warn-and-continue: duplicate `register_field` /
  `register_reader` calls, read options passed to a reader without selective
  read, an H5hut species count that disagrees with the config, and ambiguous
  BATSRUS step files now raise. The serial iPIC3D reader no longer swallows
  HDF5 errors while probing optional moments.
- The three iPIC3D readers read per-species moments through one shared path
  (`read_species_moments`); each variant supplies only a loader for its file
  layout. Every variant now treats a missing moment as "not written" the way
  H5hut always did, and a total (`rho_c`, `J_*`) whose per-species terms are
  only partly present raises instead of summing what it found.
- Dataset metadata from every reader now starts from the config's metadata,
  as the iPIC3D readers' always did; BATSRUS, OpenGGCM and the simple reader
  used to drop it. OpenGGCM's redundant ``metadata["timestep"]`` is gone
  (it duplicated ``step``).
- The simple reader's format probe is glob-only, as the registry contract
  says probes must be: any ``*.h5`` (+0.2), the default ``output_*.h5``
  pattern (+0.2) and ``simulation.toml`` (+0.3). It no longer opens a file
  to look for ``fields/`` and ``grid/`` groups.
- Built-in readers register in `pypic.readers.__init__`, in one place, rather
  than each subpackage registering itself at import time.
- Every alias table lives in one module, `pypic._aliases`, which imports
  nothing from pypic; `pypic.grid` is only `GridInfo` now. The species
  qualifier regex and the operator-suffix vocabulary (`par`, `perp`) are
  defined once and shared by `fields` and `compute`. The identity entries
  (``P_11 → P_11``) and the `Ve_s{N}_{i}` metadata pattern, which named an
  electron velocity of an arbitrary species, are gone.
- `dataset`, `comparison`, `regrid` and `reconnection` import at module scope
  what they used to import inside functions; only the calls from
  `FieldDataset` up into `compute` and `reductions` stay deferred, and
  `docs/architecture.md` now states that rule.
- The recipe tables (`_REGISTRY`, the species templates) live in
  `pypic._recipes` and the field metadata literal in `pypic._field_table`;
  `compute` and `fields` keep the lookup, execution and registration API,
  and every public name still imports from where it did. The display-unit
  vocabulary moved from `compute` to `units`.
- `pypic.cli` is a package: one module per command group, shared helpers
  and option types in `_shared` and `_options`, commands registered in one
  list in the package init. Closed-vocabulary options (`--metric`, `--scale`,
  `--format`, `--reduction`, `--nan-policy`, comparison `--units`) are
  Typer choices now: `--help` lists them and a bad value fails before the
  command runs, with Typer's own message. `convert fields` and
  `reduce apply` share one write path.
- `pypic.reduce` is split into validation, dispatch and provenance helpers;
  behaviour and messages are unchanged.
- `plot_field_slice`, `plot_comparison` and `plot_kymograph` build their
  color scale from one rule. Symlog keeps a signed field's zero-centred
  limits; log scaling starts at the smallest positive value (comparisons
  used 1e-10 whenever the data touched zero); and a comparison given only
  `vmin` or `vmax` honors it instead of discarding both.
- Every plot clips the axes it draws on to the theme's rounded corners, so
  an overlay drawn onto caller-supplied axes is clipped like the base plot;
  line, scatter, kymograph and spectrum plots used to skip it there.
- `save_theme` writes exact values (it rounded colors to three decimals),
  always writes name lists as TOML arrays and always writes `[plot]`.
  `load_theme` raises `ValueError` naming the key when a value has the
  wrong TOML type, e.g. `arrows = "false"`.
- `PlotTheme.rcparams` is read-only; derive a changed theme with
  `customize`. Themes still pickle and deep-copy.

### Removed

- `UnitsPIC`, `UnitsMHD`, `UnitsCustom` and `UnitsReferenceTable` from
  `pypic.schema`. Replaced by `UnitsFromSpecies` and `UnitsExplicit`; the
  nested `[units.reference]` sub-table no longer exists.
- `UnitSystem.PIC`, `UnitSystem.MHD` and `UnitSystem.CUSTOM`.
- `src/pypic/schema/simulation.schema.v1.0.json`, replaced by
  `simulation.schema.v2.0.json`. `get_schema_path()` defaults to the new
  version; a caller pinning `get_schema_path("1.0")` gets a missing file.

- `pypic.readers.ipic3d.to_toml`: unused, untested, and it emitted a document
  the schema rejects (`n_steps = 0`).
- `pypic.schema.cli` moved to `pypic._schema_cli`. It imported typer at module
  scope, which made `pypic.schema`'s "only stdlib and pydantic" promise false
  and would have carried a typer dependency into any standalone lift of the
  subpackage. The `pypic schema export|validate|diff` commands are unchanged.
- The `kind` and `status_code` classvars on `PypicError` and its subclasses.
  They were HTTP/WebSocket routing metadata on library types that only the
  server read; `pypic.server.exceptions.error_routing(exc)` now returns the
  `(kind, status)` pair, walking the MRO so an unmapped subclass degrades to
  its nearest mapped base instead of emitting a kind the wire format rejects.
- The `err_prev` kwarg on `i_step_controller` and
  `i_step_controller_batched`. It was reserved for a PI upgrade, accepted and
  immediately discarded; offering a kwarg that silently does nothing is worse
  than not offering it. Pass nothing — the elementary (I) controller is
  unchanged.

## [0.1.3] — 2026-08-23

### Added

- `SECURITY.md`, a pull-request template, and a Dependabot configuration
  scoped to GitHub Actions.
- API reference pages for `pypic.io`, `pypic.plotting`, `pypic.compute`,
  `pypic.comparison`, `pypic.regrid`, `pypic.exceptions`, and `pypic.schema` —
  seven public modules previously had none, including two headline features —
  plus a command-line reference covering every subcommand.
- Eleven names are now reachable straight from `pypic`: `power_spectrum_1d`,
  `power_spectrum_2d`, `power_spectrum_3d`, `schindler_xi`,
  `parallel_component`, `perpendicular_vector`, `perpendicular_magnitude`,
  `to_icechunk_virtual`, `trace_field_lines_adaptive`, `TraceDirection`, and
  `quantity_dimension`.
- A zero-data on-ramp in the getting-started guide: `FieldDataset.from_arrays`
  is now documented with a runnable example, so pypic can be tried without
  simulation output.
- A numbered example series in `examples/` — arrays to `FieldDataset`,
  canonical HDF5, non-canonical field mapping, and a full `simulation.toml` —
  with an `examples/README.md` index. Four of these existed but were hidden
  from the repository by an over-broad `.gitignore` rule and had rotted onto
  pre-`B_1` field names and the pre-v1.0 schema; `tests/test_examples.py` now
  runs every committed example and fails if one is not listed.
- `scripts/check.sh` runs the six checks CI gates, over the paths CI covers.
  CI, `CONTRIBUTING.md`, the pull-request template and `CLAUDE.md` reference it
  instead of each carrying its own path list — three of them under-claimed
  `src tests`, so touching `scripts/` or `benchmarks/` gave a green local run
  and a red pull request.
- Parity tests pinning the documented extras table to
  `[project.optional-dependencies]`, the documented CLI commands to the Typer
  app, `pypic.__all__` to the physics modules' own `__all__`, and the bundled
  data files (themes, JSON Schema) to `importlib.resources`.
- An ORCID for the author in `CITATION.cff`.

### Changed

- Documentation and repository cleanup for the public release: added an
  Ecosystem section explaining the rustpic / webpic siblings, extracted the
  architecture and style rules into `docs/architecture.md` for contributors,
  and corrected several stale references.
- The feature and extras lists in `README.md`, `docs/index.md`, and
  `docs/getting-started.md` now agree with each other and with
  `pyproject.toml`. All eight extras are documented in one table with what each
  pulls in and what it enables, and the previously undocumented Arrow IPC
  server and the `validate` / `plot-compare` / `serve` / `export` CLI commands
  are listed.
- `docs/index.md` is a landing page rather than a second copy of the README's
  feature list — the two had already drifted apart in wording.
- `README.md` links are absolute, so they resolve on PyPI where the file is the
  package long description.
- Twenty-five public names — twenty in `pypic.derived`, three in
  `pypic.diagnostics`, plus `quantity_units` and `SPECIES_TEMPLATES` — are
  re-exported from `pypic`. They were public in their own modules and rendered
  in the API reference, but reachable only as `pypic.derived.x`.
- Name resolution failures raise `UnknownFieldError` rather than a bare
  `KeyError` in `reduce`, `compute`, and `field_info`. It subclasses
  `KeyError`, so existing handlers are unaffected; the server can now return
  404 instead of 500 for a bad `fields=` over the wire.
- `FieldDataset.from_arrays` accepts any `Mapping`, not only `dict` — `dict`
  is invariant, so a `dict[str, NDArray[float64]]` was rejected.
- `pypic.readers.ipic3d` raises `UnknownFieldError` on an unrecognized
  `columns=` entry instead of silently returning fewer columns.
- The visual-inspection scripts moved from `tests/` to `scripts/visual/`.
  They collected zero tests but `--doctest-modules` imported them anyway,
  making a matplotlib backend switch a session-wide side effect and pyvista a
  hard requirement for collecting the suite.
- `pypic.plotting.pyvista` no longer writes to stdout when saving a screenshot;
  it logs instead.
- Workflow actions moved off the deprecated Node 20 runtime.
- Missing-extra errors come from one guard (`pypic._optional.require`) instead
  of eight hand-written copies, so the message names the missing modules, the
  feature, and the install command identically on the first failure and on
  every call after it. Every install hint now quotes the extra
  (`pip install "pypic-plasma[zarr]"`) — unquoted brackets are a zsh glob.
- `docs/architecture.md` is the single source for the design rules and style
  conventions; `CLAUDE.md` points at it rather than restating them in a second
  wording. The docstring rule now says what the code does: a runnable doctest
  where the function works on synthetic arrays alone, exempt where it needs a
  file, a display backend, or a server.
- `TASKS.md` and `TASKS-schema-extension.md` record shipped work as one line
  each; the design detail behind it lives in this changelog, the docs, and the
  code.

### Fixed

- The BATSRUS `.out` read path was unreachable: `available_fields()` dispatched
  to the wrong parser and a binary `.out` raised `UnicodeDecodeError`. That path
  also skipped SI conversion and computed the wrong grid spacing. OpenGGCM's
  stagger metadata was corrected at the same time.
- `pypic validate` raised `TypeError: max_div_b() missing 1 required positional
  argument: 'd3'` on any 2D dataset — the shape of most BATSRUS output. It now
  reports that ∇·B needs a 3D grid and carries on.
- Documented examples that raised when run: `plane="xy"` passed where a
  `PlaneSelection` is required, a three-value unpack of `plot_field_slice`'s
  two-tuple return, `Simulation.read(steps=...)`, and a `"|u|"` lookup for a
  quantity that is not registered.
- Docs claimed PI step control for the adaptive tracer; the shipped controller
  is an error-norm (I) controller. `register_field()` was named as the
  extension point for `compute()` dispatch, which is `register_recipe()`.
  `docs/api/exceptions.md` listed the `unknown_simulation` error kind, which is
  spelled `unknown_sim` on the wire.
- Dead pointers in the docs and in the OpenGGCM reader docstring, which
  referenced example data that is not distributed with the repository. The
  OpenGGCM quick-start now runs against the committed fixture.
- `.hypothesis/` is ignored by the root `.gitignore`, so a local `uv build` no
  longer sweeps the Hypothesis example database into the sdist. Releases built
  in CI were never affected.
- A test that races icechunk's virtual-chunk checksum.

### Removed

- `vulture_whitelist.py`, the `vulture` dev dependency, and their two
  `pyproject.toml` config blocks. Under the invocation `CONTRIBUTING.md`
  documented, the whitelist suppressed 0 of 136 findings — it was linted,
  formatted and type-checked in CI while doing nothing.
- `examples/ex5_custom_reader.py`, which imported a module that no longer
  exists and duplicated the tested `examples/custom_reader_example.py`.
- The `lazy` extra. It installed dask, but nothing under `src/` ever imported
  it — the chunked-loading path it was reserved for is still unwritten. It will
  come back when there is code behind it.

## [0.1.2] — 2026-08-22

### Fixed

- Missing `cli` extra now produces an explanation instead of a raw traceback.

## [0.1.1] — 2026-08-22

First release published to PyPI.

### Added

- Zenodo concept DOI recorded in `CITATION.cff` and the README.

### Fixed

- `pypic --version` after the distribution rename to `pypic-plasma`.

## [0.1.0] — 2026-08-22

Initial public release on GitHub and Zenodo, after 420 commits of development.
Not published to PyPI — the release workflow failed on this tag, and `0.1.1` is
the first version available from the index.

### Added

- **Multi-code readers** — iPIC3D (parallel HDF5, serial HDF5, H5hut), BATSRUS
  (IDL cell and HDF5 BATL with AMR regridding), OpenGGCM (Fortran binary 3df),
  and a generic HDF5 reader, with confidence-based auto-detection.
- **Canonical schema** — the `simulation.toml` v1.0 contract, validated by
  Pydantic v2 models in `pypic.schema` and shipped as a bundled JSON Schema.
- **Derived quantities** — magnitudes, plasma beta, Alfvén speed, Mach numbers,
  Poynting flux, energy densities, pressure-tensor decomposition,
  characteristic scales, entropy, and reconnection diagnostics, all as pure
  array-in / array-out functions.
- **Unit system** — PIC, MHD, SI, and custom normalizations with round-trip
  `Normalization.normalize()` / `.to_si()` and display-unit conversion.
- **Selections and reductions** — `PlaneSelection`, `BoxSelection`,
  `SphereSelection`, and `pypic.reduce`.
- **Geometry-aware operators** — divergence, curl, and gradient with metric
  factors (Cartesian implemented).
- **Field-line tracing** — adaptive Dormand-Prince 5(4) tracer with error-norm
  step control, plus Poincaré sections.
- **Modern I/O** — Zarr v3, Icechunk versioned storage, VirtualiZarr views over
  legacy HDF5, and Parquet/Arrow for particle data.
- **Arrow IPC server** — `pypic.server`, behind the `server` extra.
- **Command line** — `pypic info`, `fields`, `stats`, `compare`, `plot`,
  `convert`, `reduce`, `serve`, and `schema validate`.

[Unreleased]: https://github.com/rusaitis/pypic/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/rusaitis/pypic/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/rusaitis/pypic/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/rusaitis/pypic/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/rusaitis/pypic/releases/tag/v0.1.0
