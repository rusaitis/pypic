# Changelog

All notable changes to pypic are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) — with the caveat
that the public API may still change before 1.0.

The distribution is `pypic-plasma` on PyPI; the import name is `pypic`.

## [Unreleased]

### Added

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

### Fixed

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

### Changed

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

### Removed

- `pypic.readers.ipic3d.to_toml`: unused, untested, and it emitted a document
  the schema rejects (`n_steps = 0`).

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
