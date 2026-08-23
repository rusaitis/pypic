# Changelog

All notable changes to pypic are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) — with the caveat
that the public API may still change before 1.0.

The distribution is `pypic-plasma` on PyPI; the import name is `pypic`.

## [Unreleased]

### Added

- `SECURITY.md`, a pull-request template, and a Dependabot configuration
  scoped to GitHub Actions.
- API reference pages for `pypic.io`, `pypic.plotting`, `pypic.compute`,
  `pypic.comparison`, `pypic.regrid`, `pypic.exceptions`, and
  `pypic.schema` — ten public modules previously had none, including two
  headline features.
- A zero-data on-ramp in the getting-started guide: `FieldDataset.from_arrays`
  is now documented with a runnable example, so pypic can be tried without
  simulation output.
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
- `pypic.plotting.pyvista` no longer writes to stdout when saving a screenshot;
  it logs instead.
- Workflow actions moved off the deprecated Node 20 runtime.

### Fixed

- Dead pointers in the docs and in the OpenGGCM reader docstring, which
  referenced example data that is not distributed with the repository. The
  OpenGGCM quick-start now runs against the committed fixture.
- `.hypothesis/` is ignored by the root `.gitignore`, so a local `uv build` no
  longer sweeps the Hypothesis example database into the sdist. Releases built
  in CI were never affected.
- A test that races icechunk's virtual-chunk checksum.

### Removed

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
  `normalize()` / `to_si()` and display-unit conversion.
- **Selections and reductions** — `PlaneSelection`, `BoxSelection`,
  `SphereSelection`, and `pypic.reduce`.
- **Geometry-aware operators** — divergence, curl, and gradient with metric
  factors (Cartesian implemented).
- **Field-line tracing** — adaptive Dormand-Prince 5(4) tracer with PI step
  control, plus Poincaré sections.
- **Modern I/O** — Zarr v3, Icechunk versioned storage, VirtualiZarr views over
  legacy HDF5, and Parquet/Arrow for particle data.
- **Arrow IPC server** — `pypic.server`, behind the `server` extra.
- **Command line** — `pypic info`, `fields`, `stats`, `compare`, `plot`,
  `convert`, `reduce`, `serve`, and `schema validate`.

[Unreleased]: https://github.com/rusaitis/pypic/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/rusaitis/pypic/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/rusaitis/pypic/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/rusaitis/pypic/releases/tag/v0.1.0
