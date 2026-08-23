# Examples

Every script here is self-contained: it generates its own input, runs
against the installed `pypic`, and exits non-zero if an assertion fails.
None of them need simulation output, network access, or an extra beyond
what `uv sync` installs. `tests/test_examples.py` runs them all, so a
change that breaks one fails CI.

## A progressive on-ramp

Read them in order — each adds one layer over the last.

| Script | What it introduces |
|---|---|
| [`ex1_minimal_fields.py`](ex1_minimal_fields.py) | Arrays to `FieldDataset`. No files, no config, no reader — just the container, geometry aliases, and one derived quantity. |
| [`ex2_simple_hdf5.py`](ex2_simple_hdf5.py) | Writing canonical HDF5 and reading it back with `open_simple`. Grid metadata comes from the file. |
| [`ex3_field_mapping.py`](ex3_field_mapping.py) | Non-canonical dataset names (`mag_x` → `B_1`) via `field_map`, with an explicit `GridInfo`. |
| [`ex4_toml_config.py`](ex4_toml_config.py) | A full `simulation.toml` beside the data: species, normalization, and species-dependent derived quantities. |

## Standalone

| Script | What it shows |
|---|---|
| [`custom_reader_example.py`](custom_reader_example.py) | Writing a reader for a new simulation code — the worked version of [Adding a reader](../CONTRIBUTING.md#adding-a-reader). |
| [`ex_schindler_xi.py`](ex_schindler_xi.py) | The 3D Schindler reconnection criterion on a guide-field Harris sheet, with a figure. Honors `PYPIC_EXAMPLE_OUTPUT_DIR`. |
| [`advanced_calculations.py`](advanced_calculations.py) | Cases `compute()` does not resolve automatically: multi-species sums, custom derived fields, manual tensor work. |

## Configuration reference

[`ipic3d-double-harris.toml`](ipic3d-double-harris.toml) maps a real
iPIC3D input deck onto the v1.0 schema. The annotated template with all
three scenarios (PIC, MHD, hybrid) lives at
[`pypic.simulation.toml`](../pypic.simulation.toml) in the repo root.

## Running them

```sh
uv run python examples/ex1_minimal_fields.py
uv run pytest tests/test_examples.py     # all of them, as tests
```

This directory doubles as a scratch area for local simulation data:
`.gitignore` ignores subdirectories and data files here, so you can drop
a real run in `examples/my-run/` without it showing up in `git status`.
