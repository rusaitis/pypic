# Command Line

The `pypic` command ships with the `cli` extra:

```sh
uv add "pypic-plasma[cli]"
```

The command itself is always installed; without the extra it prints an
install hint rather than a traceback. Every subcommand takes `--help`, and
the global flags `--log-level` / `--quiet` / `--debug` apply throughout.

## Inspection

| Command | Purpose |
|---|---|
| `pypic info PATH` | Simulation metadata: code, grid, species, normalization |
| `pypic fields PATH` | Canonical field names available at a timestep |
| `pypic stats PATH` | Per-field min, max, mean, rms, NaN count |
| `pypic validate PATH` | Health check: NaN census, ∇·B, field energy |
| `pypic compare A B` | Compare fields between two simulations |

```sh
pypic info path/to/run
pypic fields path/to/run --step 0
pypic stats path/to/run --field B_1 --field beta
pypic compare run_a run_b --field B_1 --metric l2
```

## Figures

| Command | Purpose |
|---|---|
| `pypic plot PATH` | A 2D field slice |
| `pypic plot-compare A B` | Three-panel A \| B \| difference |

Both accept `--theme`, `--units`, and `--vmin` / `--vmax`. Overlaid contours
(`--contour`, `--contour-levels`), animation (`--animate`, `--fps`) and
parallel batching (`--jobs`) are `pypic plot` only. Requires the `plot` extra.

```sh
pypic plot path/to/run --field "|B|" --plane xy --theme dark --output slice.png
pypic plot-compare run_a run_b --field beta --plane xz --output compare.png
```

## Conversion and reduction

| Command | Purpose |
|---|---|
| `pypic convert fields` | Fields to a Zarr v3 store |
| `pypic convert particles` | Particles to a partitioned Parquet dataset |
| `pypic convert all` | Both, with defaults |
| `pypic reduce apply` | Collapse axes: column densities, slab means, peak maps |

Single-step input writes through `to_zarr`; multi-step promotes `time` to a
leading dimension via `to_zarr_timeseries`. Requires the `zarr` or `arrow`
extra depending on the target. See [Modern I/O](io.md).

Both take a simulation directory or HDF5 file as input, not an already
converted store.

```sh
pypic convert fields path/to/run --output run.zarr
pypic reduce apply path/to/run --axis z --reduction integrate \
    --output column.zarr
```

## Schema and codegen

| Command | Purpose |
|---|---|
| `pypic schema export` | Emit the `simulation.toml` schema as JSON Schema 2020-12 |
| `pypic schema validate` | Check `simulation.toml` files against v1.0 |
| `pypic schema diff` | Diff two JSON Schema documents |
| `pypic export bundle` | Schema, aliases, recipes, and field metadata as one JSON bundle |
| `pypic export aliases` | Compute aliases, group aliases, and the species regex |
| `pypic export recipes` | Recipe registry and per-species templates (metadata only) |
| `pypic export fields` | Static field metadata (units, LaTeX, long names) |

`schema export` regenerates the bundled artifact; CI fails if the committed
copy differs. `pypic export` is a command group — invoked bare it prints help.
Its subcommands feed the cross-language codegen described in
[Codegen](codegen.md).

```sh
pypic schema validate path/to/simulation.toml
pypic schema export -o src/pypic/schema/simulation.schema.v1.0.json
```

## Server

`pypic serve ROOT` runs the JSON HTTP + Arrow IPC WebSocket server over a
directory whose subdirectories each contain a `simulation.toml`. Binds to
`127.0.0.1` by default. Requires the `server` extra — see [Server](server.md).

```sh
pypic serve path/to/runs --port 8000
```
