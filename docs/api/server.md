# Server

`pypic.server` is the Arrow IPC + JSON HTTP bridge between pypic and
webpic (the Three.js/WebGPU viewer). It exposes JSON discovery routes
over HTTP and one WebSocket endpoint that streams
[`FieldDataset`][pypic.dataset.FieldDataset] slices as Arrow IPC bytes.

Install the optional dependencies:

```sh
pip install 'pypic-plasma[server]'
```

Run it:

```sh
pypic serve /data/runs --host 127.0.0.1 --port 8000
```

Where `/data/runs` is a directory whose subdirectories each contain a
`simulation.toml` file. Each subdirectory becomes one addressable
simulation; the directory name is the simulation's URL identifier.

## Why Arrow IPC + WebSocket

The transport choice is deliberate:

| Choice | Why |
|---|---|
| **Arrow IPC** | Browser-native reader (`tableFromIPC` in the `apache-arrow` npm package), Rust reader (`arrow-ipc`), Python reader (`pa.ipc.open_stream`). No additional schema layer needed — Arrow's own schema metadata carries everything cross-language. |
| **WebSocket** | Persistent bidirectional connection (no HTTP-per-request overhead), text + binary frame multiplexing on one wire. |
| **Not Arrow Flight** | No Flight JS client exists for browsers; gRPC-Web would require an Envoy proxy and lose Flight's advantages. |

## HTTP routes

All return JSON.

| Route | Returns |
|---|---|
| `GET /health` | `{"status": "ok", "pypic_version": "..."}` |
| `GET /sims` | `{"sims": [<name>, ...]}` — names of subdirectories of `root` that contain `simulation.toml`. |
| `GET /sims/{sim}` | Identity + grid + normalization + species for one simulation. JSON dump of typed metadata; client can reconstruct `SimulationConfig`-equivalent state. |
| `GET /sims/{sim}/steps` | `{"steps": [0, 100, 200, ...]}` — available timestep indices. |
| `GET /sims/{sim}/fields?step=N` | `{"step": N, "fields": {canonical: native_name_or_null, ...}}` — `step` defaults to the first available step when omitted. |

FastAPI auto-generates an OpenAPI schema at `/openapi.json` and an
interactive browser at `/docs` — useful for webpic developers
exploring the surface. (WebSocket routes are out of the OpenAPI 3.0
spec by design; this page is the wire-protocol reference for them.)

## WebSocket: `WS /sims/{sim}/stream`

One persistent connection per webpic tab. Each `subscribe` frame is
one request/response exchange; the connection stays open across
exchanges and survives errors.

### Request frame (JSON text)

```json
{
  "type": "subscribe",
  "request_id": "uuid-or-any-client-string",
  "step": 0,
  "fields": ["B_1", "B_2", "B_3"],
  "selection": {"kind": "box", "ranges": {"x": [10, 30]}},
  "reduction": {"axis": "z", "op": "mean", "weight": null},
  "units": "code"
}
```

| Key | Type | Meaning |
|---|---|---|
| `type` | `"subscribe"` | Required tag. Other frame types reserved for follow-ups. |
| `request_id` | string | Echoed back in the ack / error so the client can match responses to requests. |
| `step` | int | Timestep to read. Must be in `GET /sims/{sim}/steps`. |
| `fields` | list of str (optional) | Canonical or alias field names. Empty / omitted → every available field is encoded. |
| `selection` | object (optional) | One of the selection specs below. Applied before reduction. |
| `reduction` | object (optional) | Reduction spec; see below. |
| `units` | `"code"` or `"si"` | `"code"` (default) keeps stored values; `"si"` calls `FieldDataset.in_si` per field. |

#### Selection specs (tagged union on `kind`)

```json
{"kind": "box", "ranges": {"x": [10, 30], "y": [10, 30]}}
{"kind": "plane", "normal": "z", "index": 5}
{"kind": "sphere", "center": [0, 0, 0], "radius": 5.0, "keep": "inside"}
```

Maps to [`BoxSelection`][pypic.selections.BoxSelection],
[`PlaneSelection`][pypic.selections.PlaneSelection], and
[`SphereSelection`][pypic.selections.SphereSelection] respectively.
`plane.index = null` (or omitted) selects the midplane;
`sphere.keep` defaults to `"inside"`.

#### Reduction spec

```json
{
  "axis": "z",
  "op": "integrate",
  "weight": "rho_c",
  "nan_policy": "omit"
}
```

| Key | Type | Default | Meaning |
|---|---|---|---|
| `axis` | str or list of str | required | Single axis name or list (e.g. `["y", "z"]`). |
| `op` | str | `"integrate"` | One of `integrate`, `sum`, `mean`, `median`, `max`, `min`, `std`, `var`, `argmax`, `argmin` (the `Reduction` Literal). |
| `weight` | str or null | `null` | Weight field for `mean` / `integrate` only — see [Reductions](reductions.md). |
| `nan_policy` | `"omit"`, `"propagate"`, `"raise"` | `"omit"` | NaN handling. |

### Response frames

For each request, the server emits **two frames**:

1. **JSON text frame** — `Ack` announcing the binary payload:

```json
{
  "type": "ack",
  "request_id": "uuid-or-any-client-string",
  "shape": [40, 20],
  "dims": ["x", "y"],
  "fields": ["B_1"],
  "units": "code"
}
```

2. **Binary frame** — a complete Arrow IPC stream (start + one
   `RecordBatch` + EOS marker) containing the field arrays. Each
   field is a 1-D column (row-major flattened to `prod(shape)`
   entries); reshape using `shape` from the ack. Schema metadata
   under the `"pypic"` key carries grid, normalization, species,
   coordinate arrays, and per-field attrs (quantity type, SI unit,
   reduction provenance, ...).

### Error frame (JSON text)

```json
{
  "type": "error",
  "request_id": "uuid-or-any-client-string",
  "kind": "unknown_field",
  "message": "..."
}
```

Server emits one error frame on failure and stays connected so the
client can retry. `kind` is one of:

| `kind` | Cause |
|---|---|
| `validation` | The request frame failed Pydantic validation, or pypic raised `ValueError` (bad axis, malformed selection, ...). |
| `unknown_field` | A requested field name didn't resolve. Same code is used when a requested step isn't available. |
| `unknown_sim` | The path's `{sim}` doesn't exist or its directory lacks `simulation.toml`. |
| `geometry_unsupported` | Spatial-axis reduction on a non-Cartesian grid (Jacobian-aware integration is deferred to TASKS Step 43b). |
| `internal` | Anything else. The server logs a full traceback; the client receives the exception message. |

## Schema metadata in the Arrow payload

Every binary frame carries a `b"pypic"` JSON metadata entry on the
Arrow schema:

```json
{
  "schema_version": "1.0",
  "shape": [4, 3, 2],
  "dims": ["x", "y", "z"],
  "units": "code",
  "grid": { ... },
  "normalization": { ... },
  "species": [ ... ],
  "coords": {"x": [0.5, 1.5, 2.5, 3.5], "y": [...], "z": [...]},
  "fields": {
    "B_1": {
      "quantity_type": "b_field",
      "si_unit": "T",
      "latex": "B_x",
      "long_name": "Magnetic field x-component",
      "unit_dimension": [1, 1, -2, -1, 0, 0, 0],
      "reduction": {"axis": "z", "op": "integrate", "length_axes": 1}
    }
  }
}
```

`schema_version` matches the `[schema].version` tag from
`simulation.toml` and the on-disk Zarr layout. A client that can read
v1.0 data on disk reads v1.0 data over the wire by the same code path.

## JavaScript client (sketch)

```javascript
import { tableFromIPC } from 'apache-arrow';

const ws = new WebSocket('ws://localhost:8000/sims/run0/stream');
ws.binaryType = 'arraybuffer';

ws.onmessage = (event) => {
  if (typeof event.data === 'string') {
    const frame = JSON.parse(event.data);
    if (frame.type === 'ack') {
      pendingShape = frame.shape;
    } else if (frame.type === 'error') {
      console.error(frame.kind, frame.message);
    }
  } else {
    const table = tableFromIPC(new Uint8Array(event.data));
    const meta = JSON.parse(table.schema.metadata.get('pypic'));
    const flat = table.getChild('B_1').toArray();  // Float32Array | Float64Array
    // Reshape using meta.shape, upload to Three.js BufferAttribute, ...
  }
};

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    request_id: crypto.randomUUID(),
    step: 0,
    fields: ['B_1'],
    reduction: {axis: 'z', op: 'mean'},
  }));
};
```

## CORS

The default `pypic serve` runs with `--cors-origin "*"` so a webpic
dev build on a different port can hit the server immediately.
Production deployments must tighten this:

```sh
pypic serve /data/runs \
    --cors-origin https://webpic.example.org \
    --cors-origin https://staging.example.org
```

The wide-open default exists because foundations are for local dev;
the production-only allowlist is the responsibility of whoever
mounts the app behind a real ingress.

## What's deferred (TASKS Step 37 follow-ups)

These belong to follow-up tickets, not the foundations:

* **Authentication** — token validation in a middleware. Out of scope.
* **Particle WebSocket** — analogous endpoint for `ParticleData` (the
  [Parquet interchange](../schema.md#43-particle-layout-parquet) already
  exists; the WS bridge would mirror `field_dataset_to_arrow_ipc` for
  particles).
* **Progressive / chunked transfer** — multiple `RecordBatch` frames
  per response for very large datasets. The current single-batch
  encoding scales to a few hundred MB of double precision; beyond
  that, switch to chunking.
* **Server-side frame transforms** — accepting a `frame` field in the
  request and applying `FieldDataset.transform_to` before encoding.
  Easy to add when TASKS Step 40 (time-dependent frame transforms) lands.
* **Selection-provenance round-trip** — TASKS Step 37b ships the
  `SelectionSpec` shape defined here into stored Zarr `attrs.selections`
  so reduced datasets can replay their region definition.
* **Caching layer** — currently every request reads from disk. A
  per-step-per-field cache (with eviction) drops in cleanly because
  the encoding is a pure function of the resulting `FieldDataset`.

## API reference

::: pypic.server.app
    options:
      members:
        - create_app
        - serve

::: pypic.server.arrow
    options:
      members:
        - field_dataset_to_arrow_ipc
        - decode_field_dataset_ipc

::: pypic.server.exceptions

::: pypic.server.protocol
    options:
      members:
        - SubscribeRequest
        - Ack
        - ErrorFrame
        - SelectionSpec
        - BoxSpec
        - PlaneSpec
        - SphereSpec
        - ReductionSpec
        - to_selection
        - to_reduction_kwargs
