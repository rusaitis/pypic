# Schema Validation

The Pydantic v2 models that define the `simulation.toml` v1.0 contract, and
`validate_simulation_toml()`, the entry point that returns a strictly typed
`SimulationSchema`.

This subpackage is authoritative: `readers.config.load_config()` is a thin
translator that delegates all shape validation here and then maps the result
onto the internal `SimulationConfig` / `GridInfo` / `Normalization` /
`SpeciesInfo`. It deliberately has no pypic-internal imports — only stdlib and
pydantic — so it can be lifted into a standalone distribution.

The prose specification is [Schema](../schema.md); the annotated reference
template is `pypic.simulation.toml` at the repository root. A generated JSON
Schema ships in the wheel for consumers that are not running Python:

```sh
uv run pypic schema export -o simulation.schema.v2.0.json
uv run pypic schema validate path/to/simulation.toml
uv run pypic schema diff path/to/proposed.json
```

::: pypic.schema
