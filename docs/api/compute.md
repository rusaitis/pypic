# Field Registry and Compute

`FieldDataset.compute("beta")` dispatches through this registry: a `Recipe`
names the derived function, the canonical field names it consumes, and the
quantity type of what it returns. `RECIPES` is the registry itself, exposed
read-only.

`register_recipe` / `unregister_recipe` extend it at runtime, and
`SPECIES_TEMPLATES` synthesizes per-species entries (`omega_p_s2`,
`lambda_D_s3`, ...) on demand rather than enumerating every species up front.

`field_dependencies` reports what a quantity needs, which is what lets
`Simulation.read` load exactly the fields a later `compute()` call will
require. `available_quantities` lists everything computable from a given
dataset.

Recipes marked `supports_relativistic=True` receive `c` automatically when
`physics.relativistic` is set in the dataset config — see
[Equations § 8](../equations.md#8-relativistic-corrections).

::: pypic.compute
