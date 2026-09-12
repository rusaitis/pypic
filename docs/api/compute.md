# Field Registry and Compute

`FieldDataset.compute("beta")` dispatches through this registry: a `Recipe`
names the derived function, the canonical field names it consumes, and the
quantity type of what it returns. `RECIPES` is the registry itself, exposed
read-only.

`register_recipe` / `unregister_recipe` extend it at runtime, and
`SPECIES_TEMPLATES` synthesizes per-species entries (`omega_p_s2`,
`lambda_D_s3`, ...) on demand rather than enumerating every species up front.

**Adding a quantity goes through `register_recipe`, not `Recipe`.** It takes
the name and `quantity_type` alongside the function and its inputs, and
registers both the recipe and the field metadata, so `compute()`, `in_si()`
and `field_info()` all light up together:

```python
register_recipe(
    "e_mag_fraction",
    func=lambda e_b, e_k: e_b / (e_b + e_k),
    fields=("e_B", "e_k"),          # stored fields or other derived names
    quantity_type="dimensionless",
)
```

`Recipe` is the record the registry stores. It carries neither the name nor
the quantity type — those are registry keys — and is exported for the
cross-language export in [`pypic.codegen`](codegen.md), not for registration.
For a single array on a single dataset, `FieldDataset.with_field` is the
lighter option: it stamps the metadata into that dataset's attrs and leaves
the global registry alone. `examples/advanced_calculations.py` runs both.

`field_dependencies` reports what a quantity needs, which is what lets
`Simulation.read` load exactly the fields a later `compute()` call will
require. `available_quantities()` takes no arguments and lists every registered
quantity name and alias, excluding the per-species names synthesized on
demand.

Recipes marked `supports_relativistic=True` receive `c` automatically when
`physics.relativistic` is set in the dataset config — see
[Equations § 8](../equations.md#8-relativistic-corrections).

::: pypic.compute
