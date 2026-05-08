# Field Name Aliases (pypic library)

This page documents aliases that pypic's `FieldDataset` resolves at
read time. Aliases are a **library-side ergonomic convenience** —
they are not part of the cross-tool schema contract documented in
[Schema](schema.md). A non-pypic consumer (Rust simulation code, JS
WebGPU viewer, generic Zarr/HDF5 reader) only needs to handle the
canonical names listed in [Schema § 3](schema.md#3-canonical-field-names).

For the canonical naming hierarchy (numbered canonical, geometry-specific
Cartesian aliases, species-name aliases auto-generated from
`[[species]].name`), see [Schema § 3](schema.md#3-canonical-field-names).
Everything below is in addition to that.

## Electron/ion convenience aliases

For the common two-species PIC case (species 0 = electrons, 1 = ions),
`e`/`i` suffixed names alias the canonical `_s0`/`_s1` forms:

| Alias | Canonical | Meaning |
|-------|-----------|---------|
| `n_e`, `n_i` | `n_s0`, `n_s1` | Number density |
| `Pe`, `Pi` | `P_s0`, `P_s1` | Scalar pressure (or Tr(tensor)/3) |
| `Te`, `Ti` | `T_s0`, `T_s1` | Temperature |
| `Ve1`..`Ve3` | `V1_s0`..`V3_s0` | Electron bulk velocity |
| `Vi1`..`Vi3` | `V1_s1`..`V3_s1` | Ion bulk velocity |
| `EFe`, `EFi` | `EF_s0`, `EF_s1` | Energy flux (vector group) |
| `s_e`, `s_i` | `s_s0`, `s_s1` | Per-species entropy |
| `beta_e`, `beta_i` | `beta_s0`, `beta_s1` | Per-species plasma beta |
| `P_par_e`, `P_par_i` | `P_par_s0`, `P_par_s1` | Per-species parallel pressure |
| `P_perp_e`, `P_perp_i` | `P_perp_s0`, `P_perp_s1` | Per-species perpendicular pressure |
| `agyrotropy_e`, `agyrotropy_i` | `agyrotropy_s0`, `agyrotropy_s1` | Per-species agyrotropy |
| `s_gyro_e`, `s_gyro_i` | `s_gyro_s0`, `s_gyro_s1` | Per-species gyrotropic entropy |

The dataset's alias resolver is bidirectional for these e/i ↔ `_sN`
pairs: a reader that emits `Pe` (e.g. iPIC3D) satisfies a recipe asking
for `P_s0`, and vice versa. Storage is the same data either way; the
canonical form is a documentation choice.

## Two-species assumption

The e/i convenience aliases hardcode species 0 = electrons, 1 = ions
(standard for two-species PIC simulations). For multi-species runs
(H⁺ + He²⁺ + O⁺, dust + plasma, multi-fluid hybrids), the e/i forms
are unsafe — use the explicit `_sN` form or the species-name alias
(`n_protons`, `T_oxygen`) documented in [Schema § 3](schema.md#3-canonical-field-names).

For total pressure across N species, compute explicitly:

```python
P = sum(data.compute(f"P_s{i}") for i in range(n_species))
```

See `examples/advanced_calculations.py`.

## Vector-group expansion with e/i forms

The vector-group expansion rules in [Schema § 3](schema.md#3-canonical-field-names)
extend to the e/i convenience forms: `"EFe"`, `"EFi"`, `"KEFe"`,
`"HFi"`, ... resolve to the corresponding per-species prefix
(`EF_s0`, `EF_s1`, ...) and then expand to their three components.
Derived quantities still expand to their dependencies — `"Pi"` loads
the six ion pressure tensor components.

## Why these aren't in the schema contract

- **Hardcoded ordering.** The e/i aliases assume species 0 = electrons,
  1 = ions. Any 3+ species run breaks this assumption silently.
- **No information gain.** `Pe` carries no information that isn't in
  `P_s0` plus the user's knowledge that `species[0].name == "electrons"`.
- **Read-time only.** The aliases never appear on disk. The Zarr writer
  emits canonical `_sN` names; the JS/Rust consumer never sees `Pe`.

Cross-tool consumers don't need to know about e/i shortcuts. They are
documented here so pypic users can find them, not in `schema.md` so
non-pypic implementations can ignore them.
