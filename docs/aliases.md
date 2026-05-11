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
| `Ve1`..`Ve3` | `V_s0_1`..`V_s0_3` | Electron bulk velocity |
| `Vi1`..`Vi3` | `V_s1_1`..`V_s1_3` | Ion bulk velocity |
| `EFe`, `EFi` | `EF_s0`, `EF_s1` | Energy flux (vector group) |
| `s_e`, `s_i` | `s_s0`, `s_s1` | Per-species entropy |
| `beta_e`, `beta_i` | `beta_s0`, `beta_s1` | Per-species plasma beta |
| `P_par_e`, `P_par_i` | `P_s0_par`, `P_s1_par` | Per-species parallel pressure |
| `P_perp_e`, `P_perp_i` | `P_s0_perp`, `P_s1_perp` | Per-species perpendicular pressure |
| `agyrotropy_e`, `agyrotropy_i` | `agyrotropy_s0`, `agyrotropy_s1` | Per-species agyrotropy |
| `s_gyro_e`, `s_gyro_i` | `s_gyro_s0`, `s_gyro_s1` | Per-species gyrotropic entropy |
| `q_e`, `q_i` | `q_s0`, `q_s1` | Per-species conductive heat flux (vector group) |

## Characteristic-scale literature aliases (NRL Plasma Formulary)

Derived/compute-only quantities use the Tier-3 `<field>_s<N>` form
as the canonical recipe ID. The NRL Plasma Formulary spelling
remains a registered alias for human-friendly compute calls and
docstring readability:

| Alias | Canonical | Meaning |
|-------|-----------|---------|
| `omega_pe`, `omega_pi` | `omega_p_s0`, `omega_p_s1` | Plasma frequency |
| `omega_ce`, `omega_ci` | `omega_c_s0`, `omega_c_s1` | Cyclotron frequency |
| `d_e`, `d_i` | `d_s0`, `d_s1` | Inertial / skin depth |
| `r_e`, `r_i` | `r_s0`, `r_s1` | Thermal gyroradius |
| `v_th_e`, `v_th_i` | `v_th_s0`, `v_th_s1` | Thermal speed (NRL convention) |
| `lambda_D` | `lambda_D_s0` | Electron Debye length |

Multi-species runs (`omega_p_s2`, `lambda_D_s3`, ...) synthesize via
`_SPECIES_TEMPLATES` on demand — no static recipe needed.

The dataset's alias resolver is bidirectional for these e/i ↔ `_sN`
pairs: a reader that emits `Pe` (e.g. iPIC3D) satisfies a recipe asking
for `P_s0`, and vice versa. Storage is the same data either way; the
canonical form is a documentation choice.

## Species position for derived modifiers (Stage E)

The Tier-3 rule distinguishes **generic operators** (apply to many
fields in principle) from **compound-name descriptors** (specific to
one field family). Species qualifier position follows the kind:

| Kind | Examples | Species position | Form |
|------|----------|------------------|------|
| Generic operator | `\|·\|`, `_par`, `_perp`, `_<i>`, `_<ij>` | *before* the operator | `\|V_s0\|`, `P_s0_par`, `V_s0_1` |
| Compound-name descriptor | `_gyro`, `_m`, `_c`, `_th`, `_k`, `_int`, `_trace` | *after* the compound name | `s_gyro_s0`, `rho_m_s0`, `e_th_s0` |

The discriminator: would the suffix meaningfully apply to many parent
fields (parallel projection, magnitude, vector index)? Then it's a
generic operator and species comes first. Or is it part of an atomic
name that gives the field its identity (`s_gyro` is a distinct entropy
formula from `s`, not a transformation of it)? Then it's a compound-name
descriptor and species goes at the end.

| Alias | Canonical | Notes |
|-------|-----------|-------|
| `\|V\|_s0`, `\|V\|_s1` | `\|V_s0\|`, `\|V_s1\|` | Legacy (pre-Stage-E) pipe-outside-species form |
| `P_par_s0`, `P_par_s1` | `P_s0_par`, `P_s1_par` | Legacy (pre-Stage-E) modifier-after-species form |
| `P_perp_s0`, `P_perp_s1` | `P_s0_perp`, `P_s1_perp` | Same |
| `V_s0_mag`, `V_s1_mag` | `\|V_s0\|`, `\|V_s1\|` | ASCII spelling of the pipe-bracketed canonical |

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
`"HFi"`, `"q_e"`, `"q_i"`, ... resolve to the corresponding
per-species prefix (`EF_s0`, `EF_s1`, ..., `q_s0`, `q_s1`) and
then expand to their three components. Derived quantities still
expand to their dependencies — `"Pi"` loads the six ion pressure
tensor components.

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
