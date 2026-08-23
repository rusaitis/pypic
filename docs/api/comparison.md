# Comparison

Compare the same field between two runs, across codes or across parameter
studies.

Cross-model comparison converts to SI at the comparison boundary — different
normalizations make code units incomparable — so `units` defaults to `"si"`.
Same-model comparisons with identical normalization can pass `units="code"`,
and dimensionless quantities (beta, Mach numbers, entropy) need no conversion
at all.

Datasets on different grids must be aligned first; see
[`align_grids`][pypic.regrid.align_grids] in [Regridding](regrid.md).

`nan_policy` is forwarded unchanged to the underlying diagnostics, so masked
regions from `SphereSelection`, `FieldDataset.where()`, or out-of-domain
regrid fills behave the same whether you pass arrays or datasets — see
[Conventions § NaN handling](../conventions.md#nan-handling-in-diagnostics).

::: pypic.comparison
