# Regridding

Interpolate a `FieldDataset` onto a different grid, and find a common grid
for two datasets that do not share one.

`regrid` targets an explicit `GridInfo`. `common_grid` derives the
intersection of two domains at a chosen resolution, and `align_grids` applies
it to both datasets in one step — the usual preparation for
[`compare_fields`][pypic.comparison.compare_fields].

Cells that fall outside the source domain are filled with NaN by design, which
is why the comparison diagnostics default to `nan_policy="omit"`.

Cartesian is implemented. Spherical and cylindrical regridding need
metric-factor-aware interpolation (the $\sin\theta$ Jacobian matters near the
poles) and are not yet available; `GeometryUnsupportedError` is raised for
those geometries.

::: pypic.regrid
