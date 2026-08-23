# Plotting

Publication figures from a `FieldDataset`. Requires the `plot` extra
(matplotlib); the 3D surface in `pypic.plotting.pyvista` requires `3d`.

Beyond field slices and comparisons, this subpackage covers line plots and
time series, kymographs, quiver and streamline overlays, scatter plots,
power spectra, Poincaré sections, and a theme system with annotation
helpers (badges, planets, contours, insets, legends).

## Themes

`set_theme` / `use_theme` select a bundled theme; `available_themes` lists
them. `load_theme` and `save_theme` read and write theme TOML, and
`export_themes` emits the bundle that the webpic viewer consumes.

Themes are looked up in `$PYPIC_THEME_DIR` when that variable is set,
otherwise under `$XDG_CONFIG_HOME/pypic/themes` (falling back to
`~/.config/pypic/themes`), and finally the themes bundled with the package.

::: pypic.plotting
