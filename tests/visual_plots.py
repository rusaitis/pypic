"""Visual plot test script for manual inspection.

Generates PNG files exercising every plot type, including crowded
multi-panel layouts.  Run with::

    uv run python tests/visual_plots.py              # both themes
    uv run python tests/visual_plots.py --theme dark  # dark only

Output goes to ``tests/output/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from pypic.plotting import (  # noqa: E402
    DARK,
    LIGHT,
    PlotTheme,
    VectorLegendEntry,
    add_inset_colorbar,
    add_panel_label,
    add_status_badge,
    add_vector_legend,
    plot_comparison,
    plot_field_slice,
    plot_line,
    plot_quiver,
    plot_streamlines,
    plot_time_series,
    use_theme,
)
from pypic.readers.base import FieldDataset, GridInfo, TabularData  # noqa: E402
from pypic.units import Normalization  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "output"

GRID = GridInfo(
    dimensions=(40, 30, 20),
    spacing=(0.5, 0.5, 0.5),
    origin=(0.0, 0.0, 0.0),
)


def _make_harris_fields(
    y_center: float = 7.5,
) -> dict[str, np.ndarray]:
    """Harris-like current sheet with density enhancement at midplane."""
    x, y, z = np.meshgrid(
        np.linspace(0, 19.5, 40),
        np.linspace(0, 14.5, 30),
        np.linspace(0, 9.5, 20),
        indexing="ij",
    )

    bx = np.tanh((y - y_center) / 2.0)
    by = 0.1 * np.sin(2 * np.pi * x / 20.0)
    bz = 0.05 * np.cos(2 * np.pi * z / 10.0)

    ex = 0.01 * np.sin(np.pi * y / 15.0)
    ey = -0.02 * np.cos(np.pi * x / 20.0)
    ez = 0.005 * np.ones_like(x)

    vx = 0.1 * np.tanh((y - y_center) / 3.0)
    vy = 0.05 * np.sin(2 * np.pi * x / 20.0)
    vz = 0.02 * np.cos(np.pi * z / 10.0)

    rho_m = 1.0 + 0.5 / np.cosh((y - y_center) / 2.0) ** 2
    pressure = 0.5 + 0.3 / np.cosh((y - y_center) / 2.0) ** 2

    return {
        "B1": bx,
        "B2": by,
        "B3": bz,
        "E1": ex,
        "E2": ey,
        "E3": ez,
        "V1": vx,
        "V2": vy,
        "V3": vz,
        "rho_m": rho_m,
        "P": pressure,
    }


def _make_dataset(y_center: float = 7.5) -> FieldDataset:
    return FieldDataset.from_arrays(
        _make_harris_fields(y_center),
        GRID,
        Normalization.identity(),
    )


def _make_tabular() -> TabularData:
    rng = np.random.default_rng(42)
    cycles = np.arange(200, dtype=np.float64)
    total = 10.0 * np.exp(-cycles / 80.0) + 0.05 * rng.standard_normal(200)
    kinetic = 4.0 * np.exp(-cycles / 60.0) + 0.03 * rng.standard_normal(200)
    magnetic = 5.0 * np.exp(-cycles / 100.0) + 0.02 * rng.standard_normal(200)
    return TabularData(
        name="diagnostics",
        columns={
            "cycle": cycles,
            "total_energy": total,
            "kinetic_energy": kinetic,
            "magnetic_energy": magnetic,
        },
        index_column="cycle",
    )


def _save(fig: plt.Figure, name: str, theme_name: str) -> None:
    path = OUTPUT_DIR / f"{name}_{theme_name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  {path.name}")


def generate(theme: PlotTheme) -> None:
    name = theme.name
    print(f"Generating {name} theme plots...")

    ds_a = _make_dataset(y_center=7.5)
    ds_b = _make_dataset(y_center=8.0)
    tabular = _make_tabular()

    # 1. Single slice — stored field + badge
    fig, ax = plot_field_slice(ds_a, "B1", theme=theme, step=100, time=5.0)
    add_status_badge(ax, step=100, time=5.0)
    _save(fig, "slice_B1", name)

    # 2. Single slice — derived positive-definite field
    fig, _ = plot_field_slice(ds_a, "|B|", theme=theme, step=100)
    _save(fig, "slice_Bmag", name)

    # 3. Comparison — three-panel A|B|diff + badge on first panel
    fig, axes = plot_comparison(
        ds_a, ds_b, "B1", theme=theme, labels=("y₀=7.5", "y₀=8.0"), step=100
    )
    add_status_badge(axes["a"], step=100, loc="upper left")
    _save(fig, "comparison", name)

    # 4. Line overlay — B components along x
    fig, ax = plot_line(
        ds_a, "B1", axis="x", theme=theme, label="$B_x$", color="C0"
    )
    plot_line(ds_a, "B2", axis="x", ax=ax, theme=theme, label="$B_y$", color="C1")
    plot_line(ds_a, "B3", axis="x", ax=ax, theme=theme, label="$B_z$", color="C2")
    ax.set_title("Magnetic field components along x")
    _save(fig, "lines_overlay", name)

    # 5. Time series
    fig, _ = plot_time_series(
        tabular,
        ["total_energy", "kinetic_energy", "magnetic_energy"],
        labels=["Total", "Kinetic", "Magnetic"],
        theme=theme,
        title="Energy evolution",
        ylabel="Energy",
    )
    _save(fig, "time_series", name)

    # 6. Crowded 2x3 grid of slices + badges on each panel
    slice_fields = ["B1", "|B|", "beta", "v_A", "e_B", "rho_m"]
    with use_theme(theme):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for i, (ax, field) in enumerate(zip(axes.flat, slice_fields, strict=True)):
            plot_field_slice(ds_a, field, ax=ax, theme=theme)
            add_status_badge(ax, step=100 + i * 10, fontsize=7)
        fig.suptitle("Field overview", fontsize=13)
        fig.tight_layout()
    _save(fig, "grid_slices", name)

    # 7. Crowded 2x3 grid of line profiles
    with use_theme(theme):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, field in zip(axes.flat, slice_fields):
            plot_line(ds_a, field, axis="x", ax=ax, theme=theme)
        fig.suptitle("Line profiles along x", fontsize=13)
        fig.tight_layout()
    _save(fig, "grid_lines", name)

    # 8. Badge showcase — 2x3 grid: dark/light mode, custom labels, colors
    with use_theme(theme):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax in axes.flat:
            plot_field_slice(ds_a, "B1", ax=ax, theme=theme)

        # Dark mode (default)
        add_status_badge(axes[0, 0], step=42, loc="upper left")
        # Light mode
        add_status_badge(axes[0, 1], time=3.14, dark_mode=False, loc="upper right")
        # Both step+time
        add_status_badge(axes[0, 2], step=100, time=5.0, loc="lower left")
        # Custom label
        add_status_badge(axes[1, 0], step=100, label="cycle", loc="upper left")
        # Custom colors override
        add_status_badge(
            axes[1, 1], step=42, label="",
            bg_color="#1a5276", text_color="gold", bg_alpha=0.75,
            loc="upper right",
        )
        # Time with custom label
        add_status_badge(
            axes[1, 2], time=0.005, time_units="ns", label="time",
            loc="lower right",
        )
        fig.suptitle("Badge placement showcase", fontsize=13)
        fig.tight_layout()
    _save(fig, "badge_showcase", name)

    # 9. Badge progress — 1x3 row showing early/mid/late + show_max toggle
    with use_theme(theme):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        step_range = (0, 500)
        for ax, s in zip(axes, [20, 250, 480], strict=True):
            plot_field_slice(ds_a, "|B|", ax=ax, theme=theme)
            add_status_badge(
                ax,
                step=s,
                step_range=step_range,
                show_max=s != 250,  # middle panel: no max
                loc="upper right",
            )
        fig.suptitle("Progress bar showcase", fontsize=13)
        fig.tight_layout()
    _save(fig, "badge_progress", name)

    # 10. Streamlines — B field on midplane
    fig, _ = plot_streamlines(ds_a, "B", theme=theme, step=100)
    _save(fig, "streamlines_B", name)

    # 11. Streamlines — colored by |B| (3D magnitude)
    fig, _ = plot_streamlines(
        ds_a, "B", color_field="|B|", theme=theme, step=100
    )
    _save(fig, "streamlines_B_color_Bmag", name)

    # 12. Quiver — V field on midplane
    fig, _ = plot_quiver(ds_a, "V", stride=2, theme=theme, step=100)
    _save(fig, "quiver_V", name)

    # 13. Quiver — B field with custom stride
    fig, _ = plot_quiver(ds_a, "B", stride=(3, 2), theme=theme, step=100)
    _save(fig, "quiver_B", name)

    # 14. Side-by-side streamlines + quiver
    with use_theme(theme):
        fig, (ax_stream, ax_quiv) = plt.subplots(1, 2, figsize=(14, 5))
        plot_streamlines(ds_a, "B", ax=ax_stream, theme=theme)
        ax_stream.set_title("Streamlines")
        plot_quiver(ds_a, "B", stride=3, ax=ax_quiv, theme=theme)
        ax_quiv.set_title("Quiver")
        fig.suptitle("Vector field visualization", fontsize=13)
        fig.tight_layout()
    _save(fig, "vectors_side_by_side", name)

    # 15. Streamlines overlay — black lines on scalar field
    fig, ax = plot_field_slice(ds_a, "rho_m", theme=theme, step=100)
    plot_streamlines(
        ds_a, "B", ax=ax, color="black", linewidth=0.8, alpha=0.6,
        colorbar=False, theme=theme,
    )
    _save(fig, "streamlines_overlay", name)

    # 16. Quiver overlay — white arrows on scalar field
    fig, ax = plot_field_slice(ds_a, "|B|", theme=theme, step=100)
    plot_quiver(
        ds_a, "V", ax=ax, color="white", alpha=0.7, stride=3,
        colorbar=False, theme=theme,
    )
    _save(fig, "quiver_overlay", name)

    # 17. Inset colorbar — single slice
    fig, ax = plot_field_slice(ds_a, "B1", theme=theme, colorbar="inset", step=100)
    _save(fig, "inset_colorbar_slice", name)

    # 18. Inset colorbar + badge on same plot
    fig, ax = plot_field_slice(ds_a, "|B|", theme=theme, colorbar="inset", step=100)
    add_status_badge(ax, step=100, time=5.0, loc="upper right")
    _save(fig, "inset_colorbar_with_badge", name)

    # 19. Dark-mode inset colorbar
    fig, ax = plot_field_slice(ds_a, "B1", theme=theme, colorbar="inset", step=100)
    # Manually add a dark-mode inset colorbar on a second slice
    _save(fig, "inset_colorbar_darkmode_auto", name)

    # 20. Side colorbar vs inset colorbar comparison
    with use_theme(theme):
        fig, (ax_side, ax_inset) = plt.subplots(1, 2, figsize=(12, 5))
        plot_field_slice(ds_a, "B1", ax=ax_side, theme=theme, colorbar=True)
        ax_side.set_title("Side colorbar")
        plot_field_slice(ds_a, "B1", ax=ax_inset, theme=theme, colorbar="inset")
        ax_inset.set_title("Inset colorbar")
        fig.suptitle("Colorbar comparison", fontsize=13)
        fig.tight_layout()
    _save(fig, "colorbar_side_vs_inset", name)

    # 21. Inset colorbar with dark_mode on manually placed colorbar
    fig, ax = plot_field_slice(ds_a, "rho_m", theme=theme, colorbar=False, step=100)
    mesh = ax.get_children()[0]
    add_inset_colorbar(
        ax, mesh, r"$\rho_m$", dark_mode=True, loc="lower left", fontsize=8,
    )
    _save(fig, "inset_colorbar_dark_manual", name)

    # 22. Streamlines overlay with auto-legend
    fig, ax = plot_field_slice(ds_a, "rho_m", theme=theme, step=100)
    plot_streamlines(
        ds_a, "B", ax=ax, color="black", linewidth=0.8, alpha=0.6,
        colorbar=False, theme=theme,
    )
    _save(fig, "streamlines_auto_legend", name)

    # 23. Quiver overlay with auto-legend
    fig, ax = plot_field_slice(ds_a, "|B|", theme=theme, step=100)
    plot_quiver(
        ds_a, "V", ax=ax, color="white", alpha=0.7, stride=3,
        colorbar=False, theme=theme,
    )
    _save(fig, "quiver_auto_legend", name)

    # 24. Multi-entry vector legend (B + V overlaid)
    fig, ax = plot_field_slice(ds_a, "rho_m", theme=theme, step=100)
    plot_streamlines(
        ds_a, "B", ax=ax, color="black", linewidth=0.8, alpha=0.6,
        colorbar=False, legend=False, theme=theme,
    )
    plot_quiver(
        ds_a, "V", ax=ax, color="red", alpha=0.7, stride=3,
        colorbar=False, legend=False, theme=theme,
    )
    add_vector_legend(ax, [
        VectorLegendEntry(label="B field", color="black", linewidth=0.8, alpha=0.6),
        VectorLegendEntry(label="V flow", color="red", alpha=0.7),
    ])
    _save(fig, "multi_entry_vector_legend", name)

    # 25. Panel label grid (2x3 with a-f)
    slice_labels = ["B1", "|B|", "beta", "v_A", "e_B", "rho_m"]
    with use_theme(theme):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for i, (ax, fld) in enumerate(zip(axes.flat, slice_labels, strict=True)):
            plot_field_slice(ds_a, fld, ax=ax, theme=theme)
            add_panel_label(ax, chr(ord("a") + i))
        fig.suptitle("Panel labels a-f", fontsize=13)
        fig.tight_layout()
    _save(fig, "panel_labels_grid", name)

    # 26. Combined: panel labels + badges + vector legend
    with use_theme(theme):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plot_field_slice(ds_a, "rho_m", ax=ax1, theme=theme)
        add_panel_label(ax1, "a")
        add_status_badge(ax1, step=100, time=5.0, loc="upper right")
        plot_streamlines(
            ds_a, "B", ax=ax1, color="black", linewidth=0.8, alpha=0.5,
            colorbar=False, legend=False, theme=theme,
        )
        add_vector_legend(ax1, VectorLegendEntry(label="B", color="black"),
                          loc="lower left")

        plot_field_slice(ds_b, "rho_m", ax=ax2, theme=theme)
        add_panel_label(ax2, "b")
        add_status_badge(ax2, step=100, time=5.0, loc="upper right")
        fig.suptitle("Combined overlays", fontsize=13)
        fig.tight_layout()
    _save(fig, "combined_overlays", name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visual test plots for manual inspection."
    )
    parser.add_argument(
        "--theme",
        choices=["light", "dark", "both"],
        default="both",
        help="Which theme(s) to generate (default: both)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    themes: list[PlotTheme] = []
    if args.theme in ("light", "both"):
        themes.append(LIGHT)
    if args.theme in ("dark", "both"):
        themes.append(DARK)

    for theme in themes:
        generate(theme)

    count = len(list(OUTPUT_DIR.glob("*.png")))
    print(f"\nDone. {count} PNGs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
