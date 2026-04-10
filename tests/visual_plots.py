"""Visual plot test script for manual inspection.

Generates PNG files exercising every plot type, including crowded
multi-panel layouts.  Run with::

    uv run python tests/visual_plots.py              # all themes
    uv run python tests/visual_plots.py --theme dark  # dark only
    uv run python tests/visual_plots.py --theme 4     # dark by number

Output goes to ``tests/output/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

# Allow running as `python tests/visual_plots.py` from any cwd by adding
# the project root to sys.path so `tests._helpers` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from pypic.plotting import (
    LegendEntry,
    PlotTheme,
    add_badge,
    add_contours,
    add_inset_colorbar,
    add_label,
    add_legend,
    available_themes,
    plot_comparison,
    plot_cross_section,
    plot_field_grid,
    plot_field_slice,
    plot_kymograph,
    plot_line,
    plot_line_comparison,
    plot_lines,
    plot_power_spectrum,
    plot_quiver,
    plot_scatter,
    plot_streamlines,
    plot_time_series,
    use_theme,
)
from pypic.plotting.styles import apply_rounding
from pypic.readers._field_dataset import TabularData
from tests._helpers import make_harris_dataset

OUTPUT_DIR = Path(__file__).parent / "output"


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


def _round_all(fig: plt.Figure) -> None:
    for ax in fig.get_axes():
        apply_rounding(ax)


def _save(fig: plt.Figure, plot_name: str, theme: PlotTheme) -> None:
    path = OUTPUT_DIR / f"{plot_name}_{theme.name}.png"
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  {path.name}")


def generate(theme: PlotTheme) -> None:
    print(f"Generating {theme.name} theme plots...")

    ds_a = make_harris_dataset(y_center=7.5)
    ds_b = make_harris_dataset(y_center=8.0)
    tabular = _make_tabular()

    # 1. Single slice — stored field + badge
    fig, ax = plot_field_slice(ds_a, "B1", theme=theme, step=100, time=5.0)
    add_badge(ax, step=100, time=5.0)
    _save(fig, "slice_B1", theme)

    # 2. Single slice — derived positive-definite field
    fig, _ = plot_field_slice(ds_a, "|B|", theme=theme, step=100)
    _save(fig, "slice_Bmag", theme)

    # 3. Comparison — three-panel A|B|diff + badge on first panel
    fig, axes = plot_comparison(
        ds_a, ds_b, "B1", theme=theme, labels=("y₀=7.5", "y₀=8.0"), step=100
    )
    add_badge(axes["a"], step=100, loc="upper left")
    _save(fig, "comparison", theme)

    # 4. Line overlay — B components along x
    fig, ax = plot_line(ds_a, "B1", axis="x", theme=theme, label="$B_x$", color="C0")
    plot_line(ds_a, "B2", axis="x", ax=ax, theme=theme, label="$B_y$", color="C1")
    plot_line(ds_a, "B3", axis="x", ax=ax, theme=theme, label="$B_z$", color="C2")
    ax.set_title("Magnetic field components along x")
    _save(fig, "lines_overlay", theme)

    # 5. Time series
    fig, _ = plot_time_series(
        tabular,
        ["total_energy", "kinetic_energy", "magnetic_energy"],
        labels=["Total", "Kinetic", "Magnetic"],
        theme=theme,
        title="Energy evolution",
        ylabel="Energy",
    )
    _save(fig, "time_series", theme)

    # 6. Crowded 2x3 grid of slices + badges on each panel
    slice_fields = ["B1", "|B|", "beta", "v_A", "e_B", "rho_m"]
    with use_theme(theme):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for i, (ax, field) in enumerate(zip(axes.flat, slice_fields, strict=True)):
            plot_field_slice(ds_a, field, ax=ax, theme=theme)
            add_badge(ax, step=100 + i * 10, fontsize=7)
        fig.suptitle("Field overview", fontsize=13)
        fig.tight_layout()
        _round_all(fig)
    _save(fig, "grid_slices", theme)

    # 7. Crowded 2x3 grid of line profiles
    with use_theme(theme):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, field in zip(axes.flat, slice_fields, strict=True):
            plot_line(ds_a, field, axis="x", ax=ax, theme=theme)
        fig.suptitle("Line profiles along x", fontsize=13)
        fig.tight_layout()
        _round_all(fig)
    _save(fig, "grid_lines", theme)

    # 8. Badge showcase — 2x3 grid: dark/light mode, custom labels, colors
    with use_theme(theme):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax in axes.flat:
            plot_field_slice(ds_a, "B1", ax=ax, theme=theme)

        add_badge(axes[0, 0], step=42, loc="upper left")
        add_badge(axes[0, 1], time=3.14, variant="lighter", loc="upper right")
        add_badge(axes[0, 2], step=100, time=5.0, loc="lower left")
        add_badge(axes[1, 0], step=100, label="cycle", loc="upper left")
        add_badge(
            axes[1, 1],
            step=42,
            label="",
            bg_color="#1a5276",
            text_color="gold",
            bg_alpha=0.75,
            loc="upper right",
        )
        add_badge(
            axes[1, 2],
            time=0.005,
            time_units="ns",
            label="time",
            loc="lower right",
        )
        fig.suptitle("Badge placement showcase", fontsize=13)
        fig.tight_layout()
        _round_all(fig)
    _save(fig, "badge_showcase", theme)

    # 9. Badge progress — 1x3 row showing early/mid/late + show_max toggle
    with use_theme(theme):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        step_range = (0, 500)
        for ax, s in zip(axes, [20, 250, 480], strict=True):
            plot_field_slice(ds_a, "|B|", ax=ax, theme=theme)
            add_badge(
                ax,
                step=s,
                step_range=step_range,
                show_max=s != 250,
                loc="upper right",
            )
        fig.suptitle("Progress bar showcase", fontsize=13)
        fig.tight_layout()
        _round_all(fig)
    _save(fig, "badge_progress", theme)

    # 10. Streamlines — B field on midplane
    fig, _ = plot_streamlines(ds_a, "B", theme=theme, step=100)
    _save(fig, "streamlines_B", theme)

    # 11. Streamlines — colored by |B|
    fig, _ = plot_streamlines(ds_a, "B", color_field="|B|", theme=theme, step=100)
    _save(fig, "streamlines_B_color_Bmag", theme)

    # 12. Quiver — V field on midplane
    fig, _ = plot_quiver(ds_a, "V", stride=2, theme=theme, step=100)
    _save(fig, "quiver_V", theme)

    # 13. Quiver — B field with custom stride
    fig, _ = plot_quiver(ds_a, "B", stride=(3, 2), theme=theme, step=100)
    _save(fig, "quiver_B", theme)

    # 14. Side-by-side streamlines + quiver
    with use_theme(theme):
        fig, (ax_stream, ax_quiv) = plt.subplots(1, 2, figsize=(14, 5))
        plot_streamlines(ds_a, "B", ax=ax_stream, theme=theme)
        ax_stream.set_title("Streamlines")
        plot_quiver(ds_a, "B", stride=3, ax=ax_quiv, theme=theme)
        ax_quiv.set_title("Quiver")
        fig.suptitle("Vector field visualization", fontsize=13)
        fig.tight_layout()
        _round_all(fig)
    _save(fig, "vectors_side_by_side", theme)

    # 15. Streamlines overlay — black lines on scalar field
    fig, ax = plot_field_slice(ds_a, "rho_m", theme=theme, step=100)
    plot_streamlines(
        ds_a,
        "B",
        ax=ax,
        color="black",
        linewidth=0.8,
        alpha=0.6,
        colorbar=False,
        theme=theme,
    )
    _save(fig, "streamlines_overlay", theme)

    # 16. Quiver overlay — white arrows on scalar field
    fig, ax = plot_field_slice(ds_a, "|B|", theme=theme, step=100)
    plot_quiver(
        ds_a,
        "V",
        ax=ax,
        color="white",
        alpha=0.7,
        stride=3,
        colorbar=False,
        theme=theme,
    )
    _save(fig, "quiver_overlay", theme)

    # 17. Inset colorbar — single slice
    fig, _ = plot_field_slice(ds_a, "B1", theme=theme, colorbar="inset", step=100)
    _save(fig, "inset_colorbar_slice", theme)

    # 18. Inset colorbar + badge on same plot
    fig, ax = plot_field_slice(ds_a, "|B|", theme=theme, colorbar="inset", step=100)
    add_badge(ax, step=100, time=5.0, loc="upper right")
    _save(fig, "inset_colorbar_with_badge", theme)

    # 19. Side colorbar vs inset colorbar comparison
    with use_theme(theme):
        fig, (ax_side, ax_inset) = plt.subplots(1, 2, figsize=(12, 5))
        plot_field_slice(ds_a, "B1", ax=ax_side, theme=theme, colorbar=True)
        ax_side.set_title("Side colorbar")
        plot_field_slice(ds_a, "B1", ax=ax_inset, theme=theme, colorbar="inset")
        ax_inset.set_title("Inset colorbar")
        fig.suptitle("Colorbar comparison", fontsize=13)
        fig.tight_layout()
        _round_all(fig)
    _save(fig, "colorbar_side_vs_inset", theme)

    # 20. Inset colorbar with darker variant on manually placed colorbar
    fig, ax = plot_field_slice(ds_a, "rho_m", theme=theme, colorbar=False, step=100)
    mesh = ax.get_children()[0]
    add_inset_colorbar(
        ax,
        mesh,
        r"$\rho_m$",
        variant="darker",
        loc="lower left",
        fontsize=8,
    )
    _save(fig, "inset_colorbar_manual", theme)

    # 21. Multi-entry vector legend (B + V overlaid)
    fig, ax = plot_field_slice(ds_a, "rho_m", theme=theme, step=100)
    plot_streamlines(
        ds_a,
        "B",
        ax=ax,
        color="black",
        linewidth=0.8,
        alpha=0.6,
        colorbar=False,
        legend=False,
        theme=theme,
    )
    plot_quiver(
        ds_a,
        "V",
        ax=ax,
        color="red",
        alpha=0.7,
        stride=3,
        colorbar=False,
        legend=False,
        theme=theme,
    )
    add_legend(
        ax,
        [
            LegendEntry(label="B field", color="black", linewidth=0.8, alpha=0.6),
            LegendEntry(label="V flow", color="red", alpha=0.7),
        ],
    )
    _save(fig, "multi_entry_vector_legend", theme)

    # 22. Panel label grid (2x3 with a-f)
    slice_labels = ["B1", "|B|", "beta", "v_A", "e_B", "rho_m"]
    with use_theme(theme):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for i, (ax, fld) in enumerate(zip(axes.flat, slice_labels, strict=True)):
            plot_field_slice(ds_a, fld, ax=ax, theme=theme)
            add_label(ax, chr(ord("a") + i))
        fig.suptitle("Panel labels a-f", fontsize=13)
        fig.tight_layout()
        _round_all(fig)
    _save(fig, "panel_labels_grid", theme)

    # 23. Combined: panel labels + badges + vector legend
    with use_theme(theme):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plot_field_slice(ds_a, "rho_m", ax=ax1, theme=theme)
        add_label(ax1, "a")
        add_badge(ax1, step=100, time=5.0, loc="upper right")
        plot_streamlines(
            ds_a,
            "B",
            ax=ax1,
            color="black",
            linewidth=0.8,
            alpha=0.5,
            colorbar=False,
            legend=False,
            theme=theme,
        )
        add_legend(ax1, LegendEntry(label="B", color="black"), loc="lower left")

        plot_field_slice(ds_b, "rho_m", ax=ax2, theme=theme)
        add_label(ax2, "b")
        add_badge(ax2, step=100, time=5.0, loc="upper right")
        fig.suptitle("Combined overlays", fontsize=13)
        fig.tight_layout()
        _round_all(fig)
    _save(fig, "combined_overlays", theme)

    # 24. Scalar + vector overlay — streamlines
    fig, ax = plot_field_slice(ds_a, "|B|", theme=theme, step=100, time=5.0)
    plot_streamlines(
        ds_a,
        "B",
        ax=ax,
        theme=theme,
        colorbar=False,
        legend=False,
        title="",
    )
    _save(fig, "overlay_streamlines", theme)

    # 25. Scalar + vector overlay — quiver
    fig, ax = plot_field_slice(ds_a, "rho_m", theme=theme, step=100)
    plot_quiver(
        ds_a,
        "V",
        ax=ax,
        stride=3,
        theme=theme,
        colorbar=False,
        legend=False,
        title="",
    )
    _save(fig, "overlay_quiver", theme)

    # 26. Multi-panel field grid
    fig, _ = plot_field_grid(
        ds_a,
        ["B1", "|B|", "beta", "v_A", "e_B", "rho_m"],
        ncols=3,
        theme=theme,
        step=100,
        colorbar="inset",
    )
    _save(fig, "field_grid", theme)

    # 27. Cross-section — B1 with x-cut
    fig, _ = plot_cross_section(
        ds_a,
        "B1",
        cut_axis="x",
        theme=theme,
        step=100,
    )
    _save(fig, "cross_section_x", theme)

    # 28. Cross-section — |B| with y-cut
    fig, _ = plot_cross_section(
        ds_a,
        "|B|",
        cut_axis="y",
        cut_index=10,
        theme=theme,
        step=100,
    )
    _save(fig, "cross_section_y", theme)

    # 29. Contour lines over scalar field
    fig, ax = plot_field_slice(ds_a, "|B|", theme=theme, step=100)
    add_contours(ax, ds_a, "P", levels=6, colors="white", linewidths=0.7)
    _save(fig, "contours_over_slice", theme)

    # 30. Multi-field line overlay (plot_lines)
    fig, _ = plot_lines(
        ds_a,
        ["B1", "B2", "B3"],
        axis="x",
        labels=["$B_x$", "$B_y$", "$B_z$"],
        theme=theme,
        title="B components along x",
    )
    _save(fig, "plot_lines", theme)

    # 31. Log scale — density
    fig, _ = plot_field_slice(
        ds_a,
        "rho_m",
        theme=theme,
        log_scale=True,
        step=100,
    )
    _save(fig, "log_scale_rho", theme)

    # 32. badge=True shortcut
    fig, _ = plot_field_slice(
        ds_a,
        "B1",
        theme=theme,
        step=100,
        time=5.0,
        badge=True,
    )
    _save(fig, "badge_shortcut", theme)

    # 33. Custom text badge + progress
    fig, ax = plot_field_slice(ds_a, "|B|", theme=theme, step=100)
    add_badge(ax, "Harris sheet, δ = 0.5 d_i", progress=0.6)
    _save(fig, "badge_custom_text", theme)

    # 34. coord_units as tuple
    fig, _ = plot_field_slice(
        ds_a,
        "B1",
        theme=theme,
        coord_units=("$d_i$", "$d_i$"),
        step=100,
    )
    _save(fig, "coord_units_tuple", theme)

    # 35. Comparison with error metric
    fig, _ = plot_comparison(
        ds_a,
        ds_b,
        "B1",
        theme=theme,
        labels=("y₀=7.5", "y₀=8.0"),
        show_error=True,
    )
    _save(fig, "comparison_with_error", theme)

    # 36. Extremes modes — transparent, semi, none, darken
    fig, ax = plot_field_slice(
        ds_a,
        "|B|",
        theme=theme,
        step=100,
        vmin=0.2,
        vmax=0.8,
        extremes="transparent",
        title="|B| clipped — transparent outside",
    )
    _save(fig, "transparent_extremes", theme)

    fig, ax = plot_field_slice(
        ds_a,
        "rho_m",
        theme=theme,
        step=100,
        colorbar="inset",
    )
    plot_field_slice(
        ds_a,
        "B1",
        theme=theme,
        ax=ax,
        colorbar=False,
        vmin=-0.3,
        vmax=0.3,
        extremes="transparent",
        alpha=0.7,
        cmap="coolwarm",
    )
    ax.set_title(r"$\rho_m$ + transparent $B_x$ overlay")
    _save(fig, "transparent_overlay", theme)

    fig, _ = plot_field_slice(
        ds_a,
        "|B|",
        theme=theme,
        step=100,
        vmin=0.2,
        vmax=0.8,
        extremes="semi",
        title="|B| clipped — semi-transparent outside",
    )
    _save(fig, "semi_extremes", theme)

    fig, _ = plot_field_slice(
        ds_a,
        "|B|",
        theme=theme,
        step=100,
        vmin=0.2,
        vmax=0.8,
        extremes=None,
        title="|B| clipped — matplotlib default",
    )
    _save(fig, "none_extremes", theme)

    fig, _ = plot_field_slice(
        ds_a,
        "|B|",
        theme=theme,
        step=100,
        vmin=0.2,
        vmax=0.8,
        extremes="darken",
        title="|B| clipped — darkened outside",
    )
    _save(fig, "darken_extremes", theme)

    # 37. Customized theme — larger fonts
    big_theme = theme.customize(name="talk", font_size=14, axes_titlesize=16)
    fig, _ = plot_field_slice(
        ds_a,
        "|B|",
        theme=big_theme,
        step=100,
    )
    _save(fig, "customized_theme", theme)

    # 38. Badge variants showcase
    with use_theme(theme):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax in axes.flat:
            plot_field_slice(ds_a, "B1", ax=ax, theme=theme)
        add_badge(axes[0, 0], step=42, loc="upper left")
        add_badge(axes[0, 1], step=102312, label="Cycle", loc="upper left")
        add_badge(axes[0, 2], time="13:34", loc="upper right")
        add_badge(axes[1, 0], "Run A: high β", loc="upper left")
        add_badge(
            axes[1, 1],
            step=250,
            step_range=(0, 500),
            loc="upper right",
        )
        add_badge(
            axes[1, 2],
            "Processing...",
            progress=0.4,
            loc="upper right",
        )
        fig.suptitle("Badge variants", fontsize=13)
        fig.tight_layout()
        _round_all(fig)
    _save(fig, "badge_variants", theme)

    # 39. Overlay with badge
    fig, ax = plot_field_slice(ds_a, "|B|", theme=theme, step=100, badge=True)
    plot_streamlines(
        ds_a,
        "B",
        ax=ax,
        theme=theme,
        colorbar=False,
        legend=False,
        title="",
    )
    _save(fig, "overlay_badge", theme)

    # 40. Kymograph — Bx current sheet drift
    kymo_coords = np.linspace(0, 19.5, 40)
    kymo_times = np.arange(8, dtype=float)
    kymo_profiles = np.array(
        [np.tanh((kymo_coords - 10 + 0.3 * t) / 2.0) for t in kymo_times]
    )
    fig, _ = plot_kymograph(
        kymo_profiles,
        kymo_coords,
        kymo_times,
        label="$B_x$",
        xlabel="$x$ [$d_i$]",
        ylabel=r"$t$ [$\Omega_i^{-1}$]",
        title="Current sheet drift",
        theme=theme,
    )
    _save(fig, "kymograph_bx", theme)

    # 41. Scatter — Bx vs By colored by |B|
    from pypic.selections import PlaneSelection

    fig, _ = plot_scatter(
        ds_a,
        "B1",
        "B2",
        color_field="|B|",
        plane=PlaneSelection(normal="z"),
        theme=theme,
        title="Field component correlation",
    )
    _save(fig, "scatter_B1_B2", theme)

    # 42. Scatter density — rho vs P (equation of state)
    fig, _ = plot_scatter(
        ds_a,
        "rho_m",
        "P",
        density=True,
        plane=PlaneSelection(normal="z"),
        theme=theme,
        title="Equation of state",
    )
    _save(fig, "scatter_density_rho_P", theme)

    # 43. Power spectrum — 1D Bx spectrum with reference slope
    from pypic.spectral import power_spectrum_1d

    bx_profile = ds_a["B1"][:, 15, 10]
    k, power = power_spectrum_1d(bx_profile, ds_a.grid.spacing[0])
    fig, _ = plot_power_spectrum(
        k,
        power,
        label="$B_x$",
        reference_slopes=[-5.0 / 3.0],
        theme=theme,
        title="Magnetic field spectrum",
    )
    _save(fig, "power_spectrum_bx", theme)

    # 44. Line comparison — two runs
    fig, _ = plot_line_comparison(
        [ds_a, ds_b],
        "B1",
        axis="x",
        labels=["$y_0 = 7.5$", "$y_0 = 8.0$"],
        coord_units="$d_i$",
        theme=theme,
        title="Current sheet comparison",
    )
    _save(fig, "line_comparison_B1", theme)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visual test plots for manual inspection."
    )
    all_available = available_themes()
    names = sorted(all_available)
    numbered = [f"{i + 1}={n}" for i, n in enumerate(names)]
    parser.add_argument(
        "--theme",
        default="all",
        help=f"Theme name, number, or 'all' ({', '.join(numbered)})",
    )
    args = parser.parse_args()

    # Clear previous output
    if OUTPUT_DIR.exists():
        for f in OUTPUT_DIR.glob("*.png"):
            f.unlink()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    themes: list[PlotTheme] = []
    if args.theme == "all":
        themes = list(all_available.values())
    elif args.theme.isdigit():
        themes = [all_available[names[int(args.theme) - 1]]]
    else:
        themes = [all_available[args.theme]]

    for theme in themes:
        generate(theme)

    count = len(list(OUTPUT_DIR.glob("*.png")))
    print(f"\nDone. {count} PNGs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
