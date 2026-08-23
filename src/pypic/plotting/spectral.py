"""Power spectrum plots on log-log axes with reference slopes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pypic.plotting._guard import ensure_matplotlib

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from pypic.plotting.styles import ThemeArg
    from pypic.types import FloatArray


def plot_power_spectrum(
    k: FloatArray,
    power: FloatArray,
    *,
    label: str | None = None,
    compensated: float | None = None,
    reference_slopes: list[float] | None = None,
    theme: ThemeArg = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    save: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401 — matplotlib passthrough
) -> tuple[Figure, Axes]:
    r"""Plot a power spectrum on log-log axes.

    Designed for output from [`power_spectrum_1d`][pypic.spectral.power_spectrum_1d]
    or [`power_spectrum_2d`][pypic.spectral.power_spectrum_2d].

    Parameters
    ----------
    k : FloatArray
        Wavenumber array.
    power : FloatArray
        Power spectral density.
    label : str | None
        Legend label for this spectrum.
    compensated : float | None
        Multiply power by $k^n$ before plotting (e.g. ``5/3`` for
        Kolmogorov compensation). ``None`` plots raw power.
    reference_slopes : list[float] | None
        Draw dashed reference lines with these slopes on the log-log
        plot (e.g. ``[-5/3, -3]``). Each line is auto-positioned to
        pass through the geometric midpoint of the spectrum.
    theme : PlotTheme | None
        Plot theme. ``None`` uses ``DEFAULT``.
    xlabel : str | None
        X-axis label. ``None`` defaults to ``"$k$"``.
    ylabel : str | None
        Y-axis label. ``None`` auto-generates based on *compensated*.
    title : str | None
        Axes title.
    ax : Axes | None
        Existing axes. ``None`` creates a new figure.
    save : str | None
        Save figure to this path.
    figsize : tuple[float, float] | None
        Figure size override.
    **kwargs
        Passed to ``ax.loglog()`` (color, linestyle, linewidth, etc.).

    Returns
    -------
    tuple[Figure, Axes]
    """
    ensure_matplotlib()

    from pypic.plotting._resolve import get_or_create_axes
    from pypic.plotting.styles import (
        _resolve_theme_arg,
        apply_grid,
        apply_rounding,
        style_legend,
        use_theme,
    )

    owned = ax is None
    theme = _resolve_theme_arg(theme)

    plot_power = power * k**compensated if compensated is not None else power

    with use_theme(theme):
        fig, ax = get_or_create_axes(theme, ax, figsize)

        ax.loglog(k, plot_power, label=label, **kwargs)

        # Reference slope lines
        if reference_slopes:
            mid_idx = len(k) // 2
            k_mid = k[mid_idx]
            p_mid = plot_power[mid_idx]

            for slope in reference_slopes:
                ref_line = p_mid * (k / k_mid) ** slope
                # Format slope as fraction where possible
                if abs(slope - round(slope)) < 1e-10:
                    slope_str = f"{round(slope)}"
                else:
                    from fractions import Fraction

                    frac = Fraction(slope).limit_denominator(10)
                    slope_str = f"{frac.numerator}/{frac.denominator}"
                ax.loglog(
                    k,
                    ref_line,
                    "--",
                    alpha=0.4,
                    color="gray",
                    linewidth=0.8,
                    label=f"$k^{{{slope_str}}}$",
                )

        ax.set_xlabel(xlabel if xlabel is not None else "$k$")
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        elif compensated is not None:
            ax.set_ylabel(f"$k^{{{compensated:.2g}}} P(k)$")
        else:
            ax.set_ylabel("$P(k)$")

        if title is not None:
            ax.set_title(title)

        apply_grid(ax, theme, minor=True)

        if label is not None or reference_slopes:
            ax.legend()
            style_legend(ax)

        if owned:
            fig.tight_layout()
            apply_rounding(ax)

    from pypic.plotting._resolve import maybe_save

    maybe_save(fig, save)
    return fig, ax
