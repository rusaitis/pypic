"""Layout and clipping every plot applies to the axes it draws on."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pypic.plotting import (
    plot_field_slice,
    plot_kymograph,
    plot_line,
    plot_power_spectrum,
    plot_quiver,
    plot_scatter,
    plot_streamlines,
    plot_time_series,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes

    from pypic.containers import TabularData
    from pypic.dataset import FieldDataset

type Draw = Callable[[FieldDataset, TabularData, Axes], object]

_WAVENUMBERS = np.linspace(0.1, 10.0, 20)
_DRAWS: list[Draw] = [
    lambda ds, _, ax: plot_field_slice(ds, "B_1", ax=ax),
    lambda ds, _, ax: plot_streamlines(ds, "B", ax=ax),
    lambda ds, _, ax: plot_quiver(ds, "B", ax=ax),
    lambda ds, _, ax: plot_scatter(ds, "B_1", "B_2", ax=ax),
    lambda ds, _, ax: plot_line(ds, "B_1", axis="x", ax=ax),
    lambda _, table, ax: plot_time_series(table, "total_energy", ax=ax),
    lambda ds, _, ax: plot_kymograph(ds["B_1"], np.arange(8.0), np.arange(10.0), ax=ax),
    lambda _, __, ax: plot_power_spectrum(_WAVENUMBERS, _WAVENUMBERS**-2, ax=ax),
]
_IDS = [
    "slice",
    "streamlines",
    "quiver",
    "scatter",
    "line",
    "time-series",
    "kymograph",
    "spectrum",
]


@pytest.mark.parametrize("draw", _DRAWS, ids=_IDS)
def test_drawing_on_supplied_axes_leaves_the_callers_layout_alone(
    ds_2d: FieldDataset, tabular: TabularData, draw: Draw
) -> None:
    """Only a plot that created the figure lays it out; a neighbour panel of
    the caller's figure keeps its position."""
    fig, (target, neighbour) = plt.subplots(1, 2)
    before = neighbour.get_position().bounds
    draw(ds_2d, tabular, target)
    assert neighbour.get_position().bounds == before
    plt.close(fig)


def test_an_overlay_is_clipped_to_the_rounded_axes_like_the_base_plot(
    ds_2d: FieldDataset,
) -> None:
    """A second line drawn onto rounded axes gets the same rounded clip."""
    fig, ax = plot_line(ds_2d, "B_1", axis="x", theme="light")
    plot_line(ds_2d, "B_2", axis="x", ax=ax, theme="light")
    assert all(line.get_clip_path() is not None for line in ax.lines)
    plt.close(fig)
