"""Field-line and trajectory tubes: one color scale per batch, singles included."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("pyvista")

from pypic.dataset import FieldDataset
from pypic.plotting import get_theme
from pypic.plotting.pyvista import (
    add_field_line,
    add_field_lines,
    add_trajectories,
    add_trajectory,
)
from pypic.traces import FieldLine, ParticleTrace
from pypic.units import Normalization
from tests._helpers import make_uniform_grid

_STRAIGHT = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])


class _RecordingPlotter:
    """Stands in for ``pv.Plotter``; tubes build without a render window."""

    def __init__(self) -> None:
        self.meshes: list[dict[str, Any]] = []

    def add_mesh(self, mesh: object, **options: Any) -> dict[str, Any]:
        self.meshes.append(options)
        return options


def _styles(plotter: _RecordingPlotter) -> list[tuple[object, ...]]:
    return [
        (m.get("color"), getattr(m.get("cmap"), "name", None), m.get("clim"))
        for m in plotter.meshes
    ]


def _line(b_1: list[float]) -> FieldLine:
    return FieldLine(
        points=_STRAIGHT,
        field_name="B",
        seed_point=(0.0, 0.0, 0.0),
        normalization=Normalization.identity(),
        scalars={"B_1": np.array(b_1)},
    )


def _trace(**scalars: np.ndarray) -> ParticleTrace:
    return ParticleTrace(
        points=_STRAIGHT,
        time=np.array([0.0, 0.5, 1.0]),
        velocity=np.ones((3, 3)),
        species_name="electrons",
        normalization=Normalization.identity(),
        scalars=scalars,
    )


@pytest.fixture
def b_field_metadata() -> FieldDataset:
    return FieldDataset.from_arrays(
        {"B_1": np.ones((3, 3))}, make_uniform_grid(3, 3), Normalization.identity()
    )


@pytest.mark.parametrize("scalar", [None, "B_1"], ids=["uniform", "scalar"])
def test_a_single_field_line_draws_like_a_one_line_batch(
    b_field_metadata: FieldDataset, scalar: str | None
) -> None:
    single, batch = _RecordingPlotter(), _RecordingPlotter()
    line = _line([-1.0, 0.2, 0.5])
    options = {"scalar": scalar, "data": b_field_metadata, "show_scalar_bar": False}
    add_field_line(single, line, **options)
    add_field_lines(batch, [line], **options)
    assert _styles(single) == _styles(batch)


@pytest.mark.parametrize("scalar", [None, "speed"], ids=["uniform", "scalar"])
def test_a_single_trajectory_draws_like_a_one_trace_batch(scalar: str | None) -> None:
    single, batch = _RecordingPlotter(), _RecordingPlotter()
    add_trajectory(single, _trace(), scalar=scalar, show_scalar_bar=False)
    add_trajectories(batch, [_trace()], scalar=scalar, show_scalar_bar=False)
    assert _styles(single) == _styles(batch)


def test_signed_field_lines_share_one_zero_centred_diverging_scale(
    b_field_metadata: FieldDataset,
) -> None:
    plotter = _RecordingPlotter()
    lines = [_line([-1.0, 0.2, 0.5]), _line([-0.2, 0.1, 2.0])]
    add_field_lines(
        plotter, lines, scalar="B_1", data=b_field_metadata, show_scalar_bar=False
    )
    expected = (None, get_theme().diverging_cmap, (-2.0, 2.0))
    assert set(_styles(plotter)) == {expected}


def test_an_unknown_trajectory_scalar_raises() -> None:
    with pytest.raises(KeyError, match="energy"):
        add_trajectories(_RecordingPlotter(), [_trace()], scalar="energy")


def test_a_trace_scalar_named_speed_wins_over_the_derived_one() -> None:
    plotter = _RecordingPlotter()
    add_trajectory(plotter, _trace(speed=np.array([5.0, 6.0, 7.0])), scalar="speed")
    assert plotter.meshes[0]["clim"] == (5.0, 7.0)
