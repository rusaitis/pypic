"""Internal label builders for axis labels, colorbar text, and titles."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pypic.fields import FieldInfo


def axis_label(axis_name: str, *, unit_str: str = "") -> str:
    """Build an axis label string.

    Parameters
    ----------
    axis_name : str
        Axis name (e.g. ``"x"``, ``"r"``).
    unit_str : str
        Unit string to append in brackets.

    Returns
    -------
    str
    """
    label = f"${axis_name}$"
    if unit_str:
        label += f" [{unit_str}]"
    return label


def field_label(info: FieldInfo, *, unit_str: str = "") -> str:
    """Build a field label from metadata (colorbars, y-axes, etc.).

    Parameters
    ----------
    info : FieldInfo
        Field metadata.
    unit_str : str
        Unit string to append in brackets.

    Returns
    -------
    str
    """
    text = info.latex or info.long_name
    if unit_str:
        text += f" [{unit_str}]"
    return text


def figure_title(
    info: FieldInfo,
    *,
    step: int | None = None,
    time: float | None = None,
) -> str:
    """Build a figure title from field metadata.

    Parameters
    ----------
    info : FieldInfo
        Field metadata.
    step : int | None
        Timestep number.
    time : float | None
        Simulation time.

    Returns
    -------
    str
    """
    parts = [info.long_name]
    if step is not None:
        parts.append(f"step {step}")
    if time is not None:
        parts.append(f"t = {time:.2f}")
    return ", ".join(parts)
