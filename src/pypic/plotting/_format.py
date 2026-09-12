"""Shared formatting helpers for pypic plotting overlays.

The matplotlib (`pypic.plotting._badge`) and pyvista
(`pypic.plotting.pyvista._badge`) backends import the same `BadgeLoc`
type alias and the same formatters from here, so badge content and
positioning stay in lockstep across the two.
"""

from __future__ import annotations

from typing import Literal

type BadgeLoc = Literal[
    "upper left", "upper right", "lower left", "lower right", "upper center"
]
"""Canonical overlay location strings shared by both plotting backends.

Space-separated form (e.g. ``"upper right"``) matches matplotlib's own
``AnchoredOffsetbox`` convention. The legacy pyvista underscore form
(``"upper_right"``) is still accepted by `_normalize_loc` for
backward compatibility.
"""


def _normalize_loc(loc: str) -> str:
    """Return the space-separated canonical form of a location string.

    Accepts both the matplotlib convention (``"upper right"``) and the
    legacy pyvista underscore form (``"upper_right"``). Normalization is
    idempotent: already-canonical strings pass through unchanged.
    """
    return loc.replace("_", " ")


def _format_time_value(time: float, units: str) -> tuple[str, str]:
    """Format a numeric time value, returning ``(value_str, suffix_str)``.

    When *units* is ``"s"`` and *time* >= 60, produces human-readable
    durations like ``"2min 30s"`` or ``"1h 5min 12s"`` (suffix is empty
    since units are embedded). Otherwise falls back to numeric formatting.
    """
    if units == "s" and abs(time) >= 60:
        sign = "-" if time < 0 else ""
        t = abs(time)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        if h > 0:
            return f"{sign}{h}h {m}min {s}s", ""
        return f"{sign}{m}min {s}s", ""
    suffix = f" {units}" if units else ""
    if abs(time) >= 1e4 or (0 < abs(time) < 0.01):
        return f"{time:.2e}", suffix
    return f"{time:.2f}", suffix


def _format_status_text(
    *,
    text: str | None,
    step: int | None,
    time: float | str | None,
    time_units: str,
    step_range: tuple[int, int] | None,
    label: str | None = None,
    show_max: bool = True,
) -> str:
    """Build the status text string from text/step/time parameters.

    Used by both the matplotlib and pyvista badge implementations so
    the rendered content stays identical across backends.

    - *text* (when given) wins outright; all other content parameters
      are ignored.
    - Otherwise, *step* and *time* each contribute a part; both together
      are joined with ``", "``.
    - *label* overrides the auto-prefix (``"step"`` / ``"t"``); ``""``
      suppresses the prefix entirely.
    - *step_range* + *show_max* renders ``"step X / Y"``.
    - *time* may be a float (auto-formatted via `_format_time_value`)
      or a string (used verbatim).
    """
    if text is not None:
        return text

    parts: list[str] = []

    if step is not None:
        step_label = "step" if label is None else label
        prefix = f"{step_label} " if step_label else ""
        if step_range is not None and show_max:
            parts.append(f"{prefix}{step} / {step_range[1]}")
        else:
            parts.append(f"{prefix}{step}")

    if time is not None:
        if isinstance(time, str):
            value = time
            suffix = f" {time_units}" if time_units else ""
        else:
            value, suffix = _format_time_value(time, time_units)
        # When time-only, label replaces "t"; with step, time keeps "t"
        time_label = "t" if step is not None else ("t" if label is None else label)
        if time_label:
            parts.append(f"{time_label} = {value}{suffix}")
        else:
            parts.append(f"{value}{suffix}")

    return ", ".join(parts) if parts else ""
