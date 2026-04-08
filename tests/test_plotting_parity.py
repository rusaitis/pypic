"""Plotting parity tests: matplotlib backend ↔ pyvista backend.

These tests treat the public parameter sets of paired plotting functions
as data and assert their symmetric difference is contained in a curated
"intentional gap" allowlist. When a parameter is added to one backend,
the test fails until the author either lifts it to the other backend or
documents it here as a backend-specific gap.

Unit 9 of the cleanup sweep — see ``TASKS-cleanup.md``.
"""

from __future__ import annotations

import inspect
from typing import Callable

import pytest

pyvista = pytest.importorskip("pyvista")

import pypic.plotting as mpl_plot  # noqa: E402
import pypic.plotting.pyvista as pv_plot  # noqa: E402
from pypic.plotting.annotations import add_circle, add_planet  # noqa: E402


def _params(func: Callable[..., object]) -> set[str]:
    """Return the public keyword parameters of *func* (excluding the first positional)."""
    sig = inspect.signature(func)
    names = list(sig.parameters)
    # Drop the leading positional (ax / plotter / fig) — it's the backend
    # handle and always differs.
    return set(names[1:])


# (mpl_function, pv_function, intentional_gaps)
# intentional_gaps are parameters that legitimately exist in only one
# backend; each entry is a (param, side, reason) triple where *side*
# is "mpl" or "pv". Add an entry here when introducing a backend-only
# parameter and document why.
PAIRS: list[tuple[Callable[..., object], Callable[..., object], list[tuple[str, str, str]]]] = [
    # ── add_badge ─────────────────────────────────────────────────────
    (
        mpl_plot.add_badge,
        pv_plot.add_badge,
        [
            ("width", "pv", "Legacy alias for bar_width (predates rename)"),
            ("height", "pv", "Legacy alias for bar_height (predates rename)"),
            ("theme", "pv", "Pyvista has no use_theme() context — explicit theme arg"),
        ],
    ),
    # ── add_label ─────────────────────────────────────────────────────
    (
        mpl_plot.add_label,
        pv_plot.add_label,
        [
            ("width", "pv", "Box width override (no equivalent in mpl AnchoredOffsetbox path)"),
            ("height", "pv", "Box height override"),
            ("theme", "pv", "Pyvista has no use_theme() context"),
        ],
    ),
    # ── add_inset_colorbar (mpl) ↔ add_colorbar (pv) ──────────────────
    # The mpl side uses add_inset_colorbar because the pyvista
    # add_colorbar takes (cmap, clim) instead of (mappable) and is
    # therefore the closer analog of the inset overlay variant.
    (
        mpl_plot.add_inset_colorbar,
        pv_plot.add_colorbar,
        [
            # mpl-only: tied to the mappable-based pipeline
            ("mappable", "mpl", "Pyvista builds its own gradient strip from cmap+clim"),
            ("pad", "mpl", "Inset offset; pyvista positions via overlay_margin"),
            ("n_ticks", "mpl", "Pyvista uses n_labels (same role, different name)"),
            ("fontsize", "mpl", "Pyvista derives font size from theme.colorbar_*_font_scale"),
            ("bg_color", "mpl", "Pyvista colorbar uses theme overlay variant only"),
            ("bg_alpha", "mpl", "Pyvista colorbar uses theme overlay variant only"),
            ("text_color", "mpl", "Pyvista colorbar uses theme overlay variant only"),
            ("text_alpha", "mpl", "Pyvista colorbar uses theme overlay variant only"),
            # pv-only: pyvista needs the cmap/clim because there is no mappable
            ("cmap", "pv", "No matplotlib mappable to inspect; user passes cmap directly"),
            ("clim", "pv", "No matplotlib mappable to inspect; user passes clim directly"),
            ("fmt", "pv", "Tick label format (mpl uses MaxNLocator + FuncFormatter internally)"),
            ("n_labels", "pv", "Mirrors mpl's n_ticks under a different name"),
            ("theme", "pv", "Pyvista has no use_theme() context"),
        ],
    ),
    # ── add_planet (annotations) ↔ pyvista add_planet ─────────────────
    (
        add_planet,
        pv_plot.add_planet,
        [
            # mpl draws a 2D wedge in axes data space; pyvista draws a
            # 3D sphere mesh. The differences are inherent to those
            # rendering models, not parity oversights.
            ("edgecolor", "mpl", "2D wedge has a stroke; 3D sphere does not"),
            ("linewidth", "mpl", "2D wedge stroke width"),
            ("zorder", "mpl", "2D z-ordering; pyvista uses depth from the camera"),
            ("resolution", "pv", "Sphere tessellation count (no analog for a 2D wedge)"),
        ],
    ),
    # ── add_circle (mpl) ↔ add_reference_circles (pv) ─────────────────
    # Different by design: mpl draws ONE labeled circle, pyvista draws
    # MANY unlabeled reference rings. Listed here to keep the contract
    # explicit — if either side ever grows toward the other, this
    # allowlist is the place to document why.
    (
        add_circle,
        pv_plot.add_reference_circles,
        [
            ("radius", "mpl", "Single circle (mpl); pyvista takes a list as `radii`"),
            ("label", "mpl", "Mpl supports a perimeter label; pyvista does not"),
            ("label_position", "mpl", "Label angle on the perimeter"),
            ("alpha", "mpl", "Mpl uses `alpha`; pyvista uses `opacity`"),
            ("linestyle", "mpl", "Mpl line style; pyvista renders dashed via point gaps"),
            ("linewidth", "mpl", "Mpl uses `linewidth`; pyvista uses `width`"),
            ("fontsize", "mpl", "Label font size — pyvista has no labels"),
            ("text_alpha", "mpl", "Label opacity — pyvista has no labels"),
            ("variant", "mpl", "Label overlay variant — pyvista has no labels"),
            ("zorder", "mpl", "2D z-ordering; pyvista uses depth"),
            ("radii", "pv", "Pyvista renders multiple circles in one call"),
            ("z", "pv", "z-plane for the rings (no analog in 2D mpl axes)"),
            ("opacity", "pv", "Pyvista uses `opacity`; mpl uses `alpha`"),
            ("width", "pv", "Pyvista uses `width`; mpl uses `linewidth`"),
            ("n_points", "pv", "Polyline resolution (mpl draws a true Circle patch)"),
            ("theme", "pv", "Pyvista has no use_theme() context"),
        ],
    ),
]


@pytest.mark.parametrize(
    "mpl_func,pv_func,gaps",
    PAIRS,
    ids=[f"{m.__name__}↔{p.__name__}" for m, p, _ in PAIRS],
)
def test_paired_function_parity(
    mpl_func: Callable[..., object],
    pv_func: Callable[..., object],
    gaps: list[tuple[str, str, str]],
) -> None:
    """Symmetric diff of paired-function params must equal the allowlist."""
    mpl_params = _params(mpl_func)
    pv_params = _params(pv_func)
    expected_mpl_only = {p for p, side, _ in gaps if side == "mpl"}
    expected_pv_only = {p for p, side, _ in gaps if side == "pv"}

    actual_mpl_only = mpl_params - pv_params
    actual_pv_only = pv_params - mpl_params

    missing_from_allowlist_mpl = actual_mpl_only - expected_mpl_only
    extras_in_allowlist_mpl = expected_mpl_only - actual_mpl_only
    missing_from_allowlist_pv = actual_pv_only - expected_pv_only
    extras_in_allowlist_pv = expected_pv_only - actual_pv_only

    msg_parts = []
    if missing_from_allowlist_mpl:
        msg_parts.append(
            f"mpl-only params not in allowlist: {sorted(missing_from_allowlist_mpl)} "
            f"(add to PAIRS or lift to pyvista)"
        )
    if extras_in_allowlist_mpl:
        msg_parts.append(
            f"allowlist mpl entries that no longer exist: {sorted(extras_in_allowlist_mpl)} "
            f"(remove from PAIRS)"
        )
    if missing_from_allowlist_pv:
        msg_parts.append(
            f"pv-only params not in allowlist: {sorted(missing_from_allowlist_pv)} "
            f"(add to PAIRS or lift to matplotlib)"
        )
    if extras_in_allowlist_pv:
        msg_parts.append(
            f"allowlist pv entries that no longer exist: {sorted(extras_in_allowlist_pv)} "
            f"(remove from PAIRS)"
        )
    assert not msg_parts, "; ".join(msg_parts)


def test_pyvista_colorbar_accepts_lifted_params() -> None:
    """Verify the Unit 9 lifts (extremes, ticks) made it into the signature."""
    sig = inspect.signature(pv_plot.add_colorbar)
    assert "extremes" in sig.parameters
    assert "ticks" in sig.parameters


def test_pyvista_equatorial_surface_accepts_lifted_params() -> None:
    """add_equatorial_surface should expose extremes / colorbar_ticks for parity with plot_field_slice."""
    sig = inspect.signature(pv_plot.add_equatorial_surface)
    assert "extremes" in sig.parameters
    assert "colorbar_ticks" in sig.parameters
