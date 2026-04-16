# Source: docs/schema.md § "Vector group shorthand in ``read()``":
#           "Passing a bare prefix like ``'B'`` to ``read(fields=...)``
#            expands to ``B1, B2, B3``. Per-species groups also work:
#            ``'EF_s0'`` expands to ``EF1_s0, EF2_s0, EF3_s0``."
#         + src/pypic/readers/_registry.py:353 (expansion loop).
# Claim: (a) ``read(fields=["B"])`` loads the same field set as
#            ``read(fields=["B1","B2","B3"])``.
#        (b) Adding explicit components to the bare prefix is a no-op
#            (idempotence): ``read(fields=["B","B1","B2","B3"])`` loads
#            the same set as ``read(fields=["B"])``.
#        (c) Per-species shorthand ``"EF_s0"`` expands to the three
#            ``EFk_s0`` components.
#        (d) Cartesian aliases (``"Bx"``) map to a single numbered
#            component, NOT the whole group — the caller opted out of
#            the group by naming one axis.
#        (e) With ``strict_fields=True``, an unknown bare prefix raises
#            ``KeyError`` (ties into backlog #8's fail-loud contract).
# Backlog #9 in autoresearcher-pypic.md.
"""Vector-group shorthand ``fields=[...]`` idempotence and coverage."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pypic.containers import SimulationConfig
from pypic.coordinates.geometry import CARTESIAN
from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.readers._registry import Simulation
from pypic.units import Normalization, SpeciesInfo

if TYPE_CHECKING:
    from pypic.types import FloatArray


class _RecordingReader:
    """Non-selective reader that returns a fixed field set.

    ``Simulation.read`` routes through the non-selective branch (does not
    pass ``fields=`` to the reader), then filters the returned dataset via
    ``select_fields(canonical & available)``. That filter is exactly
    where the vector-group expansion becomes observable as a field-name
    set — which is what we want to assert on.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[Path, int]] = []

    def available_timesteps(self, path: Path) -> list[int]:
        del path
        return [0]

    def read_timestep(self, path: Path, step: int) -> FieldDataset:
        self.calls.append((path, step))
        grid = GridInfo(
            dimensions=(4, 3, 2), spacing=(1.0, 1.0, 1.0), geometry=CARTESIAN
        )
        fields: dict[str, FloatArray] = {
            name: np.ones((4, 3, 2))
            for name in (
                "B1",
                "B2",
                "B3",
                "E1",
                "E2",
                "E3",
                "rho_c",
                "EF1_s0",
                "EF2_s0",
                "EF3_s0",
            )
        }
        return FieldDataset.from_arrays(fields, grid, Normalization.identity())


def _make_sim() -> Simulation:
    grid = GridInfo(dimensions=(4, 3, 2), spacing=(1.0, 1.0, 1.0), geometry=CARTESIAN)
    config = SimulationConfig(
        model_name="test",
        model_type="PIC",
        grid=grid,
        normalization=Normalization.identity(),
        species=(SpeciesInfo(name="e", charge=-1.0, mass=1.0),),
    )
    return Simulation(_RecordingReader(), config, path=Path("/tmp/test"))


def test_bare_prefix_expands_to_numbered_components() -> None:
    """``fields=["B"]`` loads ``{B1, B2, B3}`` — the canonical expansion."""
    sim = _make_sim()
    ds = sim.read(0, fields=["B"])
    assert set(ds.field_names()) == {"B1", "B2", "B3"}


def test_explicit_components_match_bare_prefix() -> None:
    """``fields=["B"]`` and ``fields=["B1","B2","B3"]`` load the same set.

    This is the primary equivalence claim from schema.md § 3 — the
    shorthand is a pure convenience wrapper, never a different result.
    """
    sim = _make_sim()
    from_group = set(sim.read(0, fields=["B"]).field_names())
    from_explicit = set(sim.read(0, fields=["B1", "B2", "B3"]).field_names())
    assert from_group == from_explicit == {"B1", "B2", "B3"}


def test_shorthand_is_idempotent() -> None:
    """Expanding twice == expanding once. Mixing the bare prefix with its
    explicit components must not duplicate, reorder, or drop anything.
    """
    sim = _make_sim()
    once = set(sim.read(0, fields=["B"]).field_names())
    twice = set(sim.read(0, fields=["B", "B1", "B2", "B3"]).field_names())
    assert once == twice


def test_per_species_shorthand_expands() -> None:
    """``"EF_s0"`` → ``{EF1_s0, EF2_s0, EF3_s0}``.

    Per schema.md: "component index comes before the species suffix",
    so the expansion injects the axis index between the prefix and the
    species suffix.
    """
    sim = _make_sim()
    ds = sim.read(0, fields=["EF_s0"])
    assert set(ds.field_names()) == {"EF1_s0", "EF2_s0", "EF3_s0"}


def test_cartesian_alias_loads_single_component() -> None:
    """``fields=["Bx"]`` loads ``{B1}``, not the whole group — the
    Cartesian alias names a single axis, so group expansion must be
    suppressed when the input is already alias-resolved.

    This encodes the "if name not in alias_map" guard in
    ``readers/_registry.py:368``: aliases skip the axis-fanout.
    """
    sim = _make_sim()
    ds = sim.read(0, fields=["Bx"])
    assert set(ds.field_names()) == {"B1"}


def test_unknown_prefix_with_strict_raises() -> None:
    """Bare-prefix expansion of an unknown name still yields zero loaded
    fields (``Xunknown1..3`` are not in the dataset); ``strict_fields=True``
    must turn that into ``KeyError`` rather than a silent empty dataset.
    """
    sim = _make_sim()
    with pytest.raises(KeyError):
        sim.read(0, fields=["Xunknown"], strict_fields=True)


def test_mixed_known_and_unknown_strict_raises() -> None:
    """Partial match — ``B`` resolves, ``Xunknown`` does not — still
    raises under strict mode. Encoding the "single bad entry fails the
    whole call" half of the fail-loud contract (backlog #8 at the
    ``read()`` boundary rather than ``resolve_key``).
    """
    sim = _make_sim()
    with pytest.raises(KeyError):
        sim.read(0, fields=["B", "Xunknown"], strict_fields=True)
