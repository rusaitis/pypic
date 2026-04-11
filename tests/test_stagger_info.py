"""Tests for StaggerInfo provenance metadata."""

from __future__ import annotations

import copy
from types import MappingProxyType

import numpy as np
import pytest

from pypic.containers import StaggerInfo
from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.units import Normalization


class TestStaggerInfoConstruction:
    """StaggerInfo dataclass basics."""

    def test_minimal(self):
        si = StaggerInfo(convention="node")
        assert si.convention == "node"
        assert si.field_locations is None
        assert si.interpolation_order is None
        assert si.notes is None

    def test_full(self):
        si = StaggerInfo(
            convention="staggered",
            field_locations={"B": "face", "E": "edge"},
            interpolation_order=1,
            notes="Yee mesh",
        )
        assert si.convention == "staggered"
        assert si.field_locations is not None
        assert si.field_locations["B"] == "face"
        assert si.interpolation_order == 1
        assert si.notes == "Yee mesh"

    def test_field_locations_frozen_to_mapping_proxy(self):
        si = StaggerInfo(
            convention="staggered",
            field_locations={"B": "face"},
        )
        assert isinstance(si.field_locations, MappingProxyType)

    def test_frozen(self):
        si = StaggerInfo(convention="cell")
        with pytest.raises(AttributeError):
            si.convention = "node"  # type: ignore[misc]

    def test_copy_replace(self):
        si = StaggerInfo(convention="node")
        si2 = copy.replace(si, convention="cell")
        assert si2.convention == "cell"
        assert si.convention == "node"


class TestStaggerInFieldDataset:
    """StaggerInfo round-trips through FieldDataset.metadata."""

    def test_stagger_in_metadata(self):
        grid = GridInfo(dimensions=(4, 4), spacing=(1.0, 1.0))
        norm = Normalization.identity()
        stagger = StaggerInfo(convention="node")
        fds = FieldDataset.from_arrays(
            {"B1": np.ones((4, 4))},
            grid,
            norm,
            metadata={"stagger": stagger},
        )
        assert fds.metadata["stagger"] is stagger
        assert fds.metadata["stagger"].convention == "node"

    def test_stagger_survives_selection(self):
        """Stagger metadata propagates through isel."""
        grid = GridInfo(dimensions=(4, 4, 4), spacing=(1.0, 1.0, 1.0))
        norm = Normalization.identity()
        stagger = StaggerInfo(convention="cell")
        fds = FieldDataset.from_arrays(
            {"B1": np.ones((4, 4, 4))},
            grid,
            norm,
            metadata={"stagger": stagger},
        )
        sliced = fds.isel(z=2)
        assert sliced.metadata["stagger"].convention == "cell"
