"""Tests for the TabularData container."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pypic.readers.base import TabularData


class TestConstruction:
    def test_basic_construction(self) -> None:
        tab = TabularData(
            name="test",
            columns={
                "cycle": np.array([0.0, 1.0, 2.0]),
                "energy": np.array([5.0, 4.9, 4.8]),
            },
        )
        assert tab.name == "test"
        assert len(tab) == 3

    def test_empty_columns(self) -> None:
        tab = TabularData(name="empty", columns={})
        assert len(tab) == 0

    def test_with_index_column(self) -> None:
        tab = TabularData(
            name="indexed",
            columns={"t": np.array([0.0, 0.1]), "v": np.array([1.0, 2.0])},
            index_column="t",
        )
        assert tab.index_column == "t"

    def test_with_metadata(self) -> None:
        tab = TabularData(
            name="meta",
            columns={"x": np.array([1.0])},
            metadata={"source": "test"},
        )
        assert tab.metadata["source"] == "test"

    def test_default_metadata_empty(self) -> None:
        tab = TabularData(
            name="no_meta",
            columns={"x": np.array([1.0])},
        )
        assert tab.metadata == {}


class TestValidation:
    def test_unequal_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            TabularData(
                name="bad",
                columns={
                    "a": np.array([1.0, 2.0]),
                    "b": np.array([1.0]),
                },
            )

    def test_missing_index_column_raises(self) -> None:
        with pytest.raises(ValueError, match="index_column"):
            TabularData(
                name="bad",
                columns={"a": np.array([1.0])},
                index_column="missing",
            )


class TestGetitem:
    def test_get_existing_column(self) -> None:
        tab = TabularData(
            name="test",
            columns={"x": np.array([1.0, 2.0, 3.0])},
        )
        assert_allclose(tab["x"], [1.0, 2.0, 3.0])

    def test_missing_column_raises_keyerror(self) -> None:
        tab = TabularData(
            name="test",
            columns={"x": np.array([1.0])},
        )
        with pytest.raises(KeyError, match="missing"):
            tab["missing"]


class TestContains:
    def test_existing_column(self) -> None:
        tab = TabularData(
            name="test",
            columns={"a": np.array([1.0]), "b": np.array([2.0])},
        )
        assert "a" in tab
        assert "b" in tab

    def test_missing_column(self) -> None:
        tab = TabularData(
            name="test",
            columns={"a": np.array([1.0])},
        )
        assert "z" not in tab


class TestLen:
    def test_nonempty(self) -> None:
        tab = TabularData(
            name="test",
            columns={"x": np.array([1.0, 2.0, 3.0])},
        )
        assert len(tab) == 3

    def test_empty(self) -> None:
        tab = TabularData(name="test", columns={})
        assert len(tab) == 0


class TestColumnNames:
    def test_sorted(self) -> None:
        tab = TabularData(
            name="test",
            columns={
                "z": np.array([1.0]),
                "a": np.array([2.0]),
                "m": np.array([3.0]),
            },
        )
        assert tab.column_names == ["a", "m", "z"]


class TestIndex:
    def test_with_index_column(self) -> None:
        tab = TabularData(
            name="test",
            columns={
                "cycle": np.array([0.0, 10.0, 20.0]),
                "energy": np.array([5.0, 4.9, 4.8]),
            },
            index_column="cycle",
        )
        assert_allclose(tab.index, [0.0, 10.0, 20.0])

    def test_without_index_column(self) -> None:
        tab = TabularData(
            name="test",
            columns={"x": np.array([1.0, 2.0, 3.0])},
        )
        assert_allclose(tab.index, [0.0, 1.0, 2.0])

    def test_empty_index(self) -> None:
        tab = TabularData(name="test", columns={})
        assert len(tab.index) == 0


class TestFrozen:
    def test_cannot_set_attribute(self) -> None:
        tab = TabularData(
            name="test",
            columns={"x": np.array([1.0])},
        )
        with pytest.raises(AttributeError):
            tab.name = "new_name"  # type: ignore[misc]
