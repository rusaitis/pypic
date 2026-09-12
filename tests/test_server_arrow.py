"""Tests for ``pypic.server.arrow``: FieldDataset ↔ Arrow IPC bytes."""

from __future__ import annotations

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")

from pypic.coordinates import CARTESIAN  # noqa: E402
from pypic.dataset import FieldDataset  # noqa: E402
from pypic.exceptions import UnknownFieldError  # noqa: E402
from pypic.grid import GridInfo  # noqa: E402
from pypic.reductions import reduce  # noqa: E402
from pypic.server.arrow import (  # noqa: E402
    decode_field_dataset_ipc,
    field_dataset_to_arrow_ipc,
)
from pypic.units import Normalization  # noqa: E402


@pytest.fixture
def small_cartesian_3d() -> FieldDataset:
    """4x3x2 Cartesian dataset — deliberately smaller and
    differently seeded than the shared ``cartesian_3d`` in
    conftest, to keep Arrow payload sizes assertable."""
    grid = GridInfo(
        dimensions=(4, 3, 2),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        geometry=CARTESIAN,
    )
    rng = np.random.default_rng(7)
    fields = {
        "B_1": rng.standard_normal((4, 3, 2)),
        "B_2": rng.standard_normal((4, 3, 2)),
        "B_3": rng.standard_normal((4, 3, 2)),
    }
    return FieldDataset.from_arrays(fields, grid, Normalization.identity())


def test_round_trip_preserves_values(small_cartesian_3d: FieldDataset) -> None:
    ipc = field_dataset_to_arrow_ipc(small_cartesian_3d)
    decoded = decode_field_dataset_ipc(ipc)
    np.testing.assert_array_equal(decoded["fields"]["B_1"], small_cartesian_3d["B_1"])
    np.testing.assert_array_equal(decoded["fields"]["B_2"], small_cartesian_3d["B_2"])
    np.testing.assert_array_equal(decoded["fields"]["B_3"], small_cartesian_3d["B_3"])


def test_round_trip_preserves_shape(small_cartesian_3d: FieldDataset) -> None:
    ipc = field_dataset_to_arrow_ipc(small_cartesian_3d)
    decoded = decode_field_dataset_ipc(ipc)
    assert decoded["fields"]["B_1"].shape == small_cartesian_3d["B_1"].shape


def test_schema_metadata_carries_dims_and_shape(
    small_cartesian_3d: FieldDataset,
) -> None:
    ipc = field_dataset_to_arrow_ipc(small_cartesian_3d)
    meta = decode_field_dataset_ipc(ipc)["metadata"]
    assert meta["shape"] == [4, 3, 2]
    assert meta["dims"] == ["x", "y", "z"]


def test_schema_metadata_carries_normalization(
    small_cartesian_3d: FieldDataset,
) -> None:
    ipc = field_dataset_to_arrow_ipc(small_cartesian_3d)
    meta = decode_field_dataset_ipc(ipc)["metadata"]
    norm = meta["normalization"]
    assert "length_ref" in norm
    assert "b_field_ref" in norm
    # Identity normalization → all refs equal 1.0
    assert norm["length_ref"] == 1.0
    assert norm["b_field_ref"] == 1.0


def test_schema_metadata_carries_field_attrs(small_cartesian_3d: FieldDataset) -> None:
    ipc = field_dataset_to_arrow_ipc(small_cartesian_3d)
    meta = decode_field_dataset_ipc(ipc)["metadata"]
    b1_attrs = meta["fields"]["B_1"]
    assert b1_attrs["quantity_type"] == "b_field"
    assert b1_attrs["si_unit"] == "T"


def test_coord_arrays_round_trip(small_cartesian_3d: FieldDataset) -> None:
    ipc = field_dataset_to_arrow_ipc(small_cartesian_3d)
    decoded = decode_field_dataset_ipc(ipc)
    assert set(decoded["coords"]) == {"x", "y", "z"}
    assert decoded["coords"]["x"].shape == (4,)
    assert decoded["coords"]["z"].shape == (2,)


def test_fields_subset(small_cartesian_3d: FieldDataset) -> None:
    ipc = field_dataset_to_arrow_ipc(small_cartesian_3d, fields=["B_1"])
    decoded = decode_field_dataset_ipc(ipc)
    assert set(decoded["fields"]) == {"B_1"}


def test_fields_alias_resolves(small_cartesian_3d: FieldDataset) -> None:
    # "Bx" is an alias for "B_1" on Cartesian grids — the encoder
    # should resolve the alias and emit a column under "B_1".
    ipc = field_dataset_to_arrow_ipc(small_cartesian_3d, fields=["Bx"])
    decoded = decode_field_dataset_ipc(ipc)
    assert "B_1" in decoded["fields"]


def test_unknown_field_raises_keyerror(small_cartesian_3d: FieldDataset) -> None:
    with pytest.raises(KeyError, match="nope"):
        field_dataset_to_arrow_ipc(small_cartesian_3d, fields=["B_1", "nope"])


def test_nan_preserved(small_cartesian_3d: FieldDataset) -> None:
    grid = small_cartesian_3d.grid
    f = np.full((4, 3, 2), np.nan)
    f[0, 0, 0] = 1.5
    ds = FieldDataset.from_arrays({"B_1": f}, grid, Normalization.identity())
    ipc = field_dataset_to_arrow_ipc(ds)
    decoded = decode_field_dataset_ipc(ipc)
    assert decoded["fields"]["B_1"][0, 0, 0] == 1.5
    assert np.isnan(decoded["fields"]["B_1"][1, 1, 1])


def test_units_si_applies_normalization() -> None:
    norm = Normalization.pic_electron(1e18)
    grid = GridInfo(
        dimensions=(2, 2, 2),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        geometry=CARTESIAN,
    )
    ds = FieldDataset.from_arrays({"B_1": np.full((2, 2, 2), 1.0)}, grid, norm)
    ipc_code = field_dataset_to_arrow_ipc(ds, units="code")
    ipc_si = field_dataset_to_arrow_ipc(ds, units="si")
    code = decode_field_dataset_ipc(ipc_code)
    si = decode_field_dataset_ipc(ipc_si)
    np.testing.assert_array_equal(code["fields"]["B_1"], 1.0)
    np.testing.assert_array_equal(si["fields"]["B_1"], norm.b_field_ref)
    assert si["metadata"]["units"] == "si"
    assert code["metadata"]["units"] == "code"


def test_units_invalid_raises(small_cartesian_3d: FieldDataset) -> None:
    with pytest.raises(ValueError, match="units"):
        field_dataset_to_arrow_ipc(small_cartesian_3d, units="bogus")


def test_reduction_provenance_attr_round_trips(
    small_cartesian_3d: FieldDataset,
) -> None:
    # After reduce(integrate), each field carries reduction provenance
    # — must travel through the Arrow schema metadata intact.
    column = reduce(small_cartesian_3d, "z", reduction="integrate")
    ipc = field_dataset_to_arrow_ipc(column)
    meta = decode_field_dataset_ipc(ipc)["metadata"]
    red = meta["fields"]["B_1"]["reduction"]
    assert red["op"] == "integrate"
    assert red["axis"] == "z"
    assert red["length_axes"] == 1


def test_reduced_dataset_2d_round_trip(small_cartesian_3d: FieldDataset) -> None:
    # A reduced FieldDataset has a 2-D shape — the encoder must follow
    # the surviving dims, not the original 3-D shape.
    column = reduce(small_cartesian_3d, "z", reduction="mean")
    ipc = field_dataset_to_arrow_ipc(column)
    decoded = decode_field_dataset_ipc(ipc)
    assert decoded["metadata"]["dims"] == ["x", "y"]
    assert decoded["metadata"]["shape"] == [4, 3]
    assert decoded["fields"]["B_1"].shape == (4, 3)
    np.testing.assert_array_equal(decoded["fields"]["B_1"], column["B_1"])


def test_decode_rejects_non_pypic_stream() -> None:
    # An Arrow IPC stream without the b"pypic" schema metadata key
    # cannot be decoded by us — surface a clear error.
    schema = pa.schema([pa.field("x", pa.float64())])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema) as writer:
        writer.write_batch(
            pa.RecordBatch.from_arrays([pa.array([1.0, 2.0])], schema=schema)
        )
    with pytest.raises(ValueError, match="missing pypic schema metadata"):
        decode_field_dataset_ipc(bytes(sink.getvalue().to_pybytes()))


class TestUnknownFieldRouting:
    """Unknown names must keep their typed identity through the encoder.

    ``_resolve_fields`` caught the ``UnknownFieldError`` that
    ``resolve_key`` raises and re-raised a bare ``KeyError``, discarding
    the 404 / ``"unknown_field"`` routing at the one boundary that
    consumes it — the request surfaced to a WebSocket client as
    ``kind="internal"`` and over HTTP as a 500.
    """

    def test_raises_typed_error(self, small_cartesian_3d) -> None:
        with pytest.raises(UnknownFieldError):
            field_dataset_to_arrow_ipc(small_cartesian_3d, fields=["not_a_field"])

    def test_still_catchable_as_keyerror(self, small_cartesian_3d) -> None:
        """Behaviour-preserving: the typed class subclasses ``KeyError``."""
        with pytest.raises(KeyError):
            field_dataset_to_arrow_ipc(small_cartesian_3d, fields=["not_a_field"])

    def test_routes_to_the_unknown_field_wire_kind(self, small_cartesian_3d) -> None:
        from pypic.server.exceptions import error_routing

        with pytest.raises(UnknownFieldError) as excinfo:
            field_dataset_to_arrow_ipc(small_cartesian_3d, fields=["not_a_field"])
        assert error_routing(excinfo.value) == ("unknown_field", 404)
