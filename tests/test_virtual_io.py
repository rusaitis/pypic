"""Tests for pypic.io.open_virtual (VirtualiZarr HDF5 views)."""

from __future__ import annotations

import numpy as np
import pytest

virtualizarr = pytest.importorskip("virtualizarr")
pytest.importorskip("icechunk")

import h5py  # noqa: E402

from pypic.containers import SimulationConfig  # noqa: E402
from pypic.grid import GridInfo  # noqa: E402
from pypic.io import open_virtual  # noqa: E402
from pypic.units import Normalization, SpeciesInfo  # noqa: E402
from tests._helpers import make_uniform_grid  # noqa: E402


def _write_canonical_h5(
    path,
    fields,
    grid,
    *,
    normalization=None,
    step=None,
    time=None,
):
    """Write a pypic canonical HDF5 file for testing."""
    with h5py.File(path, "w") as f:
        g = f.create_group("fields")
        for name, arr in fields.items():
            g.create_dataset(name, data=arr)

        grd = f.create_group("grid")
        grd.attrs["dimensions"] = list(grid.dimensions)
        grd.attrs["spacing"] = list(grid.spacing)
        grd.attrs["origin"] = list(grid.origin)
        grd.attrs["geometry"] = grid.geometry.type.value

        if grid.dt is not None:
            grd.attrs["dt"] = grid.dt
        if grid.boundary is not None:
            grd.attrs["boundary"] = list(grid.boundary)

        if normalization is not None:
            n = f.create_group("normalization")
            n.attrs["length_ref"] = normalization.length_ref
            n.attrs["time_ref"] = normalization.time_ref
            n.attrs["velocity_ref"] = normalization.velocity_ref
            n.attrs["b_field_ref"] = normalization.b_field_ref
            n.attrs["e_field_ref"] = normalization.e_field_ref
            n.attrs["density_ref"] = normalization.density_ref
            n.attrs["mass_ref"] = normalization.mass_ref
            n.attrs["charge_ref"] = normalization.charge_ref

        if step is not None:
            f.attrs["step"] = step
        if time is not None:
            f.attrs["time"] = time


class TestOpenVirtual:
    """Tests for open_virtual with canonical HDF5 layout."""

    def test_basic_round_trip(self, tmp_path):
        grid = make_uniform_grid(4, 3, 2)
        fields = {"B1": np.ones((4, 3, 2)), "B2": np.full((4, 3, 2), 0.5)}
        h5path = tmp_path / "test.h5"
        _write_canonical_h5(h5path, fields, grid)

        fds = open_virtual(h5path)

        assert sorted(fds.field_names()) == ["B1", "B2"]
        np.testing.assert_allclose(fds["B1"], 1.0)
        np.testing.assert_allclose(fds["B2"], 0.5)

    def test_grid_metadata_preserved(self, tmp_path):
        grid = GridInfo(
            dimensions=(4, 3, 2),
            spacing=(0.5, 0.5, 0.5),
            origin=(1.0, 2.0, 3.0),
            dt=0.01,
        )
        h5path = tmp_path / "grid.h5"
        _write_canonical_h5(h5path, {"B1": np.ones((4, 3, 2))}, grid)

        fds = open_virtual(h5path)

        assert fds.grid.dimensions == (4, 3, 2)
        assert fds.grid.spacing == (0.5, 0.5, 0.5)
        assert fds.grid.origin == (1.0, 2.0, 3.0)
        assert fds.grid.dt == 0.01

    def test_normalization_preserved(self, tmp_path):
        grid = make_uniform_grid(4, 3, 2)
        norm = Normalization.pic_electron(1e18)
        h5path = tmp_path / "norm.h5"
        _write_canonical_h5(
            h5path,
            {"B1": np.ones((4, 3, 2))},
            grid,
            normalization=norm,
        )

        fds = open_virtual(h5path)

        assert fds.normalization.density_ref == norm.density_ref
        assert fds.normalization.length_ref == norm.length_ref

    def test_identity_normalization_when_missing(self, tmp_path):
        grid = make_uniform_grid(4, 3, 2)
        h5path = tmp_path / "no_norm.h5"
        _write_canonical_h5(h5path, {"B1": np.ones((4, 3, 2))}, grid)

        fds = open_virtual(h5path)

        assert fds.normalization.is_identity

    def test_no_grid_raises(self, tmp_path):
        h5path = tmp_path / "no_grid.h5"
        with h5py.File(h5path, "w") as f:
            g = f.create_group("fields")
            g.create_dataset("B1", data=np.ones((4, 3, 2)))

        with pytest.raises(ValueError, match="No 'grid' group"):
            open_virtual(h5path)

    def test_explicit_config_overrides_h5(self, tmp_path):
        grid_h5 = make_uniform_grid(4, 3, 2, spacing=1.0)
        h5path = tmp_path / "config.h5"
        _write_canonical_h5(h5path, {"B1": np.ones((4, 3, 2))}, grid_h5)

        grid_cfg = make_uniform_grid(4, 3, 2, spacing=2.0)
        config = SimulationConfig(
            model_name="test",
            model_type="pic",
            grid=grid_cfg,
            normalization=Normalization.identity(),
            species=(SpeciesInfo(name="e", charge=-1.0, mass=1.0),),
        )
        fds = open_virtual(h5path, config=config)

        assert fds.grid.spacing == (2.0, 2.0, 2.0)
        assert len(fds.species) == 1

    def test_drop_variables(self, tmp_path):
        grid = make_uniform_grid(4, 3, 2)
        h5path = tmp_path / "drop.h5"
        _write_canonical_h5(
            h5path,
            {"B1": np.ones((4, 3, 2)), "B2": np.ones((4, 3, 2))},
            grid,
        )

        fds = open_virtual(h5path, drop_variables=["B2"])

        assert fds.has_field("B1")
        assert not fds.has_field("B2")

    def test_root_group(self, tmp_path):
        h5path = tmp_path / "root.h5"
        with h5py.File(h5path, "w") as f:
            f.create_dataset("B1", data=np.ones((4, 3)))
            grd = f.create_group("grid")
            grd.attrs["dimensions"] = [4, 3]
            grd.attrs["spacing"] = [1.0, 1.0]
            grd.attrs["origin"] = [0.0, 0.0]

        # fields_group=None reads from root; HDF5 groups (like "grid")
        # are not datasets, so VirtualiZarr ignores them automatically.
        fds = open_virtual(h5path, fields_group=None)

        assert fds.has_field("B1")

    def test_constant_field_round_trip(self, tmp_path):
        # Regression: the old Kerchunk fallback returned the fill value
        # (typically 0) when every cell of an HDF5 dataset equalled the
        # fill value, since Kerchunk encoded the chunk as "all-fill" and
        # zarr v2 read it back as the default. The Icechunk-backed path
        # preserves the actual value.
        grid = make_uniform_grid(4, 3, 2)
        const = 0.0  # the fill-value-collision case
        h5path = tmp_path / "const.h5"
        _write_canonical_h5(h5path, {"B1": np.full((4, 3, 2), const)}, grid)

        fds = open_virtual(h5path)

        np.testing.assert_array_equal(np.asarray(fds["B1"]), const)

    def test_coordinate_arrays_match_grid(self, tmp_path):
        grid = GridInfo(
            dimensions=(4, 3, 2),
            spacing=(0.5, 1.0, 2.0),
            origin=(0.0, 0.0, 0.0),
        )
        h5path = tmp_path / "coords.h5"
        _write_canonical_h5(h5path, {"B1": np.ones((4, 3, 2))}, grid)

        fds = open_virtual(h5path)

        np.testing.assert_allclose(fds.xr.coords["x"].values, [0.25, 0.75, 1.25, 1.75])
        np.testing.assert_allclose(fds.xr.coords["y"].values, [0.5, 1.5, 2.5])
        np.testing.assert_allclose(fds.xr.coords["z"].values, [1.0, 3.0])
