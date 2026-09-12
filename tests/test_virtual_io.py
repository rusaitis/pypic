"""Tests for pypic.io.open_virtual (VirtualiZarr HDF5 views)."""

from __future__ import annotations

import numpy as np
import pytest

virtualizarr = pytest.importorskip("virtualizarr")
pytest.importorskip("icechunk")

import h5py  # noqa: E402

from pypic.containers import SimulationConfig  # noqa: E402
from pypic.grid import GridInfo  # noqa: E402
from pypic.io import from_zarr, open_virtual, to_icechunk_virtual  # noqa: E402
from pypic.io._icechunk import open_icechunk_repo  # noqa: E402
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
        fields = {"B_1": np.ones((4, 3, 2)), "B_2": np.full((4, 3, 2), 0.5)}
        h5path = tmp_path / "test.h5"
        _write_canonical_h5(h5path, fields, grid)

        fds = open_virtual(h5path)

        assert sorted(fds.field_names()) == ["B_1", "B_2"]
        np.testing.assert_array_equal(fds["B_1"], 1.0)
        np.testing.assert_array_equal(fds["B_2"], 0.5)

    def test_grid_metadata_preserved(self, tmp_path):
        grid = GridInfo(
            dimensions=(4, 3, 2),
            spacing=(0.5, 0.5, 0.5),
            origin=(1.0, 2.0, 3.0),
            dt=0.01,
        )
        h5path = tmp_path / "grid.h5"
        _write_canonical_h5(h5path, {"B_1": np.ones((4, 3, 2))}, grid)

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
            {"B_1": np.ones((4, 3, 2))},
            grid,
            normalization=norm,
        )

        fds = open_virtual(h5path)

        assert fds.normalization.density_ref == norm.density_ref
        assert fds.normalization.length_ref == norm.length_ref

    def test_identity_normalization_when_missing(self, tmp_path):
        grid = make_uniform_grid(4, 3, 2)
        h5path = tmp_path / "no_norm.h5"
        _write_canonical_h5(h5path, {"B_1": np.ones((4, 3, 2))}, grid)

        fds = open_virtual(h5path)

        assert fds.normalization.is_identity

    def test_no_grid_raises(self, tmp_path):
        h5path = tmp_path / "no_grid.h5"
        with h5py.File(h5path, "w") as f:
            g = f.create_group("fields")
            g.create_dataset("B_1", data=np.ones((4, 3, 2)))

        with pytest.raises(ValueError, match="No 'grid' group"):
            open_virtual(h5path)

    def test_explicit_config_overrides_h5(self, tmp_path):
        grid_h5 = make_uniform_grid(4, 3, 2, spacing=1.0)
        h5path = tmp_path / "config.h5"
        _write_canonical_h5(h5path, {"B_1": np.ones((4, 3, 2))}, grid_h5)

        grid_cfg = make_uniform_grid(4, 3, 2, spacing=2.0)
        config = SimulationConfig(
            model_name="test",
            model_type="PIC",
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
            {"B_1": np.ones((4, 3, 2)), "B_2": np.ones((4, 3, 2))},
            grid,
        )

        fds = open_virtual(h5path, drop_variables=["B_2"])

        assert fds.has_field("B_1")
        assert not fds.has_field("B_2")

    def test_root_group(self, tmp_path):
        h5path = tmp_path / "root.h5"
        with h5py.File(h5path, "w") as f:
            f.create_dataset("B_1", data=np.ones((4, 3)))
            grd = f.create_group("grid")
            grd.attrs["dimensions"] = [4, 3]
            grd.attrs["spacing"] = [1.0, 1.0]
            grd.attrs["origin"] = [0.0, 0.0]

        # fields_group=None reads from root; HDF5 groups (like "grid")
        # are not datasets, so VirtualiZarr ignores them automatically.
        fds = open_virtual(h5path, fields_group=None)

        assert fds.has_field("B_1")

    def test_constant_field_round_trip(self, tmp_path):
        # Regression: the old Kerchunk fallback returned the fill value
        # (typically 0) when every cell of an HDF5 dataset equalled the
        # fill value, since Kerchunk encoded the chunk as "all-fill" and
        # zarr v2 read it back as the default. The Icechunk-backed path
        # preserves the actual value.
        grid = make_uniform_grid(4, 3, 2)
        const = 0.0  # the fill-value-collision case
        h5path = tmp_path / "const.h5"
        _write_canonical_h5(h5path, {"B_1": np.full((4, 3, 2), const)}, grid)

        fds = open_virtual(h5path)

        np.testing.assert_array_equal(np.asarray(fds["B_1"]), const)

    def test_byte_string_attrs_decoded(self, tmp_path):
        # HDF5 writers commonly store string attrs as bytes (h5py's
        # default for variable-length UTF-8).  str(b"cartesian") gives
        # "b'cartesian'" — use an explicit decode so metadata reads
        # stay faithful across writers.
        grid = make_uniform_grid(4, 3, 2)
        h5path = tmp_path / "bytes_attrs.h5"
        with h5py.File(h5path, "w") as f:
            f.create_group("fields").create_dataset("B_1", data=np.ones((4, 3, 2)))
            g = f.create_group("grid")
            g.attrs["dimensions"] = list(grid.dimensions)
            g.attrs["spacing"] = list(grid.spacing)
            g.attrs["origin"] = list(grid.origin)
            g.attrs["geometry"] = b"cartesian"
            g.attrs["boundary"] = np.array(
                [b"periodic", b"periodic", b"periodic"], dtype="S8"
            )
            f.attrs["model"] = b"iPIC3D"

        fds = open_virtual(h5path)

        assert fds.grid.boundary == ("periodic", "periodic", "periodic")
        # Geometry byte-decoded back to the canonical cartesian entry.
        assert fds.grid.geometry.type.value == "cartesian"
        assert fds.metadata["model"] == "iPIC3D"

    def test_coordinate_arrays_match_grid(self, tmp_path):
        grid = GridInfo(
            dimensions=(4, 3, 2),
            spacing=(0.5, 1.0, 2.0),
            origin=(0.0, 0.0, 0.0),
        )
        h5path = tmp_path / "coords.h5"
        _write_canonical_h5(h5path, {"B_1": np.ones((4, 3, 2))}, grid)

        fds = open_virtual(h5path)

        np.testing.assert_array_equal(
            fds.xr.coords["x"].values, [0.25, 0.75, 1.25, 1.75]
        )
        np.testing.assert_array_equal(fds.xr.coords["y"].values, [0.5, 1.5, 2.5])
        np.testing.assert_array_equal(fds.xr.coords["z"].values, [1.0, 3.0])


class TestToIcechunkVirtualVersioning:
    """Reviewer regression: to_icechunk_virtual must support more than
    one commit per repo (re-commits to the same branch, and forks from
    a populated main).  The first version wrote virtual refs at root
    without overwriting, hitting ContainsGroupError on any write that
    wasn't the very first.
    """

    def test_second_commit_to_same_branch(self, tmp_path):
        grid = make_uniform_grid(4, 3, 2)
        h5a = tmp_path / "a.h5"
        h5b = tmp_path / "b.h5"
        _write_canonical_h5(h5a, {"B_1": np.ones((4, 3, 2))}, grid)
        _write_canonical_h5(h5b, {"B_1": np.full((4, 3, 2), 2.0)}, grid)

        output = tmp_path / "repo"
        snap_a = to_icechunk_virtual(h5a, output, message="first")
        snap_b = to_icechunk_virtual(h5b, output, message="second")

        assert snap_a != snap_b
        repo = open_icechunk_repo(output)
        ancestry = list(repo.ancestry(branch="main"))
        # initial root snapshot plus two pypic commits
        assert len(ancestry) >= 3

        loaded = from_zarr(output)
        np.testing.assert_array_equal(loaded["B_1"], 2.0)

    def test_new_branch_fork_from_populated_main(self, tmp_path):
        grid = make_uniform_grid(4, 3, 2)
        h5a = tmp_path / "a.h5"
        h5b = tmp_path / "b.h5"
        _write_canonical_h5(h5a, {"B_1": np.ones((4, 3, 2))}, grid)
        _write_canonical_h5(h5b, {"B_1": np.full((4, 3, 2), 3.0)}, grid)

        output = tmp_path / "repo"
        to_icechunk_virtual(h5a, output, branch="main", message="main write")
        # Forking a new branch off a populated main previously failed
        # because the fresh branch inherited main's root group.
        snap_alt = to_icechunk_virtual(h5b, output, branch="alt", message="alt write")
        assert isinstance(snap_alt, str)
        assert snap_alt

        loaded_main = from_zarr(output, branch="main")
        loaded_alt = from_zarr(output, branch="alt")
        np.testing.assert_array_equal(loaded_main["B_1"], 1.0)
        np.testing.assert_array_equal(loaded_alt["B_1"], 3.0)

    def test_sources_in_different_directories(self, tmp_path):
        # Reviewer regression: commits from different parent dirs must
        # both remain readable.  The first write registered a virtual
        # chunk container for source A's parent; the second write must
        # merge in source B's parent — otherwise refs to /b/b.h5 have
        # no container and from_zarr fails with "no virtual chunk
        # container can handle the chunk location".
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        grid = make_uniform_grid(4, 3, 2)
        h5a = dir_a / "a.h5"
        h5b = dir_b / "b.h5"
        _write_canonical_h5(h5a, {"B_1": np.ones((4, 3, 2))}, grid)
        _write_canonical_h5(h5b, {"B_1": np.full((4, 3, 2), 2.0)}, grid)

        output = tmp_path / "repo"
        snap_a = to_icechunk_virtual(h5a, output, message="a")
        snap_b = to_icechunk_virtual(h5b, output, message="b")

        # Tip reads current (b); snapshot A still resolves via the
        # merged container set.
        np.testing.assert_array_equal(from_zarr(output)["B_1"], 2.0)
        np.testing.assert_array_equal(from_zarr(output, snapshot_id=snap_a)["B_1"], 1.0)
        np.testing.assert_array_equal(from_zarr(output, snapshot_id=snap_b)["B_1"], 2.0)

    def test_numpy_scalar_root_attrs(self, tmp_path):
        # Reviewer regression: h5py returns scalar HDF5 attrs as
        # numpy types (float32, int64), which xarray's Zarr attr
        # validator rejects as "Invalid attribute in Dataset.attrs".
        # ``to_icechunk_virtual`` must normalize the metadata dict
        # before writing.
        grid = make_uniform_grid(4, 3, 2)
        h5path = tmp_path / "numpy_attrs.h5"
        with h5py.File(h5path, "w") as f:
            f.create_group("fields").create_dataset("B_1", data=np.ones((4, 3, 2)))
            g = f.create_group("grid")
            g.attrs["dimensions"] = list(grid.dimensions)
            g.attrs["spacing"] = list(grid.spacing)
            g.attrs["origin"] = list(grid.origin)
            g.attrs["geometry"] = "cartesian"
            # h5py stores these as numpy scalars on read-back
            f.attrs["time"] = np.float32(1.25)
            f.attrs["step"] = np.int64(42)

        output = tmp_path / "repo"
        snap = to_icechunk_virtual(h5path, output, message="numpy attrs")
        assert snap

        loaded = from_zarr(output)
        assert loaded.metadata["time"] == pytest.approx(1.25)
        assert loaded.metadata["step"] == 42

    def test_failed_validation_cleans_up_fresh_repo(self, tmp_path):
        # Reviewer regression: the virtual writer creates and seeds
        # the Icechunk repo before ``open_virtual`` runs the canonical
        # metadata validation.  An HDF5 file with ``fields/`` but no
        # ``grid/`` raises a ``ValueError`` from
        # ``open_virtual``/``_extract_grid``, and without cleanup the
        # repo subtree was left behind — ``is_icechunk_store`` would
        # report True while ``from_zarr`` raised ``GroupNotFoundError``.
        h5path = tmp_path / "no_grid.h5"
        with h5py.File(h5path, "w") as f:
            f.create_group("fields").create_dataset("B_1", data=np.ones((4, 3, 2)))
            # Intentionally omit the ``grid/`` group.

        output = tmp_path / "broken_virtual"
        with pytest.raises((ValueError, KeyError)):
            to_icechunk_virtual(h5path, output)
        from pypic.io._icechunk import is_icechunk_store

        assert not output.exists() or not any(output.iterdir())
        assert not is_icechunk_store(output)
