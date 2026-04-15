"""Tests for pypic.io Icechunk storage backend."""

from __future__ import annotations

import numpy as np
import pytest

icechunk = pytest.importorskip("icechunk")

from pypic.coordinates.transforms import FrameTransform  # noqa: E402
from pypic.dataset import FieldDataset  # noqa: E402
from pypic.grid import GridInfo  # noqa: E402
from pypic.io import (  # noqa: E402
    from_zarr,
    icechunk_ancestry,
    icechunk_create_tag,
    to_zarr,
    to_zarr_timeseries,
)
from pypic.io._icechunk import is_icechunk_store  # noqa: E402
from pypic.units import Normalization, PhysicsParams  # noqa: E402
from tests._helpers import (  # noqa: E402
    ELECTRONS,
    IONS,
    make_test_dataset,
    make_uniform_grid,
)


class TestToZarrIcechunk:
    """Round-trip tests for to_zarr / from_zarr with Icechunk backend."""

    def test_round_trip_basic(self, tmp_path):
        fds = make_test_dataset(
            {"B1": np.ones((4, 3, 2)), "B2": np.zeros((4, 3, 2))},
        )
        store = tmp_path / "test.icechunk"
        to_zarr(fds, store, backend="icechunk")
        loaded = from_zarr(store)

        assert sorted(loaded.field_names()) == ["B1", "B2"]
        np.testing.assert_allclose(loaded["B1"], fds["B1"])
        np.testing.assert_allclose(loaded["B2"], fds["B2"])
        assert loaded.grid.dimensions == fds.grid.dimensions
        assert loaded.normalization.is_identity

    def test_round_trip_full_metadata(self, tmp_path):
        rot = (
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        transform = FrameTransform(
            "simulation",
            "GSM",
            origin=(10.0, 5.0, 3.0),
            rotation=rot,
            scale=0.5,
        )
        physics = PhysicsParams(
            gamma=1.4,
            c=1.0,
            relativistic=True,
            extra={"theta": 0.5},
        )
        grid = GridInfo(
            dimensions=(4, 3, 2),
            spacing=(0.5, 0.5, 0.5),
            origin=(1.0, 2.0, 3.0),
            dt=0.01,
            boundary=("periodic", "open", "periodic"),
        )
        fds = FieldDataset.from_arrays(
            {"B1": np.ones((4, 3, 2)), "rho_m": np.full((4, 3, 2), 2.0)},
            grid,
            Normalization.pic_electron(1e18),
            species=[ELECTRONS, IONS],
            physics=physics,
            metadata={"run_name": "test_run"},
            frame="GSM",
            transforms={"GSM": transform},
        )
        store = tmp_path / "full.icechunk"
        to_zarr(fds, store, backend="icechunk")
        loaded = from_zarr(store)

        assert loaded.grid.dt == 0.01
        assert loaded.grid.boundary == ("periodic", "open", "periodic")
        assert len(loaded.species) == 2
        assert loaded.species[0].name == "electrons"
        assert loaded.physics.gamma == 1.4
        assert loaded.physics.relativistic is True
        assert loaded.frame == "GSM"
        assert "GSM" in loaded.transforms
        np.testing.assert_allclose(loaded["B1"], 1.0)
        np.testing.assert_allclose(loaded["rho_m"], 2.0)

    def test_returns_snapshot_id(self, tmp_path):
        fds = make_test_dataset({"B1": np.ones((4, 3, 2))})
        store = tmp_path / "snap.icechunk"
        result = to_zarr(fds, store, backend="icechunk")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_custom_message(self, tmp_path):
        fds = make_test_dataset({"B1": np.ones((4, 3, 2))})
        store = tmp_path / "msg.icechunk"
        to_zarr(fds, store, backend="icechunk", message="initial data")
        history = icechunk_ancestry(store)
        messages = [h["message"] for h in history]
        assert "initial data" in messages

    def test_dtype_downcast(self, tmp_path):
        fds = make_test_dataset(
            {"B1": np.ones((4, 3, 2), dtype=np.float64)},
        )
        store = tmp_path / "f32.icechunk"
        to_zarr(fds, store, backend="icechunk", dtype="float32")
        loaded = from_zarr(store)
        assert loaded["B1"].dtype == np.float32
        np.testing.assert_allclose(loaded["B1"], 1.0, rtol=1e-6)

    def test_plain_zarr_returns_none(self, tmp_path):
        fds = make_test_dataset({"B1": np.ones((4, 3, 2))})
        store = tmp_path / "plain.zarr"
        result = to_zarr(fds, store)
        assert result is None

    def test_unknown_backend_raises(self, tmp_path):
        fds = make_test_dataset({"B1": np.ones((4, 3, 2))})
        store = tmp_path / "bad.zarr"
        with pytest.raises(ValueError, match="Unknown backend"):
            to_zarr(fds, store, backend="nosql")

    def test_failed_write_cleans_up_fresh_repo(self, tmp_path):
        # Reviewer regression: any error between
        # ``open_icechunk_repo(create=True)`` and ``session.commit``
        # used to leave a half-initialized repo behind — same stale-
        # store problem already fixed for the timeseries writer.  A
        # set is not JSON-serializable, so xarray's attr validator
        # raises during ``ds.to_zarr`` and the cleanup branch fires.
        grid = make_uniform_grid(4, 3, 2)
        fds = FieldDataset.from_arrays(
            {"B1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
            metadata={"bad": {1, 2, 3}},
        )
        store = tmp_path / "broken.icechunk"
        with pytest.raises(TypeError, match=r"Invalid attribute"):
            to_zarr(fds, store, backend="icechunk")
        assert not store.exists()
        assert not is_icechunk_store(store)

    def test_failed_write_preserves_existing_repo(self, tmp_path):
        # The cleanup must only fire on freshly-created repos: a repo
        # with prior successful commits stays intact even if a later
        # write attempt aborts before commit.
        good = make_test_dataset({"B1": np.full((4, 3, 2), 7.0)})
        store = tmp_path / "existing.icechunk"
        to_zarr(good, store, backend="icechunk")
        grid = make_uniform_grid(4, 3, 2)
        bad = FieldDataset.from_arrays(
            {"B1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
            metadata={"bad": {1, 2, 3}},
        )
        with pytest.raises(TypeError, match=r"Invalid attribute"):
            to_zarr(bad, store, backend="icechunk", branch="nightly")
        assert is_icechunk_store(store)
        loaded = from_zarr(store, branch="main")
        np.testing.assert_allclose(loaded["B1"], 7.0)


class TestFromZarrIcechunk:
    """Tests for from_zarr with Icechunk auto-detection and ref parameters."""

    def test_auto_detect(self, tmp_path):
        fds = make_test_dataset({"B1": np.ones((4, 3, 2))})
        store = tmp_path / "auto.icechunk"
        to_zarr(fds, store, backend="icechunk")
        loaded = from_zarr(store)
        np.testing.assert_allclose(loaded["B1"], 1.0)

    def test_read_by_tag(self, tmp_path):
        fds = make_test_dataset({"B1": np.ones((4, 3, 2))})
        store = tmp_path / "tag.icechunk"
        to_zarr(fds, store, backend="icechunk")
        icechunk_create_tag(store, "v1.0")

        loaded = from_zarr(store, tag="v1.0")
        np.testing.assert_allclose(loaded["B1"], 1.0)

    def test_read_by_snapshot(self, tmp_path):
        fds = make_test_dataset({"B1": np.ones((4, 3, 2))})
        store = tmp_path / "snap.icechunk"
        snap_id = to_zarr(fds, store, backend="icechunk")
        assert snap_id is not None

        loaded = from_zarr(store, snapshot_id=snap_id)
        np.testing.assert_allclose(loaded["B1"], 1.0)

    def test_read_by_branch(self, tmp_path):
        fds = make_test_dataset({"B1": np.ones((4, 3, 2))})
        store = tmp_path / "branch.icechunk"
        to_zarr(fds, store, backend="icechunk", branch="main")

        loaded = from_zarr(store, branch="main")
        np.testing.assert_allclose(loaded["B1"], 1.0)

    def test_write_creates_non_main_branch(self, tmp_path):
        # Fresh repos only have `main`; writable_session(other) would
        # otherwise raise `ref not found`.  The writer must fork the
        # branch off main's tip before opening the session.
        fds = make_test_dataset({"B1": np.full((4, 3, 2), 7.0)})
        store = tmp_path / "nonmain.icechunk"
        snap = to_zarr(fds, store, backend="icechunk", branch="analysis")
        assert snap is not None
        loaded = from_zarr(store, branch="analysis")
        np.testing.assert_allclose(loaded["B1"], 7.0)

    def test_multiple_refs_raises(self, tmp_path):
        fds = make_test_dataset({"B1": np.ones((4, 3, 2))})
        store = tmp_path / "multi.icechunk"
        to_zarr(fds, store, backend="icechunk")
        with pytest.raises(ValueError, match="at most one"):
            from_zarr(store, branch="main", tag="v1.0")


class TestTimeseriesIcechunk:
    """Tests for to_zarr_timeseries with Icechunk backend."""

    def test_timeseries_atomic_write(self, tmp_path):
        grid = make_uniform_grid(4, 3, 2)
        pairs = [
            (
                0.0,
                FieldDataset.from_arrays(
                    {"B1": np.full((4, 3, 2), 1.0)},
                    grid,
                    Normalization.identity(),
                ),
            ),
            (
                1.0,
                FieldDataset.from_arrays(
                    {"B1": np.full((4, 3, 2), 2.0)},
                    grid,
                    Normalization.identity(),
                ),
            ),
        ]
        store = tmp_path / "ts.icechunk"
        snap_id = to_zarr_timeseries(pairs, store, backend="icechunk")
        assert isinstance(snap_id, str)

        loaded = from_zarr(store)
        assert "time" in loaded.xr.dims
        assert loaded.xr.sizes["time"] == 2

    def test_timeseries_rejects_field_drift(self, tmp_path):
        grid = make_uniform_grid(4, 3, 2)
        step0 = FieldDataset.from_arrays(
            {"B1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
        )
        step1 = FieldDataset.from_arrays(
            {"B1": np.ones((4, 3, 2)), "B2": np.zeros((4, 3, 2))},
            grid,
            Normalization.identity(),
        )
        store = tmp_path / "drift.icechunk"
        with pytest.raises(ValueError, match=r"field set.*differs.*B2"):
            to_zarr_timeseries([(0.0, step0), (1.0, step1)], store, backend="icechunk")
        # The fresh repo must not survive a pre-commit failure —
        # ``open_icechunk_repo(create=True)`` persists an initial
        # snapshot before we know the source is usable, and without
        # cleanup ``is_icechunk_store`` would report True while
        # ``from_zarr`` raises GroupNotFoundError.
        assert not store.exists()

    def test_timeseries_empty_source_cleans_up_fresh_repo(self, tmp_path):
        # Reviewer regression: empty source now raises before any
        # commit, and the half-initialized repo directory is removed.
        store = tmp_path / "empty.icechunk"
        with pytest.raises(ValueError, match=r"No timesteps to write"):
            to_zarr_timeseries([], store, backend="icechunk")
        assert not store.exists()

    def test_timeseries_empty_source_cleans_up_preexisting_empty_dir(self, tmp_path):
        # Reviewer regression: a caller-supplied empty directory must
        # be treated like a missing one — ``open_icechunk_repo(create=
        # True)`` will seed a half-initialized repo into it, and an
        # empty source then aborts before commit.  Leaving the repo
        # subtree behind would leave ``is_icechunk_store`` returning
        # True while ``from_zarr`` raised ``GroupNotFoundError``.
        store = tmp_path / "preexisting_empty"
        store.mkdir()
        assert not any(store.iterdir())
        with pytest.raises(ValueError, match=r"No timesteps to write"):
            to_zarr_timeseries([], store, backend="icechunk")
        assert not store.exists() or not any(store.iterdir())
        assert not is_icechunk_store(store)

    def test_timeseries_shape_drift_cleans_up_fresh_repo(self, tmp_path):
        grid_small = make_uniform_grid(2, 2, 2)
        grid_big = make_uniform_grid(3, 2, 2)
        step0 = FieldDataset.from_arrays(
            {"B1": np.ones((2, 2, 2))},
            grid_small,
            Normalization.identity(),
        )
        step1 = FieldDataset.from_arrays(
            {"B1": np.ones((3, 2, 2))},
            grid_big,
            Normalization.identity(),
        )
        store = tmp_path / "partial.icechunk"
        with pytest.raises(ValueError, match=r"different dimension sizes"):
            to_zarr_timeseries([(0.0, step0), (1.0, step1)], store, backend="icechunk")
        assert not store.exists()

    def test_timeseries_preserves_existing_repo_on_failure(self, tmp_path):
        # A failing write on a repo that already had successful
        # commits must not destroy the pre-existing data — only
        # freshly-created repos are cleaned up.
        grid = make_uniform_grid(4, 3, 2)
        good = FieldDataset.from_arrays(
            {"B1": np.full((4, 3, 2), 5.0)},
            grid,
            Normalization.identity(),
        )
        store = tmp_path / "existing.icechunk"
        to_zarr_timeseries(
            [(0.0, good), (1.0, good)], store, backend="icechunk", branch="main"
        )
        # Attempt a failing write into the same repo on a fresh branch.
        with pytest.raises(ValueError, match=r"No timesteps to write"):
            to_zarr_timeseries([], store, backend="icechunk", branch="nightly")
        # Pre-existing data on main must still be readable.
        loaded = from_zarr(store, branch="main")
        np.testing.assert_allclose(loaded["B1"], 5.0)

    def test_timeseries_branch_created(self, tmp_path):
        grid = make_uniform_grid(4, 3, 2)
        fds = FieldDataset.from_arrays(
            {"B1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
        )
        store = tmp_path / "ts_branch.icechunk"
        snap = to_zarr_timeseries(
            [(0.0, fds), (1.0, fds)],
            store,
            backend="icechunk",
            branch="nightly",
        )
        assert snap is not None
        loaded = from_zarr(store, branch="nightly")
        assert loaded.xr.sizes["time"] == 2

    def test_timeseries_metadata_preserved(self, tmp_path):
        grid = make_uniform_grid(4, 3, 2)
        fds = FieldDataset.from_arrays(
            {"B1": np.ones((4, 3, 2))},
            grid,
            Normalization.identity(),
            species=[ELECTRONS],
            frame="sim",
        )
        store = tmp_path / "ts_meta.icechunk"
        to_zarr_timeseries(
            [(0.0, fds), (1.0, fds)],
            store,
            backend="icechunk",
        )
        loaded = from_zarr(store)
        assert loaded.frame == "sim"
        assert len(loaded.species) == 1
        assert loaded.species[0].name == "electrons"


class TestIcechunkHelpers:
    """Tests for repository helper functions."""

    def test_create_tag(self, tmp_path):
        fds = make_test_dataset({"B1": np.ones((4, 3, 2))})
        store = tmp_path / "tagged.icechunk"
        to_zarr(fds, store, backend="icechunk")
        icechunk_create_tag(store, "release-1")

        from pypic.io._icechunk import open_icechunk_repo

        repo = open_icechunk_repo(store)
        tags = list(repo.list_tags())
        assert "release-1" in tags

    def test_ancestry(self, tmp_path):
        fds = make_test_dataset({"B1": np.ones((4, 3, 2))})
        store = tmp_path / "hist.icechunk"
        to_zarr(fds, store, backend="icechunk", message="first")
        to_zarr(fds, store, backend="icechunk", message="second")

        history = icechunk_ancestry(store)
        messages = [h["message"] for h in history]
        assert "second" in messages
        assert "first" in messages

    def test_is_icechunk_store_positive(self, tmp_path):
        fds = make_test_dataset({"B1": np.ones((4, 3, 2))})
        store = tmp_path / "ic.icechunk"
        to_zarr(fds, store, backend="icechunk")
        assert is_icechunk_store(store) is True

    def test_is_icechunk_store_negative(self, tmp_path):
        assert is_icechunk_store(tmp_path / "nonexistent") is False

    def test_is_icechunk_store_plain_zarr(self, tmp_path):
        fds = make_test_dataset({"B1": np.ones((4, 3, 2))})
        store = tmp_path / "plain.zarr"
        to_zarr(fds, store)
        assert is_icechunk_store(store) is False
