"""Tests for the OpenGGCM reader (grid, field I/O, full reader)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pypic.readers.openggcm import (
    OpenGGCMGrid,
    open_openggcm,
    parse_grid_file,
)
from pypic.readers.openggcm._field_map import (
    FIELD_NAME_MAP,
    bfield_to_si,
    density_to_mass_density_si,
    density_to_si,
    pressure_to_si,
    velocity_to_si,
)

FIXTURE_DIR = Path(__file__).parent / "data" / "openggcm-small"
_HAS_DATA = FIXTURE_DIR.exists() and (FIXTURE_DIR / "grid.gc012.dat").exists()


@pytest.fixture(scope="session")
def all_fields():
    """Read 8 fields from the small .3df fixture."""
    if not _HAS_DATA:
        pytest.skip("Fixture data not available")
    from pypic.readers.openggcm._field_io import read_3df_file

    return read_3df_file(
        FIXTURE_DIR / "gc012.3df.006300",
        skip={"eflx", "efly", "eflz"},
    )


@pytest.fixture(scope="session")
def reader_cfg_ds():
    """Create reader and read timestep 6300 once for all tests."""
    if not _HAS_DATA:
        pytest.skip("Fixture data not available")
    reader, cfg = open_openggcm(FIXTURE_DIR)
    ds = reader.read_timestep(FIXTURE_DIR, 6300)
    return reader, cfg, ds


class TestUnitConversions:
    """Unit conversion functions produce correct SI values."""

    def test_velocity_to_si(self) -> None:
        v = np.array([500.0])
        np.testing.assert_allclose(velocity_to_si(v), [500e3])

    def test_bfield_to_si(self) -> None:
        b = np.array([10.0])
        np.testing.assert_allclose(bfield_to_si(b), [10e-9])

    def test_density_to_si(self) -> None:
        n = np.array([5.0])
        np.testing.assert_allclose(density_to_si(n), [5e6])

    def test_density_to_mass_density_si(self) -> None:
        from scipy import constants

        n = np.array([1.0])
        expected = 1e6 * constants.m_p
        np.testing.assert_allclose(density_to_mass_density_si(n), [expected])

    def test_pressure_to_si(self) -> None:
        p = np.array([1000.0])
        np.testing.assert_allclose(pressure_to_si(p), [1e-9])


class TestFieldNameMap:
    """Field name mapping covers all expected fields."""

    def test_all_velocity_components(self) -> None:
        assert FIELD_NAME_MAP["vx"] == "V1"
        assert FIELD_NAME_MAP["vy"] == "V2"
        assert FIELD_NAME_MAP["vz"] == "V3"

    def test_all_bfield_components(self) -> None:
        assert FIELD_NAME_MAP["bx1"] == "B1"
        assert FIELD_NAME_MAP["by1"] == "B2"
        assert FIELD_NAME_MAP["bz1"] == "B3"

    def test_density_and_pressure(self) -> None:
        assert FIELD_NAME_MAP["rr"] == "rho_m"
        assert FIELD_NAME_MAP["pp"] == "P"


@pytest.mark.skipif(not _HAS_DATA, reason="Fixture data not available")
class TestGridParser:
    """Tests for parsing the OpenGGCM grid file."""

    @pytest.fixture
    def grid(self) -> OpenGGCMGrid:
        return parse_grid_file(FIXTURE_DIR / "grid.gc012.dat")

    def test_dimensions(self, grid: OpenGGCMGrid) -> None:
        assert grid.nx == 29
        assert grid.ny == 16
        assert grid.nz == 16

    def test_x_range(self, grid: OpenGGCMGrid) -> None:
        assert grid.x[0] == pytest.approx(-22.1, abs=0.01)
        assert grid.x[-1] == pytest.approx(300.1, abs=0.1)

    def test_yz_symmetric(self, grid: OpenGGCMGrid) -> None:
        np.testing.assert_allclose(grid.y, grid.z, rtol=1e-10)
        np.testing.assert_allclose(grid.y[0], -grid.y[-1], rtol=1e-6)

    def test_x_non_uniform(self, grid: OpenGGCMGrid) -> None:
        dx = np.diff(grid.x)
        assert dx.min() < 3.0
        assert dx.max() > 50.0

    def test_stagger_grids_present(self, grid: OpenGGCMGrid) -> None:
        for comp in ("bx", "by", "bz", "ex", "ey", "ez"):
            assert comp in grid.stagger
            gx, gy, gz = grid.stagger[comp]
            assert len(gx) == grid.nx
            assert len(gy) == grid.ny
            assert len(gz) == grid.nz

    def test_metadata(self, grid: OpenGGCMGrid) -> None:
        assert "DIPOLETIME" in grid.metadata
        assert "BASETIME" in grid.metadata


@pytest.mark.skipif(not _HAS_DATA, reason="Fixture data not available")
class TestFieldIO:
    """Tests for reading .3df field files."""

    def test_read_3df_fields(self, all_fields) -> None:
        fields, ts, nx, ny, nz = all_fields
        assert ts == 6300
        assert nx == 29
        assert ny == 16
        assert nz == 16
        assert len(fields) == 8  # 11 - 3 skipped
        for name in ("vx", "vy", "vz", "rr", "pp", "bx1", "by1", "bz1"):
            assert name in fields
            assert fields[name].shape == (29, 16, 16)

    def test_density_positive(self, all_fields) -> None:
        fields, *_ = all_fields
        assert fields["rr"].min() >= 0.0

    def test_bfield_reasonable_range(self, all_fields) -> None:
        fields, *_ = all_fields
        b_mag = np.sqrt(fields["bx1"] ** 2 + fields["by1"] ** 2 + fields["bz1"] ** 2)
        assert b_mag.max() < 1e6  # sanity check: < 1 mT in nT


@pytest.mark.skipif(not _HAS_DATA, reason="Fixture data not available")
class TestOpenGGCMReader:
    """Integration tests for the full reader pipeline."""

    def test_open_openggcm(self, reader_cfg_ds) -> None:
        _reader, cfg, _ds = reader_cfg_ds
        assert cfg.model_name == "OpenGGCM"
        assert cfg.model_type == "MHD"
        assert cfg.frame == "GSM"

    def test_available_timesteps(self, reader_cfg_ds) -> None:
        reader, _cfg, _ds = reader_cfg_ds
        steps = reader.available_timesteps(FIXTURE_DIR)
        assert 6300 in steps
        assert steps == sorted(steps)

    def test_read_timestep(self, reader_cfg_ds) -> None:
        _reader, _cfg, ds = reader_cfg_ds

        names = ds.field_names()
        for expected in ("B1", "B2", "B3", "V1", "V2", "V3", "rho_m", "P", "n_s0"):
            assert expected in names, f"Missing field: {expected}"

        assert ds["B1"].shape == (29, 16, 16)

        # B-field should be in SI (Tesla), range ~1e-9 to 1e-4
        b1 = ds["B1"]
        assert np.abs(b1).max() < 1e-3  # < 1 mT

        # Velocity in SI (m/s), range ~1e4 to 1e6
        v1 = ds["V1"]
        assert np.abs(v1).max() < 1e7  # < 10,000 km/s

    def test_cartesian_aliases(self, reader_cfg_ds) -> None:
        _reader, _cfg, ds = reader_cfg_ds
        assert ds.has_field("Bx")
        assert ds.has_field("By")
        assert ds.has_field("Bz")
        np.testing.assert_array_equal(ds["Bx"], ds["B1"])

    def test_non_uniform_coordinates(self, reader_cfg_ds) -> None:
        _reader, _cfg, ds = reader_cfg_ds
        x = ds.xr.coords["x"].values
        assert len(x) == 29
        dx = np.diff(x)
        assert not np.allclose(dx, dx[0])

    def test_missing_timestep_raises(self, reader_cfg_ds) -> None:
        reader, _cfg, _ds = reader_cfg_ds
        with pytest.raises(FileNotFoundError):
            reader.read_timestep(FIXTURE_DIR, 999999)
