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


@pytest.fixture(scope="session")
def all_fields():
    """Read 8 fields from the small .3df fixture."""
    from pypic.readers.openggcm._field_io import read_3df_file

    return read_3df_file(
        FIXTURE_DIR / "gc012.3df.006300",
        skip={"eflx", "efly", "eflz"},
    )


@pytest.fixture(scope="session")
def reader_cfg_ds():
    """Create reader and read timestep 6300 once for all tests."""
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
        assert FIELD_NAME_MAP["vx"] == "V_1"
        assert FIELD_NAME_MAP["vy"] == "V_2"
        assert FIELD_NAME_MAP["vz"] == "V_3"

    def test_all_bfield_components(self) -> None:
        assert FIELD_NAME_MAP["bx1"] == "B_1"
        assert FIELD_NAME_MAP["by1"] == "B_2"
        assert FIELD_NAME_MAP["bz1"] == "B_3"

    def test_density_and_pressure(self) -> None:
        assert FIELD_NAME_MAP["rr"] == "rho_m"
        assert FIELD_NAME_MAP["pp"] == "P"


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
        # Presence + value: a dropped or mis-parsed header would fail
        # the equality check.  Fixture is gc012 with known DIPOLETIME.
        assert grid.metadata["DIPOLETIME"] == "1967:01:01:00:00:00"
        assert grid.metadata["BASETIME"].strip().startswith("1967:01:01")


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

    def test_field_sanity_checks(self, all_fields) -> None:
        fields, *_ = all_fields
        # Density strictly positive (magnetohydrodynamic fixture)
        assert fields["rr"].min() > 0.0
        # Pin fixture-specific ranges.  These bracket the actual values
        # tight enough to catch endian/sign bugs that would silently
        # produce garbage values outside the physical MHD regime.
        # rr in cm⁻³: magnetotail densities 0.1–25
        assert 0.1 < fields["rr"].min() < 1.0
        assert 10.0 < fields["rr"].max() < 30.0
        # vx in km/s: solar wind ~300 km/s, jets < 1500 km/s
        assert -1000.0 < fields["vx"].min() < -500.0
        assert 800.0 < fields["vx"].max() < 1500.0
        # pp in pPa: solar wind ~1 pPa, ramp-up inside magnetosphere
        assert 0.1 < fields["pp"].min() < 1.0
        # |B| in nT: mostly < 100 nT, dipole-near values can be large
        b_mag = np.sqrt(fields["bx1"] ** 2 + fields["by1"] ** 2 + fields["bz1"] ** 2)
        assert 1.0 < b_mag.min() < 5.0
        assert 5e4 < b_mag.max() < 1e5


class TestOpenGGCMReader:
    """Integration tests for the full reader pipeline."""

    def test_open_and_timesteps(self, reader_cfg_ds) -> None:
        reader, cfg, _ds = reader_cfg_ds
        assert cfg.model_name == "OpenGGCM"
        assert cfg.model_type == "MHD"
        assert cfg.frame == "GSM"
        steps = reader.available_timesteps(FIXTURE_DIR)
        assert 6300 in steps
        assert steps == sorted(steps)

    def test_read_timestep(self, reader_cfg_ds) -> None:
        _reader, _cfg, ds = reader_cfg_ds

        names = ds.field_names()
        for expected in (
            "B_1",
            "B_2",
            "B_3",
            "V_1",
            "V_2",
            "V_3",
            "rho_m",
            "P",
            "n_s0",
        ):
            assert expected in names, f"Missing field: {expected}"

        assert ds["B_1"].shape == (29, 16, 16)

        # B-field should be in SI (Tesla), range ~1e-9 to 1e-4
        b1 = ds["B_1"]
        assert np.abs(b1).max() < 1e-3  # < 1 mT

        # Velocity in SI (m/s), range ~1e4 to 1e6
        v1 = ds["V_1"]
        assert np.abs(v1).max() < 1e7  # < 10,000 km/s

    def test_cartesian_aliases(self, reader_cfg_ds) -> None:
        _reader, _cfg, ds = reader_cfg_ds
        assert ds.has_field("Bx")
        assert ds.has_field("By")
        assert ds.has_field("Bz")
        np.testing.assert_array_equal(ds["Bx"], ds["B_1"])

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


class TestStaggerProvenance:
    """``.3df`` output is cell-centred, and the metadata must say so.

    The reader used to stamp ``convention="staggered"`` with
    ``field_locations={"B": "face", "E": "edge"}`` while returning the raw
    arrays and never destaggering. That claim was wrong twice over: it
    named a location for ``E``, which ``.3df`` does not emit, and it told
    every downstream operator that a co-located stencil was invalid.
    """

    def test_convention_is_cell_centred(self, reader_cfg_ds) -> None:
        _, _, ds = reader_cfg_ds
        assert ds.metadata["stagger"].convention == "cell"

    def test_no_destagger_was_performed(self, reader_cfg_ds) -> None:
        """``interpolation_order`` stays None: nothing was interpolated."""
        _, _, ds = reader_cfg_ds
        assert ds.metadata["stagger"].interpolation_order is None

    def test_every_field_shares_one_shape(self, reader_cfg_ds) -> None:
        """The evidence for co-location: B carries rho's cell count, not nx+1."""
        _, _, ds = reader_cfg_ds
        shapes = {np.asarray(ds[name]).shape for name in ds.field_names()}
        assert len(shapes) == 1, f"co-located grid requires one shape, got {shapes}"
