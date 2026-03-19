"""Auto-discovered smoke tests for real simulation data.

Collected only when ``--sim-data`` is passed::

    uv run pytest --sim-data -v              # scan examples/
    uv run pytest --sim-data /path/to/runs   # scan arbitrary directory
    uv run pytest --sim-data /path/to/one    # test one simulation dir

Standard ``uv run pytest`` collects zero tests from this file.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import constants

from pypic.readers.batsrus import open_batsrus
from pypic.readers.ipic3d import open_ipic3d
from pypic.readers.openggcm import open_openggcm
from pypic.units import Normalization

log = logging.getLogger(__name__)

_OPENERS = {
    "ipic3d": open_ipic3d,
    "openggcm": open_openggcm,
    "batsrus": open_batsrus,
}

_BATSRUS_STEP_RE = re.compile(r"_n(\d{8})")


def _detect_model(path: Path) -> str | None:
    if list(path.glob("*.inp")) or (path / "settings.hdf").exists():
        return "ipic3d"
    if list(path.glob("grid.*.dat")) and list(path.glob("*.3df.*")):
        return "openggcm"
    has_param = (path / "PARAM.in").exists()
    has_batl = bool(list(path.glob("*.batl")))
    has_idl = bool(list(path.glob("*.h"))) and bool(list(path.glob("*_pe*.idl")))
    has_out = bool(list(path.glob("*.out")))
    if has_param and (has_batl or has_idl or has_out):
        return "batsrus"
    if has_batl or has_idl:
        return "batsrus"
    return None


def _first_timestep(path: Path, model: str) -> int:
    """Find the first available timestep number in *path*."""
    if model == "ipic3d":
        return 0
    if model == "openggcm":
        pattern = re.compile(r"\.3df\.(\d+)$")
        steps = sorted(
            int(m.group(1)) for f in path.iterdir() if (m := pattern.search(f.name))
        )
        if steps:
            return steps[0]
    if model == "batsrus":
        steps_b: list[int] = []
        for f in path.iterdir():
            m = _BATSRUS_STEP_RE.search(f.name)
            if m:
                steps_b.append(int(m.group(1)))
        if steps_b:
            return min(steps_b)
    return 0


_AUXILIARY_PATTERNS = [
    ("ConservedQuantities.txt", "ConservedQuantities (Format A)"),
    ("info-conserved", "ConservedQuantities (Format B, multi-file)"),
    ("SimulationData.txt", "SimulationData"),
    ("SpeciesQuantities.txt", "SpeciesQuantities"),
]


def _detect_auxiliary(path: Path) -> list[str]:
    return [label for pat, label in _AUXILIARY_PATTERNS if (path / pat).exists()]


def _is_mhducla(path: Path) -> bool:
    name_lower = path.name.lower()
    if "uclamhd" in name_lower or "mhducla" in name_lower:
        return True
    return any(
        "uclamhd" in f.name.lower() or "mhducla" in f.name.lower()
        for f in path.iterdir()
    )


_MAX_SCAN_DEPTH = 3


def _discover(root: Path) -> list[tuple[str, Path]]:
    """Discover simulation directories under *root* (up to 3 levels deep).

    If *root* is itself a simulation directory, return it directly.
    Otherwise recurse into subdirectories looking for recognizable
    simulation output (e.g. ``build-run01/output/``).  Stops descending
    a branch as soon as a simulation dir is found.
    """
    if _detect_model(root) is not None:
        return [(root.name, root)]
    if not root.is_dir():
        return []

    found: list[tuple[str, Path]] = []
    _scan(root, root, 1, found)
    found.sort(key=lambda t: t[0])
    return found


def _scan(base: Path, current: Path, depth: int, out: list[tuple[str, Path]]) -> None:
    """Recursively scan for simulation dirs, collecting into *out*."""
    for child in current.iterdir():
        if not child.is_dir():
            continue
        if _detect_model(child) is not None:
            label = str(child.relative_to(base))
            out.append((label, child))
        elif depth < _MAX_SCAN_DEPTH:
            _scan(base, child, depth + 1, out)


def _collect_dirs(config: pytest.Config) -> list[tuple[str, Path]]:
    raw = config.getoption("sim_data")
    if raw is None:
        return []
    root = Path(raw).resolve()
    if not root.exists():
        return []
    return _discover(root)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize ``sim_dir`` over discovered simulation directories."""
    if "sim_dir" not in metafunc.fixturenames:
        return
    dirs = _collect_dirs(metafunc.config)
    ids = [name for name, _ in dirs]
    metafunc.parametrize("sim_dir", [p for _, p in dirs], ids=ids)


# -- Cached open/read to avoid re-reading large files per test ----------------


@lru_cache(maxsize=32)
def _cached_open(sim_dir: Path):
    """Open a simulation dir and return (reader, config, model)."""
    model = _detect_model(sim_dir)
    if model is None or model not in _OPENERS:
        return None
    reader, config = _OPENERS[model](sim_dir)
    return reader, config, model


@lru_cache(maxsize=32)
def _cached_read(sim_dir: Path):
    """Read the first timestep (cached to avoid repeated I/O)."""
    result = _cached_open(sim_dir)
    if result is None:
        return None
    reader, _config, model = result
    step = _first_timestep(sim_dir, model)
    return reader.read_timestep(sim_dir, step)


# -- Universal smoke tests ---------------------------------------------------


class TestSmoke:
    """Universal smoke tests applied to every discovered simulation dir."""

    def _get_opened(self, sim_dir: Path):
        result = _cached_open(sim_dir)
        if result is None:
            pytest.skip(f"No reader for model in {sim_dir.name}")
        return result

    def _read(self, sim_dir: Path):
        ds = _cached_read(sim_dir)
        if ds is None:
            pytest.skip(f"No reader for model in {sim_dir.name}")
        return ds

    def test_opens_without_error(self, sim_dir: Path) -> None:
        reader, config, _model = self._get_opened(sim_dir)
        assert reader is not None
        assert config is not None

    def test_reads_first_timestep(self, sim_dir: Path) -> None:
        ds = self._read(sim_dir)
        assert len(ds.field_names()) > 0

    def test_has_em_fields(self, sim_dir: Path) -> None:
        ds = self._read(sim_dir)
        has_b = ds.has_field("B1") or ds.has_field("Bx")
        has_e = ds.has_field("E1") or ds.has_field("Ex")
        assert has_b or has_e, f"No EM fields found in {sim_dir.name}"

    def test_field_shapes_consistent(self, sim_dir: Path) -> None:
        ds = self._read(sim_dir)
        ndims = {ds[name].ndim for name in ds.field_names()}
        assert len(ndims) == 1, f"Inconsistent ndim across fields: {ndims}"
        assert ndims.pop() >= 2

    def test_no_all_nan_fields(self, sim_dir: Path) -> None:
        ds = self._read(sim_dir)
        all_nan = [name for name in ds.field_names() if np.all(np.isnan(ds[name]))]
        assert not all_nan, f"All-NaN fields: {all_nan}"

    def test_no_all_zero_b_field(self, sim_dir: Path) -> None:
        ds = self._read(sim_dir)
        if not ds.has_field("B1"):
            pytest.skip("No B1 field")
        b1 = ds["B1"]
        assert not np.all(b1 == 0.0), "B1 is all zeros — missed correction?"

    def test_grid_matches_fields(self, sim_dir: Path) -> None:
        result = self._get_opened(sim_dir)
        _reader, config, _model = result
        ds = _cached_read(sim_dir)
        if ds is None:
            pytest.skip(f"No reader for model in {sim_dir.name}")
        grid_dims = tuple(config.grid.dimensions)
        first_field = next(iter(ds.field_names()))
        field_shape = ds[first_field].shape
        assert len(grid_dims) == len(field_shape), (
            f"Grid ndim={len(grid_dims)} vs field ndim={len(field_shape)}"
        )

    def test_auxiliary_files_detected(self, sim_dir: Path) -> None:
        found = _detect_auxiliary(sim_dir)
        if found:
            log.info("Auxiliary files in %s: %s", sim_dir.name, ", ".join(found))


# -- MHDUCLA physics validation ----------------------------------------------


class TestMHDUCLAPhysics:
    """Solar wind physics validation for MHDUCLA runs.

    Known parameters: n=6 cm⁻³, vx=-530 km/s, IMF |B|=8 nT,
    n_ref=0.25 cm⁻³ (2.5e5 m⁻³).
    """

    N_REF = 2.5e5  # m⁻³

    def _setup(self, sim_dir: Path):
        if not _is_mhducla(sim_dir):
            pytest.skip("Not an MHDUCLA simulation")
        result = _cached_open(sim_dir)
        if result is None:
            pytest.skip(f"No reader for {sim_dir.name}")
        _reader, _config, model = result
        if model != "ipic3d":
            pytest.skip("MHDUCLA physics checks only apply to iPIC3D h5hut data")
        ds = _cached_read(sim_dir)
        if ds is None:
            pytest.skip(f"Cannot read {sim_dir.name}")
        norm = Normalization.pic_standard(self.N_REF, constants.m_p, constants.e)
        return ds, norm

    def test_mhducla_solar_wind_density(self, sim_dir: Path) -> None:
        ds, _ = self._setup(sim_dir)
        rho_ion = ds["rho_c_s1"][0, :, :]
        assert_allclose(np.mean(rho_ion), 24.0, atol=1.0)

    def test_mhducla_quasineutrality(self, sim_dir: Path) -> None:
        ds, _ = self._setup(sim_dir)
        rho_total = ds["rho_c"][0, :, :]
        assert_allclose(np.mean(rho_total), 0.0, atol=0.5)

    def test_mhducla_velocity(self, sim_dir: Path) -> None:
        ds, _ = self._setup(sim_dir)
        v1 = ds["V1"][0, :, :]
        expected_v = 530e3 / constants.c
        assert_allclose(np.mean(np.abs(v1)), expected_v, rtol=0.05)

    def test_mhducla_imf_magnitude(self, sim_dir: Path) -> None:
        ds, norm = self._setup(sim_dir)
        b1 = ds["B1"][0, :, :]
        b2 = ds["B2"][0, :, :]
        b3 = ds["B3"][0, :, :]
        b_mag = np.sqrt(b1**2 + b2**2 + b3**2)
        expected_b = 8e-9 / norm.b_field_ref
        assert_allclose(np.mean(b_mag), expected_b, rtol=0.05)

    def test_mhducla_dipole_dominance(self, sim_dir: Path) -> None:
        ds, norm = self._setup(sim_dir)
        b1, b2, b3 = ds["B1"], ds["B2"], ds["B3"]
        max_b = np.max(np.sqrt(b1**2 + b2**2 + b3**2))
        imf_code = 8e-9 / norm.b_field_ref
        assert max_b > 100 * imf_code
