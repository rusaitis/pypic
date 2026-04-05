"""iPIC3D H5hut field output reader."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]
import numpy as np

from pypic.readers.base import FieldDataset, SimulationConfig, TabularData
from pypic.readers.ipic3d._config import IPic3DConfig, to_simulation_config
from pypic.readers.ipic3d._conserved import detect_conserved, load_ipic3d_auxiliary
from pypic.readers.ipic3d._field_map import (
    _FIELD_NAME_MAP,
    _H5HUT_DIAGONAL_PRESSURE,
    _H5HUT_EFLUX_MAP,
    _H5HUT_FIELD_MAP,
    _PRESSURE_COMPONENT_MAP,
    compute_totals_and_filter,
    expand_moment_dependencies,
    gaussian_current_to_si,
    gaussian_density_to_si,
    gaussian_pressure_to_si,
    per_species_canonical,
    per_species_eflux_canonical,
    per_species_pressure_canonical,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from pypic.types import FloatArray

log = logging.getLogger(__name__)

_FIELDS_PATTERN = re.compile(r"-Fields_(\d+)\.h5$")


def _read_field(block: h5py.Group, name: str) -> FloatArray:
    """Read a single field dataset, transpose ZYX→XYZ, promote to float64."""
    raw = block[name]["0"][()]
    data = np.ascontiguousarray(raw.transpose(2, 1, 0), dtype=np.float64)
    return data


class IPic3DH5hutReader:
    """Read iPIC3D H5hut field output.

    H5hut files store all fields for a single timestep in one file
    named ``{SimulationName}-Fields_{cycle:06d}.h5``. Arrays are stored
    in ZYX order (``(nzc+1, nyc+1, nxc+1)``) and must be transposed.

    H5hut stores **all moment quantities** (density, current, pressure)
    divided by 4π (Gaussian convention). The reader applies the 4π
    correction to density, current, and pressure, matching the phdf5/shdf5
    readers. Electromagnetic fields are unaffected.

    Parameters
    ----------
    config : IPic3DConfig
        Parsed iPIC3D configuration.
    """

    def __init__(
        self, config: IPic3DConfig, sim_config: SimulationConfig | None = None
    ) -> None:
        self._config = config
        self._sim_config = sim_config or to_simulation_config(config)

    def available_timesteps(self, path: Path) -> list[int]:
        """Return sorted list of available timestep numbers.

        Scans for ``*-Fields_*.h5`` files under *path*.

        Parameters
        ----------
        path : Path
            Simulation output directory.

        Returns
        -------
        list[int]
            Sorted timestep (cycle) indices.
        """
        steps: list[int] = []
        for entry in path.iterdir():
            m = _FIELDS_PATTERN.search(entry.name)
            if m:
                steps.append(int(m.group(1)))
        return sorted(steps)

    def _find_fields_file(self, path: Path, step: int) -> Path:
        """Locate the H5hut fields file for a given cycle."""
        sim_name = self._config.simulation_name or self._config.case
        candidate = path / f"{sim_name}-Fields_{step:06d}.h5"
        if candidate.exists():
            return candidate
        # Fall back to glob
        matches = list(path.glob(f"*-Fields_{step:06d}.h5"))
        if not matches:
            matches = list(path.glob(f"*-Fields_{step}.h5"))
        if not matches:
            msg = f"No H5hut fields file found for step {step} in {path}"
            raise FileNotFoundError(msg)
        return matches[0]

    def read_timestep(
        self,
        path: Path,
        step: int,
        *,
        fields: Iterable[str] | None = None,
    ) -> FieldDataset:
        """Read field and moment data for a single timestep.

        Parameters
        ----------
        path : Path
            Simulation output directory.
        step : int
            Cycle number (e.g. 202500).
        fields : Iterable[str] | None
            When given, only read these canonical field names.
            Dependencies (per-species fields needed for totals)
            are expanded automatically and excluded from the result.

        Returns
        -------
        FieldDataset
            Field data with canonical names. Density, current, and
            pressure tensor all corrected by 4π (Gaussian→SI-rationalized).
        """
        fields_file = self._find_fields_file(path, step)
        field_data: dict[str, FloatArray] = {}

        wanted: set[str] | None = set(fields) if fields is not None else None

        with h5py.File(fields_file, "r") as f:
            step_group = f["Step#0"]
            nspec = int(step_group.attrs["nspec"][0])
            if nspec != self._config.ns:
                log.warning(
                    "Species count mismatch: HDF5 has nspec=%d but "
                    "config has ns=%d; using config value",
                    nspec,
                    self._config.ns,
                )
                nspec = self._config.ns
            block = step_group["Block"]
            available = set(block.keys())

            expanded: set[str] | None = (
                expand_moment_dependencies(wanted, nspec)
                if wanted is not None
                else None
            )

            # Track which native keys are consumed by known-field logic
            consumed: set[str] = set()

            # Electromagnetic fields (Bx→B1, Ex→E1, etc.)
            for ipic_name, canon_name in _FIELD_NAME_MAP.items():
                if ipic_name in available:
                    consumed.add(ipic_name)
                    if expanded is not None and canon_name not in expanded:
                        continue
                    field_data[canon_name] = _read_field(block, ipic_name)

            # H5hut-specific fields (Vfx→V1, divB→div_B)
            for ipic_name, canon_name in _H5HUT_FIELD_MAP.items():
                if ipic_name in available:
                    consumed.add(ipic_name)
                    if expanded is not None and canon_name not in expanded:
                        continue
                    field_data[canon_name] = _read_field(block, ipic_name)

            # Per-species densities and currents
            # Charge density and current stored as rho/(4pi) and J/(4pi) — Gaussian convention
            for s in range(nspec):
                # Number density: N_{s} → n_s{s}
                n_key = f"N_{s}"
                if n_key in available:
                    consumed.add(n_key)
                    canon = f"n_s{s}"
                    if expanded is None or canon in expanded:
                        field_data[canon] = _read_field(block, n_key)

                # Charge density: rho_{s} → rho_c_s{s}
                rho_key = f"rho_{s}"
                if rho_key in available:
                    consumed.add(rho_key)
                    canon = per_species_canonical("rho", s)
                    if expanded is None or canon in expanded:
                        field_data[canon] = gaussian_density_to_si(
                            _read_field(block, rho_key)
                        )

                # Current density: Jx_{s} → J1_s{s}, etc.
                for comp in ("Jx", "Jy", "Jz"):
                    j_key = f"{comp}_{s}"
                    if j_key in available:
                        consumed.add(j_key)
                        canon = per_species_canonical(comp, s)
                        if expanded is None or canon in expanded:
                            field_data[canon] = gaussian_current_to_si(
                                _read_field(block, j_key)
                            )

                # Pressure tensor: Pxx_{s} → P11_s{s}, etc.
                for pcomp in _PRESSURE_COMPONENT_MAP:
                    p_key = f"{pcomp}_{s}"
                    if p_key in available:
                        consumed.add(p_key)
                        canon = per_species_pressure_canonical(pcomp, s)
                        if expanded is not None and canon not in expanded:
                            continue
                        data = _read_field(block, p_key)
                        # Negate diagonal for species with negative qom
                        # (iPIC3D stores rho*T which inherits the charge sign)
                        if pcomp in _H5HUT_DIAGONAL_PRESSURE:
                            if s >= len(self._config.qom):
                                log.warning(
                                    "Species %d in HDF5 exceeds .inp species "
                                    "count (%d); pressure sign correction "
                                    "skipped",
                                    s,
                                    len(self._config.qom),
                                )
                            elif self._config.qom[s] < 0:
                                data = -data
                        # Pressure tensor stored as P/(4π) — Gaussian convention
                        data = gaussian_pressure_to_si(data)
                        # Convert charge-weighted to mass-weighted pressure:
                        # iPIC3D deposits q·n·v·v; physical P = m·n·v·v
                        # Factor: m/|q| = 1/|qom|
                        if s < len(self._config.qom):
                            data = data / abs(self._config.qom[s])
                        field_data[canon] = data

                # Energy flux: EFx_{s} → EF1_s{s}, etc.
                for efcomp, _ef_canon_base in _H5HUT_EFLUX_MAP.items():
                    ef_key = f"{efcomp}_{s}"
                    if ef_key in available:
                        consumed.add(ef_key)
                        canon = per_species_eflux_canonical(efcomp, s)
                        if expanded is not None and canon not in expanded:
                            continue
                        data = _read_field(block, ef_key)
                        field_data[canon] = gaussian_pressure_to_si(data)

            # Pass through unknown fields with native names, no conversion
            for native_name in available - consumed:
                if expanded is not None and native_name not in expanded:
                    continue
                field_data[native_name] = _read_field(block, native_name)

        field_data = compute_totals_and_filter(field_data, nspec, expanded, wanted)

        sc = self._sim_config
        return FieldDataset.from_arrays(
            field_data,
            sc.grid,
            sc.normalization,
            species=sc.species,
            physics=dict(sc.physics),
            metadata={**dict(sc.metadata), "step": step},
            frame=sc.frame,
            transforms=sc.transforms or None,
        )

    def available_auxiliary(self, path: Path) -> list[str]:
        """Return names of auxiliary datasets at *path*."""
        return detect_conserved(path)

    def load_auxiliary(self, path: Path, name: str) -> TabularData:
        """Load a named auxiliary dataset from *path*."""
        return load_ipic3d_auxiliary(path, name)
