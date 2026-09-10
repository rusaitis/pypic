"""iPIC3D H5hut field output reader."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import h5py
import numpy as np

from pypic.readers.ipic3d._base import IPic3DReaderBase
from pypic.readers.ipic3d._field_map import (
    _FIELD_NAME_MAP,
    _H5HUT_FIELD_MAP,
    _PRESSURE_COMPONENT_MAP,
    compute_totals_and_filter,
    expand_moment_dependencies,
    infer_total_fields,
    read_species_moments,
    species_moment_names,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from pypic.dataset import FieldDataset
    from pypic.types import FloatArray

_FIELDS_PATTERN = re.compile(r"-Fields_(\d+)\.h5$")

# Every known EM or diagnostic field, native → canonical.
_KNOWN_FIELDS = _FIELD_NAME_MAP | _H5HUT_FIELD_MAP


def _read_field(block: h5py.Group, name: str) -> FloatArray:
    """Read a single field dataset, transpose ZYX→XYZ, promote to float64."""
    raw = block[name]["0"][()]
    data = np.ascontiguousarray(raw.transpose(2, 1, 0), dtype=np.float64)
    return data


def _species_loader(
    block: h5py.Group, available: set[str], species: int
) -> Callable[[str], FloatArray | None]:
    """Loader over H5hut's ``<native>_<species>`` keys for one species."""

    def load(native: str) -> FloatArray | None:
        key = f"{native}_{species}"
        return _read_field(block, key) if key in available else None

    return load


class IPic3DH5hutReader(IPic3DReaderBase):
    """Read iPIC3D H5hut field output.

    H5hut files store all fields for a single timestep in one file
    named ``{SimulationName}-Fields_{cycle:06d}.h5``. Arrays are stored
    in ZYX order (``(nzc+1, nyc+1, nxc+1)``) and must be transposed.

    H5hut stores **all moment quantities** (density, current, pressure)
    divided by 4π (Gaussian convention). The reader applies the 4π
    correction to density, current, and pressure, matching the phdf5/shdf5
    readers. Electromagnetic fields are unaffected.

    Unique to this reader: the single-file-per-timestep layout, ZYX
    transpose, H5hut-specific field naming (uppercase axis letters
    in ``_PRESSURE_COMPONENT_MAP``), and passthrough of unknown native
    fields. Field-name mapping for everything else, the Gaussian
    conversions, pressure-tensor mass correction, and config
    translation live in `pypic.readers.ipic3d._field_map` and
    `pypic.readers.ipic3d._config`, shared with the parallel and
    serial readers.
    """

    def available_timesteps(self, path: Path) -> list[int]:
        """Sorted cycle numbers, from the ``*-Fields_*.h5`` files under *path*."""
        return sorted(
            int(m.group(1))
            for entry in path.iterdir()
            if (m := _FIELDS_PATTERN.search(entry.name))
        )

    def available_fields_mapping(self, path: Path, step: int) -> dict[str, str | None]:
        """Map canonical field names to native (on-disk) names at *step*.

        Opens the H5hut fields file and inspects ``Step#0/Block/``
        keys. Unknown native keys pass through with the same name as
        both key and value, matching `read_timestep`.

        Parameters
        ----------
        path : Path
            Simulation output directory.
        step : int
            Timestep (cycle) index.

        Returns
        -------
        dict[str, str | None]
            Canonical → native name, ``None`` for computed totals.
        """
        ns = self._config.ns
        with h5py.File(self._find_fields_file(path, step), "r") as f:
            available = set(f["Step#0"]["Block"].keys())

        mapping: dict[str, str | None] = {}
        consumed: set[str] = set()
        for native, canon in _KNOWN_FIELDS.items():
            if native in available:
                mapping[canon] = native
                consumed.add(native)
        for s in range(ns):
            for canon, native in species_moment_names(
                s, _PRESSURE_COMPONENT_MAP
            ).items():
                key = f"{native}_{s}"
                if key in available:
                    mapping[canon] = key
                    consumed.add(key)
        mapping.update((native, native) for native in available - consumed)

        for total in infer_total_fields(set(mapping), ns):
            mapping[total] = None
        return mapping

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
        wanted: set[str] | None = set(fields) if fields is not None else None
        field_data: dict[str, FloatArray] = {}

        with h5py.File(fields_file, "r") as f:
            step_group = f["Step#0"]
            nspec = int(step_group.attrs["nspec"][0])
            if nspec != self._config.ns:
                msg = (
                    f"Species count mismatch in {fields_file.name}: HDF5 has "
                    f"nspec={nspec} but the config declares ns={self._config.ns}"
                )
                raise ValueError(msg)
            block = step_group["Block"]
            available = set(block.keys())
            expanded: set[str] | None = (
                expand_moment_dependencies(wanted, nspec)
                if wanted is not None
                else None
            )

            consumed: set[str] = set()
            for native, canon in _KNOWN_FIELDS.items():
                if native in available:
                    consumed.add(native)
                    if expanded is None or canon in expanded:
                        field_data[canon] = _read_field(block, native)

            for s in range(nspec):
                names = species_moment_names(s, _PRESSURE_COMPONENT_MAP)
                consumed.update(
                    key
                    for native in names.values()
                    if (key := f"{native}_{s}") in available
                )
                field_data.update(
                    read_species_moments(
                        _species_loader(block, available, s),
                        s,
                        species_qom=self._config.qom[s],
                        expanded=expanded,
                        pressure_map=_PRESSURE_COMPONENT_MAP,
                    )
                )

            # Unknown fields pass through under their native names, unconverted.
            for native in available - consumed:
                if expanded is None or native in expanded:
                    field_data[native] = _read_field(block, native)

        field_data = compute_totals_and_filter(field_data, nspec, expanded, wanted)
        return self._finish(field_data, step=step)
