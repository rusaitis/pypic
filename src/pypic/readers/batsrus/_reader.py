"""BATSRUS simulation reader."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, assert_never

import numpy as np

from pypic.containers import SimulationConfig, StaggerInfo
from pypic.coordinates import CARTESIAN, GEOMETRY_BY_NAME
from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.readers.batsrus._config import BATSRUSConfig, to_simulation_config
from pypic.readers.batsrus._field_map import (
    FIELD_NAME_MAP,
    SKIP_FIELDS,
    convert_fields_to_si,
    is_normalized,
)
from pypic.readers.batsrus._grid import (
    assemble_uniform_hdf5,
    assemble_uniform_idl,
    is_uniform_idl,
    regrid_amr_hdf5,
    regrid_amr_idl,
)
from pypic.readers.batsrus._hdf5 import read_batl
from pypic.readers.batsrus._header import BATSRUSHeader, parse_header
from pypic.readers.batsrus._idl import read_idl_cells, read_out_file
from pypic.units import Normalization, PhysicsParams

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from pypic.readers.batsrus import BATSRUSOutputFormat

_STEP_RE = re.compile(r"_n(\d{8})")


class BATSRUSReader:
    """Read BATSRUS simulation output in IDL or HDF5 format.

    Supports three output formats:

    - **Per-cell IDL** (``.h`` + ``*_pe*.idl``): raw per-processor binary
    - **Merged IDL** (``.out`` / ``.outs``): postprocessed snapshot files
    - **HDF5 BATL** (``.batl``): block-structured HDF5

    AMR grids are automatically regridded to the finest resolution.
    """

    def __init__(
        self,
        config: BATSRUSConfig,
        output_format: BATSRUSOutputFormat,
        prefix: str,
        *,
        geometry: str = "cartesian",
    ) -> None:
        self._config = config
        self._output_format = output_format
        self._prefix = prefix
        self._geometry = geometry
        self._sim_config: SimulationConfig | None = None

    def available_timesteps(self, path: Path) -> list[int]:
        """Return sorted list of available timestep indices."""
        from pypic.readers.batsrus import BATSRUSOutputFormat

        steps: set[int] = set()

        match self._output_format:
            case BATSRUSOutputFormat.HDF5:
                pattern = f"{self._prefix}*.batl"
            case BATSRUSOutputFormat.IDL:
                pattern = f"{self._prefix}*.h"
            case BATSRUSOutputFormat.OUT:
                pattern = f"{self._prefix}*.out"
            case _ as unreachable:
                assert_never(unreachable)

        for f in path.glob(pattern):
            m = _STEP_RE.search(f.stem)
            if m:
                steps.add(int(m.group(1)))

        return sorted(steps)

    def _build_var_mapping(self, var_names: tuple[str, ...]) -> dict[str, str | None]:
        """Map native BATSRUS var names to canonical, return canonical→native."""
        mapping: dict[str, str | None] = {}
        for vname in var_names:
            if vname in SKIP_FIELDS:
                continue
            canonical = FIELD_NAME_MAP.get(vname, vname)
            mapping[canonical] = vname
        return mapping

    def _get_var_names(self, path: Path, step: int) -> tuple[str, ...]:
        """Extract native variable names from header/metadata at *step*."""
        import h5py

        from pypic.readers.batsrus import BATSRUSOutputFormat

        match self._output_format:
            case BATSRUSOutputFormat.IDL:
                header_file = self._find_file(path, step, ".h")
                return parse_header(header_file).var_names
            case BATSRUSOutputFormat.HDF5:
                batl_file = self._find_file(path, step, ".batl")
                with h5py.File(batl_file, "r") as f:
                    return tuple(x.decode().strip() for x in f["NamePlotVar_V"][:])
            case BATSRUSOutputFormat.OUT:
                out_file = self._find_file(path, step, ".out")
                return parse_header(out_file).var_names
            case _ as unreachable:
                assert_never(unreachable)

    def available_fields_mapping(self, path: Path, step: int) -> dict[str, str | None]:
        """Map canonical field names to native (on-disk) names at *step*.

        Parses file headers or HDF5 metadata without loading arrays.

        Parameters
        ----------
        path : Path
            Directory containing the simulation output.
        step : int
            Timestep index.

        Returns
        -------
        dict[str, str | None]
            Canonical → native name.
        """
        return self._build_var_mapping(self._get_var_names(path, step))

    def available_fields(self, path: Path, step: int) -> list[str]:
        """List canonical field names at *step* without loading arrays.

        Parameters
        ----------
        path : Path
            Directory containing the simulation output.
        step : int
            Timestep index.

        Returns
        -------
        list[str]
            Sorted canonical field names.
        """
        return sorted(self.available_fields_mapping(path, step))

    def read_timestep(
        self,
        path: Path,
        step: int,
        *,
        fields: Iterable[str] | None = None,
        target_resolution: float | None = None,
    ) -> FieldDataset:
        """Read field data for a single timestep.

        Parameters
        ----------
        path
            Directory containing the simulation output.
        step
            Timestep index.
        fields : Iterable[str] | None
            When given, only include these canonical field names.
        target_resolution
            Target cell size in code units for AMR regridding. When
            ``None`` (default), regrids to the finest resolution. When
            set, snapped to the nearest AMR level present in the data.
            Ignored for uniform grids.

        Returns
        -------
        FieldDataset
            Field data with canonical names, optionally converted to SI.
        """
        from pypic.readers.batsrus import BATSRUSOutputFormat

        canonical_set = set(fields) if fields is not None else None
        match self._output_format:
            case BATSRUSOutputFormat.HDF5:
                return self._read_hdf5(
                    path,
                    step,
                    fields=canonical_set,
                    target_resolution=target_resolution,
                )
            case BATSRUSOutputFormat.IDL:
                return self._read_idl(
                    path,
                    step,
                    fields=canonical_set,
                    target_resolution=target_resolution,
                )
            case BATSRUSOutputFormat.OUT:
                return self._read_out(path, step, fields=canonical_set)
            case _ as unreachable:
                assert_never(unreachable)

    def _read_idl(
        self,
        path: Path,
        step: int,
        *,
        fields: set[str] | None = None,
        target_resolution: float | None = None,
    ) -> FieldDataset:
        """Read per-cell IDL format."""
        header_file = self._find_file(path, step, ".h")
        header = parse_header(header_file)
        geo = GEOMETRY_BY_NAME.get(header.geometry, CARTESIAN)

        idl_files = self._find_idl_files(path, step)
        all_coords = []
        all_dx = []
        all_state = []
        for idl_file in idl_files:
            coords, dx, state = read_idl_cells(idl_file, header)
            all_coords.append(coords)
            all_dx.append(dx)
            all_state.append(state)

        coords = np.concatenate(all_coords, axis=0)
        dx = np.concatenate(all_dx, axis=0)
        state = np.concatenate(all_state, axis=0)

        if is_uniform_idl(dx):
            field_data, grid = assemble_uniform_idl(
                coords, dx, state, header.var_names, header.ndim, geometry=geo
            )
        else:
            field_data, grid = regrid_amr_idl(
                coords,
                dx,
                state,
                header.var_names,
                header.ndim,
                target_dx=target_resolution,
                geometry=geo,
            )

        # Unit conversion
        unit_names = self._parse_unit_names(header)
        if unit_names and not is_normalized(header.unit_string):
            field_data = convert_fields_to_si(
                field_data,
                header.var_names,
                unit_names,
            )

        # Filter to requested fields
        if fields is not None:
            field_data = {k: v for k, v in field_data.items() if k in fields}

        normalization = Normalization.identity()
        if self._sim_config is None:
            self._sim_config = to_simulation_config(
                self._config, header, grid=grid, sim_dir=path
            )

        physics = PhysicsParams(gamma=self._config.gamma)
        metadata: dict[str, Any] = {
            "step": header.n_step,
            "time": header.time,
            "format": "idl",
            "stagger": StaggerInfo(convention="cell"),
        }
        if not is_uniform_idl(dx):
            metadata["is_regridded"] = True

        return FieldDataset.from_arrays(
            field_data,
            grid,
            normalization,
            physics=physics,
            metadata=metadata,
            strict_fields=False,
        )

    def _read_hdf5(
        self,
        path: Path,
        step: int,
        *,
        fields: set[str] | None = None,
        target_resolution: float | None = None,
    ) -> FieldDataset:
        """Read HDF5 BATL format."""
        batl_file = self._find_file(path, step, ".batl")

        # Compute native names to read from canonical wanted set
        native_wanted: set[str] | None = None
        if fields is not None:
            native_wanted = set()
            for native, canonical in FIELD_NAME_MAP.items():
                if canonical in fields:
                    native_wanted.add(native)
            # Also include canonical names not in the map (pass-through)
            for f in fields:
                if f not in FIELD_NAME_MAP.values():
                    native_wanted.add(f)

        batl = read_batl(batl_file, fields=native_wanted)
        geo = GEOMETRY_BY_NAME.get(self._geometry, CARTESIAN)

        is_uniform = len(set(batl.refine_level)) <= 1
        if is_uniform:
            field_data, grid = assemble_uniform_hdf5(batl, geometry=geo)
        else:
            field_data, grid = regrid_amr_hdf5(
                batl, target_dx=target_resolution, geometry=geo
            )

        unit_names = batl.unit_names
        unit_str = " ".join(unit_names)
        if unit_names and not is_normalized(unit_str):
            field_data = convert_fields_to_si(
                field_data,
                batl.var_names,
                unit_names,
            )

        # Filter to requested fields
        if fields is not None:
            field_data = {k: v for k, v in field_data.items() if k in fields}

        normalization = Normalization.identity()
        if self._sim_config is None:
            self._sim_config = to_simulation_config(
                self._config, grid=grid, sim_dir=path
            )

        physics = PhysicsParams(gamma=self._config.gamma)
        metadata: dict[str, Any] = {
            "step": batl.n_step,
            "time": batl.time,
            "format": "hdf5",
            "stagger": StaggerInfo(convention="cell"),
        }
        if not is_uniform:
            metadata["is_regridded"] = True

        return FieldDataset.from_arrays(
            field_data,
            grid,
            normalization,
            physics=physics,
            metadata=metadata,
            strict_fields=False,
        )

    def _read_out(
        self,
        path: Path,
        step: int,
        *,
        fields: set[str] | None = None,
    ) -> FieldDataset:
        """Read merged .out format."""
        out_file = self._find_file(path, step, ".out")
        coord, state, var_names, out_meta = read_out_file(out_file)

        ndim = int(out_meta["ndim"])
        dims = out_meta["dims"]

        # Determine geometry: .out files encode non-Cartesian as negative ndim
        is_cart = out_meta.get("is_cartesian", True)
        geo = CARTESIAN if is_cart else GEOMETRY_BY_NAME.get(self._geometry, CARTESIAN)

        # Build fields dict with canonical names
        field_data: dict[str, np.ndarray] = {}
        for iv, vname in enumerate(var_names):
            if vname in SKIP_FIELDS:
                continue
            canonical = FIELD_NAME_MAP.get(vname, vname)
            if fields is not None and canonical not in fields:
                continue
            field_data[canonical] = state[iv]

        # Build grid from coordinate arrays
        spacing = tuple(
            float(coord[d].flat[1] - coord[d].flat[0]) if dims[d] > 1 else 1.0
            for d in range(ndim)
        )
        origin = tuple(float(coord[d].flat[0] - spacing[d] / 2) for d in range(ndim))

        grid = GridInfo(
            dimensions=dims,
            spacing=spacing,
            origin=origin,
            geometry=geo,
        )

        normalization = Normalization.identity()
        physics = PhysicsParams(gamma=self._config.gamma)
        metadata_out: dict[str, Any] = {
            "step": out_meta.get("step", step),
            "time": out_meta.get("time", 0.0),
            "format": "out",
            "stagger": StaggerInfo(convention="cell"),
        }

        return FieldDataset.from_arrays(
            field_data,
            grid,
            normalization,
            physics=physics,
            metadata=metadata_out,
            strict_fields=False,
        )

    def _find_file(self, path: Path, step: int, suffix: str) -> Path:
        """Find a file matching the prefix and step number."""
        step_str = f"_n{step:08d}"
        # Also try time-based naming: _t{time}_n{step}
        candidates = list(path.glob(f"{self._prefix}*{step_str}*{suffix}"))
        if not candidates:
            # Try without prefix
            candidates = list(path.glob(f"*{step_str}*{suffix}"))
        if not candidates:
            msg = f"No {suffix} file found for step {step} in {path}"
            raise FileNotFoundError(msg)
        if len(candidates) > 1:
            log.warning(
                "Multiple %s files for step %d: %s; using %s",
                suffix,
                step,
                [c.name for c in candidates],
                candidates[0].name,
            )
        return candidates[0]

    def _find_idl_files(self, path: Path, step: int) -> list[Path]:
        """Find all per-processor .idl files for a given step."""
        step_str = f"_n{step:08d}"
        # Also try time-based: _t{time}_n{step}
        files = sorted(path.glob(f"*{step_str}*_pe*.idl"))
        if not files:
            # Try matching on time pattern
            for h_file in path.glob(f"*_n{step:08d}.h"):
                stem = h_file.stem
                files = sorted(path.glob(f"{stem}_pe*.idl"))
                if files:
                    break
        if not files:
            # Broadest search: find any .idl files with this step number
            files = sorted(f for f in path.glob("*.idl") if step_str in f.name)
        if not files:
            msg = f"No .idl files found for step {step} in {path}"
            raise FileNotFoundError(msg)
        return files

    def _parse_unit_names(self, header: BATSRUSHeader) -> tuple[str, ...]:
        """Extract per-variable unit strings from header.

        The unit string in the header has format:
        ``"timestamp; unit1 unit2 ... unitN"`` or just ``"unit1 unit2 ..."``.
        There are also units for scalar parameters appended at the end.
        """
        raw = header.unit_string.strip()
        if not raw or is_normalized(raw):
            return ()
        # Strip optional leading timestamp
        if ";" in raw:
            raw = raw.split(";", 1)[1].strip()
        from itertools import takewhile

        parts = raw.split()
        n_coord_units = sum(1 for _ in takewhile(lambda p: p == "R", parts))
        var_units = parts[n_coord_units : n_coord_units + header.n_plot_var]
        return tuple(var_units)
