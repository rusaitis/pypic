"""OpenGGCM simulation reader implementing the SimulationReader protocol."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from pypic.containers import StaggerInfo
from pypic.coordinates.geometry import CARTESIAN
from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.readers.openggcm._field_io import read_3df_file
from pypic.readers.openggcm._field_map import (
    DEFAULT_SKIP,
    FIELD_NAME_MAP,
    convert_fields_to_si,
)
from pypic.units import Normalization

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from pypic.readers.openggcm._grid import OpenGGCMGrid
    from pypic.types import FloatArray

log = logging.getLogger(__name__)

_3DF_PATTERN = re.compile(r"\.3df\.(\d+)$")


class OpenGGCMReader:
    """Read OpenGGCM .3df field output on a non-uniform grid.

    Parameters
    ----------
    grid : OpenGGCMGrid
        Parsed grid definition.
    prefix : str
        Filename prefix (e.g. ``"gc012"`` for ``gc012.3df.006300``).
    normalization : Normalization | None
        If provided, data is normalized from SI to code units.  If
        ``None``, data is returned in SI.
    """

    def __init__(
        self,
        grid: OpenGGCMGrid,
        prefix: str,
        normalization: Normalization | None = None,
    ) -> None:
        self._grid = grid
        self._prefix = prefix
        self._normalization = normalization

    @property
    def grid(self) -> OpenGGCMGrid:
        """The OpenGGCM non-uniform grid."""
        return self._grid

    def available_timesteps(self, path: Path) -> list[int]:
        """Return sorted list of available timestep indices.

        Scans for ``{prefix}.3df.*`` files under *path*.

        Parameters
        ----------
        path : Path
            Directory containing .3df files.

        Returns
        -------
        list[int]
            Sorted timestep indices.
        """
        steps: list[int] = []
        pattern = f"{self._prefix}.3df.*"
        for entry in path.glob(pattern):
            m = _3DF_PATTERN.search(entry.name)
            if m:
                steps.append(int(m.group(1)))
        return sorted(steps)

    def read_timestep(
        self,
        path: Path,
        step: int,
        *,
        fields: Iterable[str] | None = None,
    ) -> FieldDataset:
        """Read fields for a single timestep.

        Parameters
        ----------
        path : Path
            Directory containing .3df files.
        step : int
            Timestep index (e.g. 6300).
        fields : Iterable[str] | None
            When given, only read these canonical field names.  Skips
            WRN2 decompression for unwanted native fields.

        Returns
        -------
        FieldDataset
            Field data with canonical names in SI (or normalized) units.
            ``metadata["is_uniform_grid"]`` is ``False`` — the grid is
            non-uniform, so ``grid.spacing`` is a mean approximation.
            Use the xarray coordinates for accurate spacing.
        """
        filename = path / f"{self._prefix}.3df.{step:06d}"
        if not filename.exists():
            msg = f"File not found: {filename}"
            raise FileNotFoundError(msg)

        skip = set(DEFAULT_SKIP)
        wanted_canonical: set[str] | None = None
        if fields is not None:
            wanted_canonical = set(fields)
            wanted_native: set[str] = set()
            for native, canonical in FIELD_NAME_MAP.items():
                if canonical in wanted_canonical:
                    wanted_native.add(native)
            # n_s0 is derived from "rr" in convert_fields_to_si
            if "n_s0" in wanted_canonical:
                wanted_native.add("rr")
            # Skip known native fields that aren't wanted
            skip = skip | (set(FIELD_NAME_MAP.keys()) - wanted_native)

        raw_fields, ts, nx, ny, nz = read_3df_file(
            filename,
            skip=skip,
        )

        # Verify grid dimensions match
        if (nx, ny, nz) != (self._grid.nx, self._grid.ny, self._grid.nz):
            msg = (
                f"Dimension mismatch: file ({nx}, {ny}, {nz}) vs "
                f"grid ({self._grid.nx}, {self._grid.ny}, {self._grid.nz})"
            )
            raise ValueError(msg)

        si_fields = convert_fields_to_si(raw_fields)

        if self._normalization is not None:
            norm = self._normalization
            for name, data in si_fields.items():
                si_fields[name] = _normalize_field(name, data, norm)

        # Filter to requested canonical fields
        if wanted_canonical is not None:
            si_fields = {k: v for k, v in si_fields.items() if k in wanted_canonical}

        # Build xr.Dataset with non-uniform coordinates
        dim_names = ["x", "y", "z"]
        coords = {
            "x": self._grid.x,
            "y": self._grid.y,
            "z": self._grid.z,
        }
        data_vars = {
            name: xr.DataArray(data=arr, dims=dim_names)
            for name, arr in si_fields.items()
        }
        dataset = xr.Dataset(data_vars, coords=coords)

        # Build approximate GridInfo (mean spacing)
        grid_info = _make_grid_info(self._grid)

        return FieldDataset(
            dataset,
            grid_info,
            self._normalization or Normalization.identity(),
            metadata={
                "step": step,
                "timestep": ts,
                "prefix": self._prefix,
                "is_uniform_grid": False,
                "stagger": StaggerInfo(
                    convention="staggered",
                    field_locations={"B": "face", "E": "edge"},
                    notes="Yee mesh, B on cell faces, E on cell edges",
                ),
            },
        )


def _make_grid_info(grid: OpenGGCMGrid) -> GridInfo:
    """Create an approximate GridInfo from a non-uniform grid.

    ``GridInfo`` requires uniform spacing, so we use **mean** spacing
    as an approximation.  The true non-uniform coordinates live in the
    ``xr.Dataset`` coords and should be used for any operation that
    depends on accurate spacing (derivatives, interpolation).
    """
    dx = (grid.x[-1] - grid.x[0]) / max(grid.nx - 1, 1)
    dy = (grid.y[-1] - grid.y[0]) / max(grid.ny - 1, 1)
    dz = (grid.z[-1] - grid.z[0]) / max(grid.nz - 1, 1)
    return GridInfo(
        dimensions=(grid.nx, grid.ny, grid.nz),
        spacing=(float(np.abs(dx)), float(np.abs(dy)), float(np.abs(dz))),
        origin=(float(grid.x[0]), float(grid.y[0]), float(grid.z[0])),
        geometry=CARTESIAN,
    )


def _normalize_field(
    name: str,
    data: FloatArray,
    norm: Normalization,
) -> FloatArray:
    """Normalize a single SI field to code units."""
    match name:
        case "V1" | "V2" | "V3":
            return norm.normalize("velocity", data)  # type: ignore[return-value]
        case "B1" | "B2" | "B3":
            return norm.normalize("b_field", data)  # type: ignore[return-value]
        case "rho_m":
            return data / (norm.density_ref * norm.mass_ref)
        case "n_s0":
            return norm.normalize("density", data)  # type: ignore[return-value]
        case "P":
            return data / (norm.density_ref * norm.mass_ref * norm.velocity_ref**2)
        case _:
            return data
