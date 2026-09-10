"""OpenGGCM simulation reader implementing the SimulationReader protocol."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from pypic.containers import StaggerInfo
from pypic.coordinates.geometry import CARTESIAN
from pypic.grid import GridInfo
from pypic.readers._base import ReaderBase
from pypic.readers._config_helpers import normalize_fields
from pypic.readers.openggcm._field_io import read_3df_field_names, read_3df_file
from pypic.readers.openggcm._field_map import (
    DEFAULT_SKIP,
    FIELD_NAME_MAP,
    convert_fields_to_si,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from pypic.containers import SimulationConfig
    from pypic.dataset import FieldDataset
    from pypic.readers.openggcm._grid import OpenGGCMGrid


_3DF_PATTERN = re.compile(r"\.3df\.(\d+)$")

_STAGGER = StaggerInfo(
    convention="cell",
    notes=(
        "OpenGGCM integrates on a Yee mesh (B on faces, E on edges) under "
        "Evans-Hawley constrained transport, but writes .3df diagnostic "
        "output at cell centres: every field record carries the same "
        "nx*ny*nz count as rho and P, where a face-centred B would carry "
        "nx+1 along its normal. Nothing to destagger on load."
    ),
)


class OpenGGCMReader(ReaderBase):
    """Read OpenGGCM .3df field output on a non-uniform grid.

    Parameters
    ----------
    grid : OpenGGCMGrid
        Parsed grid definition.
    prefix : str
        Filename prefix (e.g. ``"gc012"`` for ``gc012.3df.006300``).
    sim_config : SimulationConfig
        Merged run configuration. Its normalization converts the SI
        values on disk to code units; identity leaves them in SI.
    """

    def __init__(
        self,
        grid: OpenGGCMGrid,
        prefix: str,
        sim_config: SimulationConfig,
    ) -> None:
        super().__init__(sim_config)
        self._grid = grid
        self._prefix = prefix

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
        return sorted(
            int(m.group(1))
            for entry in path.glob(f"{self._prefix}.3df.*")
            if (m := _3DF_PATTERN.search(entry.name))
        )

    def available_fields_mapping(self, path: Path, step: int) -> dict[str, str | None]:
        """Map canonical field names to native ``.3df`` record names at *step*.

        Scans the file's ``FIELD-3D-1`` markers without decoding any
        WRN2 payload.
        """
        filename = path / f"{self._prefix}.3df.{step:06d}"
        native = [
            name for name in read_3df_field_names(filename) if name not in DEFAULT_SKIP
        ]
        mapping: dict[str, str | None] = {
            FIELD_NAME_MAP.get(name, name): name for name in native
        }
        if "rr" in native:
            mapping["n_s0"] = "rr"
        return mapping

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

        raw_fields, _ts, nx, ny, nz = read_3df_file(filename, skip=skip)

        # Verify grid dimensions match
        if (nx, ny, nz) != (self._grid.nx, self._grid.ny, self._grid.nz):
            msg = (
                f"Dimension mismatch: file ({nx}, {ny}, {nz}) vs "
                f"grid ({self._grid.nx}, {self._grid.ny}, {self._grid.nz})"
            )
            raise ValueError(msg)

        sc = self._require_config()
        code_fields = normalize_fields(
            convert_fields_to_si(raw_fields), sc.normalization
        )
        if wanted_canonical is not None:
            code_fields = {
                k: v for k, v in code_fields.items() if k in wanted_canonical
            }

        return self._finish(
            code_fields,
            step=step,
            coords={"x": self._grid.x, "y": self._grid.y, "z": self._grid.z},
            extra={"is_uniform_grid": False, "stagger": _STAGGER},
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
