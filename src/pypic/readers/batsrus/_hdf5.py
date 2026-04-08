"""Read BATSRUS HDF5 ``.batl`` (BATL library) output files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from pypic.types import FloatArray


@dataclass(frozen=True, slots=True)
class BATLData:
    """Raw block-structured data from a ``.batl`` HDF5 file."""

    fields: dict[str, FloatArray]
    var_names: tuple[str, ...]
    unit_names: tuple[str, ...]
    bounding_box: FloatArray
    refine_level: NDArray[np.int32]
    coord_db: FloatArray
    ndim: int
    n_step: int
    time: float
    block_size: tuple[int, ...]
    domain_min: tuple[float, ...]
    domain_max: tuple[float, ...]


def read_batl(
    path: Path,
    *,
    fields: set[str] | None = None,
) -> BATLData:
    """Read a BATSRUS ``.batl`` HDF5 file.

    Parameters
    ----------
    path
        Path to the ``.batl`` file.
    fields : set[str] | None
        When given, only load these BATSRUS-native variable names.
        Metadata and grid data are always read.

    Returns
    -------
    BATLData
        Frozen dataclass with block-structured field data and metadata.
    """
    import h5py

    with h5py.File(path, "r") as f:
        # Metadata
        ipm = f["Integer Plot Metadata"][:]
        rpm = f["Real Plot Metadata"][:]

        ndim = int(ipm[2])
        block_size = tuple(int(ipm[7 + i]) for i in range(ndim))
        n_step = int(ipm[1])

        time = float(rpm[0])
        domain_min = tuple(float(rpm[1 + 2 * i]) for i in range(ndim))
        domain_max = tuple(float(rpm[2 + 2 * i]) for i in range(ndim))

        var_names = tuple(x.decode().strip() for x in f["NamePlotVar_V"][:])
        unit_names = tuple(x.decode().strip() for x in f["NamePlotUnit_V"][:])

        bounding_box = np.array(f["bounding box"][:], dtype=np.float64)
        refine_level = np.array(f["refine level"][:], dtype=np.int32)
        coord_db = np.array(f["Coord_DB"][:], dtype=np.float64)

        field_data: dict[str, FloatArray] = {}
        for vname in var_names:
            if fields is not None and vname not in fields:
                continue
            if vname in f:
                field_data[vname] = np.array(f[vname][:], dtype=np.float64)

    return BATLData(
        fields=field_data,
        var_names=var_names,
        unit_names=unit_names,
        bounding_box=bounding_box,
        refine_level=refine_level,
        coord_db=coord_db,
        ndim=ndim,
        n_step=n_step,
        time=time,
        block_size=block_size,
        domain_min=domain_min,
        domain_max=domain_max,
    )
