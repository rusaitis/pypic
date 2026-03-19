"""Minimal HDF5 reader — template for adding new simulation codes.

To add a reader for your simulation code, either:

1. **Use directly** with ``field_map`` and ``config``::

       reader = SimpleReader(
           field_map={"Bx_native": "B1", "By_native": "B2", ...},
           config=my_config,
       )
       ds = reader.read_timestep(path, step=100)

2. **Subclass** and override ``_read_raw`` for non-standard layouts::

       class TristanReader(SimpleReader):
           def _read_raw(self, f, step):
               return {k: np.array(f[k]) for k in f}

3. **Register** for auto-detection via ``open_simulation``::

       from pypic.readers import register_reader
       register_reader("my_code", my_probe, my_factory)

The canonical HDF5 layout (SCHEMA.md § 4) stores field arrays under
a ``fields/`` group and grid metadata as attributes on a ``grid/``
group.  When files include this metadata, no ``config`` is needed.
When they don't, pass ``config`` explicitly.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import h5py  # type: ignore[import-untyped]
import numpy as np

from pypic.coordinates.geometry import (
    CARTESIAN,
    CYLINDRICAL,
    SPHERICAL,
    CoordinateGeometry,
)
from pypic.readers.base import FieldDataset, GridInfo, SimulationConfig
from pypic.units import Normalization

if TYPE_CHECKING:
    from pathlib import Path

    from pypic.types import FloatArray
    from pypic.units import SpeciesInfo

log = logging.getLogger(__name__)

_GEOMETRY_MAP: dict[str, CoordinateGeometry] = {
    "cartesian": CARTESIAN,
    "spherical": SPHERICAL,
    "cylindrical": CYLINDRICAL,
}


def _parse_file_pattern(
    pattern: str,
) -> tuple[str, re.Pattern[str]]:
    """Convert a file pattern to a glob string and compiled regex.

    Parameters
    ----------
    pattern : str
        Python format string with a single ``{step...}`` placeholder,
        e.g. ``"output_{step:06d}.h5"``.

    Returns
    -------
    tuple[str, re.Pattern[str]]
        ``(glob_pattern, regex)`` where the regex has one capture group
        for the step number.

    Examples
    --------
    >>> g, r = _parse_file_pattern("output_{step:06d}.h5")
    >>> g
    'output_*.h5'
    >>> bool(r.match("output_000042.h5"))
    True
    """
    glob_pat = re.sub(r"\{step[^}]*\}", "*", pattern)
    escaped = re.escape(pattern)
    regex_pat = re.sub(r"\\{step[^}]*\\}", r"(\\d+)", escaped)
    return glob_pat, re.compile(regex_pat)


def _read_grid_attrs(f: h5py.File) -> GridInfo | None:
    """Extract ``GridInfo`` from the ``grid/`` group attributes.

    Returns ``None`` when the group is missing or lacks required
    attributes (``dimensions``, ``spacing``).
    """
    if "grid" not in f:
        return None
    g = f["grid"]
    attrs = g.attrs

    if "dimensions" not in attrs or "spacing" not in attrs:
        return None

    dims = tuple(int(x) for x in attrs["dimensions"])
    spacing = tuple(float(x) for x in attrs["spacing"])
    origin = tuple(float(x) for x in attrs.get("origin", [0.0] * len(dims)))

    geo_name = attrs.get("geometry", "cartesian")
    if isinstance(geo_name, bytes):
        geo_name = geo_name.decode()
    geometry = _GEOMETRY_MAP.get(str(geo_name), CARTESIAN)

    dt = float(attrs["dt"]) if "dt" in attrs else None

    boundary: tuple[str, ...] | None = None
    if "boundary" in attrs:
        raw = attrs["boundary"]
        boundary = tuple(
            b.decode() if isinstance(b, bytes) else str(b)
            for b in raw
        )

    return GridInfo(
        dimensions=dims,
        spacing=spacing,
        origin=origin,
        geometry=geometry,
        dt=dt,
        boundary=boundary,
    )


def _read_model_attrs(
    f: h5py.File,
) -> tuple[str, str]:
    """Read model name and type from HDF5 root attributes.

    Returns ``("unknown", "unknown")`` for missing attributes.
    """
    model = f.attrs.get("model", "unknown")
    model_type = f.attrs.get("model_type", "unknown")
    if isinstance(model, bytes):
        model = model.decode()
    if isinstance(model_type, bytes):
        model_type = model_type.decode()
    return str(model), str(model_type)


class SimpleReader:
    r"""Minimal HDF5 reader implementing the ``SimulationReader`` protocol.

    Reads HDF5 files where field arrays live under a configurable group
    (default ``"fields/"``).  Grid metadata is resolved in priority
    order:

    1. HDF5 ``grid/`` group attributes (self-describing files).
    2. Explicit *grid* parameter.
    3. Grid from *config*, if provided.

    You only need to supply what the HDF5 files don't already contain.

    Parameters
    ----------
    file_pattern : str
        Python format string with a ``{step}`` placeholder.
        Used both to locate files (``read_timestep``) and to scan
        for available timesteps.
    field_map : dict[str, str] | None
        Mapping from native HDF5 dataset names to canonical field
        names (e.g. ``{"Bx_code": "B1"}``).  When ``None``, dataset
        names are assumed to already be canonical.
    grid : GridInfo | None
        Explicit grid metadata.  Takes precedence over *config* but
        is overridden by HDF5 ``grid/`` attributes when present.
    normalization : Normalization | None
        Unit normalization.  Defaults to ``Normalization.identity()``.
    config : SimulationConfig | None
        Full simulation configuration.  Used as a fallback for grid,
        normalization, species, and physics when individual params
        are not given.
    fields_group : str
        HDF5 group containing field datasets.  Use ``""`` for files
        that store datasets at the root level.

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = SimpleReader(file_pattern="out_{step:04d}.h5")
    >>> reader.file_pattern
    'out_{step:04d}.h5'
    """

    def __init__(
        self,
        *,
        file_pattern: str = "output_{step:06d}.h5",
        field_map: dict[str, str] | None = None,
        grid: GridInfo | None = None,
        normalization: Normalization | None = None,
        config: SimulationConfig | None = None,
        fields_group: str = "fields",
    ) -> None:
        self._file_pattern = file_pattern
        self._field_map = dict(field_map) if field_map else None
        self._grid = grid
        self._normalization = normalization
        self._config = config
        self._fields_group = fields_group
        self._glob, self._regex = _parse_file_pattern(file_pattern)

    @property
    def file_pattern(self) -> str:
        """The file naming pattern."""
        return self._file_pattern

    def available_timesteps(self, path: Path) -> list[int]:
        """Return sorted timestep indices found in *path*.

        Scans for files matching ``file_pattern`` and extracts the
        step number from each filename.

        Parameters
        ----------
        path : Path
            Directory to scan.

        Returns
        -------
        list[int]
            Sorted step numbers.
        """
        steps: list[int] = []
        for entry in path.glob(self._glob):
            m = self._regex.match(entry.name)
            if m:
                steps.append(int(m.group(1)))
        return sorted(steps)

    def read_timestep(
        self, path: Path, step: int,
    ) -> FieldDataset:
        """Read field data for a single timestep.

        Parameters
        ----------
        path : Path
            Directory containing the HDF5 files.
        step : int
            Timestep index.

        Returns
        -------
        FieldDataset
            Field data with canonical names.

        Raises
        ------
        FileNotFoundError
            If the expected file does not exist.
        ValueError
            If grid metadata is missing from both the HDF5 file
            and ``config``.
        """
        filename = path / self._file_pattern.format(step=step)
        if not filename.exists():
            msg = f"File not found: {filename}"
            raise FileNotFoundError(msg)

        with h5py.File(filename, "r") as f:
            fields = self._read_fields(f)
            grid = self._resolve_grid(f, filename)
            normalization = self._resolve_normalization(f)
            metadata = self._read_metadata(f, step)

        physics: dict[str, Any] = {}
        species: tuple[SpeciesInfo, ...] = ()
        if self._config is not None:
            physics = dict(self._config.physics)
            species = self._config.species

        return FieldDataset.from_arrays(
            fields,
            grid,
            normalization,
            species=species,
            physics=physics,
            metadata=metadata,
        )

    def _read_fields(
        self, f: h5py.File,
    ) -> dict[str, FloatArray]:
        """Read field arrays from the configured HDF5 group."""
        group: h5py.Group | h5py.File
        if self._fields_group:
            if self._fields_group not in f:
                msg = (
                    f"Group {self._fields_group!r} not found in "
                    f"{f.filename}. Available: {list(f.keys())}"
                )
                raise KeyError(msg)
            group = f[self._fields_group]
        else:
            group = f

        fields: dict[str, FloatArray] = {}
        for name in group:
            if not isinstance(group[name], h5py.Dataset):
                continue
            canonical = name
            if self._field_map is not None:
                canonical = self._field_map.get(name, name)
            data: FloatArray = np.asarray(
                group[name], dtype=np.float64,
            )
            fields[canonical] = data
        return fields

    def _resolve_grid(
        self,
        f: h5py.File,
        filename: Path,
    ) -> GridInfo:
        """Get grid: HDF5 attrs > explicit grid > config.grid."""
        hdf5_grid = _read_grid_attrs(f)
        if hdf5_grid is not None:
            return hdf5_grid
        if self._grid is not None:
            return self._grid
        if self._config is not None:
            return self._config.grid
        msg = (
            f"No grid/ metadata in {filename.name} and no grid "
            f"or config provided. Pass grid=GridInfo(...) or "
            f"config=SimulationConfig(...)."
        )
        raise ValueError(msg)

    def _resolve_normalization(
        self, f: h5py.File,
    ) -> Normalization:
        """Get normalization: explicit > config > identity."""
        if self._normalization is not None:
            return self._normalization
        if self._config is not None:
            return self._config.normalization
        return Normalization.identity()

    def _read_metadata(
        self,
        f: h5py.File,
        step: int,
    ) -> dict[str, Any]:
        """Extract scalar metadata from root attributes."""
        meta: dict[str, Any] = {"step": step}
        if "time" in f.attrs:
            meta["time"] = float(f.attrs["time"])
        if "step" in f.attrs:
            meta["step"] = int(f.attrs["step"])
        return meta


def probe(path: Path) -> float:
    """Estimate confidence that *path* contains canonical HDF5 output.

    Low confidence by design — specific readers (iPIC3D, BATSRUS)
    should win when their signatures are present.

    Parameters
    ----------
    path : Path
        Directory to probe.

    Returns
    -------
    float
        Confidence in ``[0.0, 1.0]``.
    """
    if not path.is_dir():
        return 0.0

    h5_file = next(path.glob("*.h5"), None)
    if h5_file is None:
        return 0.0

    score = 0.2
    try:
        with h5py.File(h5_file, "r") as f:
            if "fields" in f:
                score += 0.3
            if "grid" in f:
                score += 0.2
    except Exception:
        log.debug("Failed to probe %s", h5_file, exc_info=True)

    return min(score, 1.0)


def open_simple(
    path: Path,
    *,
    file_pattern: str = "output_{step:06d}.h5",
    field_map: dict[str, str] | None = None,
    grid: GridInfo | None = None,
    normalization: Normalization | None = None,
    config: SimulationConfig | None = None,
    config_path: Path | None = None,
    fields_group: str = "fields",
) -> tuple[SimpleReader, SimulationConfig]:
    """Open a directory of HDF5 files.

    Metadata resolution (each level overrides the next):

    1. ``simulation.toml`` from *config_path* or in *path*.
    2. HDF5 ``grid/`` attributes in the first matching file.
    3. Explicit *grid* / *normalization* parameters.
    4. Explicit *config* parameter.

    For most cases you only need to supply what the files lack.

    Parameters
    ----------
    path : Path
        Directory containing HDF5 output files.
    file_pattern : str
        Filename pattern with ``{step}`` placeholder.
    field_map : dict[str, str] | None
        Native-to-canonical field name mapping.
    grid : GridInfo | None
        Explicit grid metadata (when HDF5 files lack it).
    normalization : Normalization | None
        Unit normalization (defaults to identity).
    config : SimulationConfig | None
        Full simulation configuration.  When provided, *grid* and
        *normalization* are ignored.
    config_path : Path | None
        Explicit path to a ``simulation.toml`` file.  When
        ``None``, auto-detected from *path*.
    fields_group : str
        HDF5 group name containing field datasets.

    Returns
    -------
    tuple[SimpleReader, SimulationConfig]
        ``(reader, config)`` pair.
    """
    from pypic.readers.config import load_config

    path_obj = path if hasattr(path, "glob") else __import__("pathlib").Path(path)

    # Try simulation.toml first
    if config is None:
        toml_path = config_path or (path_obj / "simulation.toml")
        if toml_path.exists():
            config = load_config(toml_path)
            log.info("Loaded config from %s", toml_path)

    if config is not None:
        reader = SimpleReader(
            file_pattern=file_pattern,
            field_map=field_map,
            config=config,
            fields_group=fields_group,
        )
        return reader, config

    # Auto-detect from first HDF5 file + explicit args
    glob_pat, _ = _parse_file_pattern(file_pattern)
    first_file = next(path_obj.glob(glob_pat), None)

    h5_grid: GridInfo | None = None
    model_name = "unknown"
    model_type = "unknown"

    if first_file is not None:
        with h5py.File(first_file, "r") as f:
            h5_grid = _read_grid_attrs(f)
            model_name, model_type = _read_model_attrs(f)

    resolved_grid = h5_grid or grid
    if resolved_grid is None:
        if first_file is None:
            msg = (
                f"No files matching {glob_pat!r} in {path} "
                f"and no grid or config provided."
            )
            raise FileNotFoundError(msg)
        msg = (
            f"No grid/ metadata in {first_file.name}. Pass "
            f"grid=GridInfo(...) or config=SimulationConfig(...)."
        )
        raise ValueError(msg)

    resolved_norm = normalization or Normalization.identity()

    auto_config = SimulationConfig(
        model_name=model_name,
        model_type=model_type,
        grid=resolved_grid,
        normalization=resolved_norm,
        species=(),
        physics={},
        frame="simulation",
        metadata={"file_pattern": file_pattern},
    )
    reader = SimpleReader(
        file_pattern=file_pattern,
        field_map=field_map,
        grid=resolved_grid,
        normalization=resolved_norm,
        config=auto_config,
        fields_group=fields_group,
    )
    return reader, auto_config


# Self-register with the reader registry
from pypic.readers._registry import register_reader as _register_reader  # noqa: E402

_register_reader("simple", probe, open_simple)
