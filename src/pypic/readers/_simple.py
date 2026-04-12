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
           def _read_raw(self, filepath, **kwargs):
               with h5py.File(filepath, "r") as f:
                   return {k: np.array(f[k]).T for k in f
                           if isinstance(f[k], h5py.Dataset)}

3. **Register** for auto-detection via ``open_simulation``::

       from pypic.readers import register_reader
       register_reader("my_code", my_probe, my_factory)

The canonical HDF5 layout (schema.md § 4) stores field arrays under
a ``fields/`` group and grid metadata as attributes on a ``grid/``
group.  When files include this metadata, no ``config`` is needed.
When they don't, pass ``config`` explicitly.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from pypic.containers import SimulationConfig, TabularData
from pypic.coordinates.geometry import CARTESIAN, GEOMETRY_BY_NAME
from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.units import Normalization

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from pypic.readers._registry import Simulation
    from pypic.types import FloatArray
    from pypic.units import SpeciesInfo

log = logging.getLogger(__name__)


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
    geometry = GEOMETRY_BY_NAME.get(str(geo_name), CARTESIAN)

    dt = float(attrs["dt"]) if "dt" in attrs else None

    boundary: tuple[str, ...] | None = None
    if "boundary" in attrs:
        raw = attrs["boundary"]
        boundary = tuple(b.decode() if isinstance(b, bytes) else str(b) for b in raw)

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

    def available_fields_mapping(self, path: Path, step: int) -> dict[str, str | None]:
        """Map canonical field names to native (on-disk) names at *step*.

        Opens the HDF5 file and lists dataset names in the fields
        group, applying ``field_map`` if configured.

        Parameters
        ----------
        path : Path
            Directory containing the data files.
        step : int
            Timestep index.

        Returns
        -------
        dict[str, str | None]
            Canonical → native name.
        """
        filepath = path / self._file_pattern.format(step=step)
        with h5py.File(filepath, "r") as f:
            group = self._resolve_fields_group(f)
            native_names = [
                name
                for name in group
                if isinstance(group[name], h5py.Dataset) and group[name].ndim >= 2
            ]
        if self._field_map is not None:
            fm = self._field_map
            return {fm.get(n, n): n for n in native_names}
        return {n: n for n in native_names}

    def available_fields(self, path: Path, step: int) -> list[str]:
        """List canonical field names at *step* without loading arrays.

        Parameters
        ----------
        path : Path
            Directory containing the data files.
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
    ) -> FieldDataset:
        """Read field data for a single timestep.

        Parameters
        ----------
        path : Path
            Directory containing the data files.
        step : int
            Timestep index.
        fields : Iterable[str] | None
            When given, only read these canonical field names.

        Returns
        -------
        FieldDataset
            Field data with canonical names.

        Raises
        ------
        FileNotFoundError
            If the expected file does not exist (default I/O path).
        ValueError
            If grid metadata is missing from both the file
            and ``config``.
        """
        filepath = path / self._file_pattern.format(step=step)
        is_custom = self._is_read_raw_overridden()

        if not is_custom and not filepath.exists():
            msg = f"File not found: {filepath}"
            raise FileNotFoundError(msg)

        canonical_set = set(fields) if fields is not None else None

        raw = self._read_raw(filepath, fields=canonical_set)
        field_data = self._apply_field_map(raw)

        if is_custom:
            if canonical_set is not None:
                field_data = {k: v for k, v in field_data.items() if k in canonical_set}
            grid = self._grid
            if grid is None and self._config is not None:
                grid = self._config.grid
            if grid is None:
                msg = (
                    "Custom _read_raw requires grid metadata. "
                    "Pass grid=GridInfo(...) or "
                    "config=SimulationConfig(...)."
                )
                raise ValueError(msg)
            normalization = (
                self._normalization
                or (self._config.normalization if self._config else None)
                or Normalization.identity()
            )
            metadata: dict[str, Any] = {"step": step}
        else:
            with h5py.File(filepath, "r") as f:
                grid = self._resolve_grid(f, filepath)
                normalization = self._resolve_normalization(f)
                metadata = self._read_metadata(f, step)

        from pypic.units import PhysicsParams

        physics: PhysicsParams = PhysicsParams()
        species: tuple[SpeciesInfo, ...] = ()
        if self._config is not None:
            physics = self._config.physics
            species = self._config.species

        return FieldDataset.from_arrays(
            field_data,
            grid,
            normalization,
            species=species,
            physics=physics,
            metadata=metadata,
        )

    def _read_raw(
        self,
        filepath: Path,
        *,
        fields: set[str] | None = None,
    ) -> dict[str, FloatArray]:
        r"""Read raw arrays from a single file.

        Returns arrays keyed by **native** field names (before
        ``field_map`` is applied).  The parent class handles renaming,
        grid resolution, and ``FieldDataset`` construction.

        Override this method to support non-standard file formats
        (binary, NetCDF, transposed HDF5, multi-file, etc.).  When
        overridden, grid metadata must come from ``grid=`` or
        ``config=`` — the framework will not try to read HDF5
        attributes.

        Parameters
        ----------
        filepath : Path
            Full path to the data file.
        fields : set[str] | None
            Canonical field names to read.  When ``None``, read all.
            Only used by the base-class implementation; subclass
            overrides may ignore this parameter.

        Returns
        -------
        dict[str, FloatArray]
            Arrays keyed by native (pre-mapping) field names.
        """
        with h5py.File(filepath, "r") as f:
            group = self._resolve_fields_group(f)
            if self._field_map is not None:
                return self._read_mapped_native(group, fields=fields)
            return self._read_all_arrays(group, fields=fields)

    def _apply_field_map(
        self,
        raw: dict[str, FloatArray],
    ) -> dict[str, FloatArray]:
        """Rename native field names to canonical using ``field_map``."""
        if self._field_map is None:
            return raw
        return {self._field_map.get(k, k): v for k, v in raw.items()}

    def _is_read_raw_overridden(self) -> bool:
        """Check whether a subclass overrides ``_read_raw``."""
        return type(self)._read_raw is not SimpleReader._read_raw

    def _resolve_fields_group(
        self,
        f: h5py.File,
    ) -> h5py.Group | h5py.File:
        """Resolve the HDF5 group containing field datasets.

        Tries ``fields_group`` first; falls back to root when the
        group is ``"fields"`` and doesn't exist.

        Raises
        ------
        TypeError
            If *fields_group* resolves to a Dataset or other non-Group
            HDF5 object — surfaces a misconfiguration with a clear
            message instead of crashing downstream during iteration.
        """
        if not self._fields_group:
            return f
        if self._fields_group in f:
            grp = f[self._fields_group]
            if not isinstance(grp, h5py.Group):
                msg = (
                    f"{self._fields_group!r} in {f.filename} is a "
                    f"{type(grp).__name__}, expected a Group. Configure "
                    f"fields_group= to point at an HDF5 Group containing "
                    f"field datasets."
                )
                raise TypeError(msg)
            return grp
        if self._fields_group != "fields":
            msg = (
                f"Group {self._fields_group!r} not found in "
                f"{f.filename}. Available: {list(f.keys())}"
            )
            raise KeyError(msg)
        # "fields" not found → fall back to root silently
        return f

    def _read_mapped_native(
        self,
        group: h5py.Group | h5py.File,
        *,
        fields: set[str] | None = None,
    ) -> dict[str, FloatArray]:
        """Read datasets from the group, return native names.

        Mapped fields are included unconditionally.  Unmapped fields
        are included only when they have ndim >= 2 (skipping scalars
        and 1-D coordinate arrays).

        When *fields* is given (canonical names), only datasets whose
        canonical name is in the set are read.
        """
        assert self._field_map is not None
        mapped_native = set(self._field_map.keys())
        result: dict[str, FloatArray] = {}
        for name in group:
            ds = group[name]
            if not isinstance(ds, h5py.Dataset):
                continue
            if name not in mapped_native and ds.ndim < 2:
                continue
            if fields is not None:
                canonical = self._field_map.get(name, name)
                if canonical not in fields:
                    continue
            result[name] = np.asarray(ds, dtype=np.float64)
        return result

    def _read_all_arrays(
        self,
        group: h5py.Group | h5py.File,
        *,
        fields: set[str] | None = None,
    ) -> dict[str, FloatArray]:
        """Read datasets with ndim >= 2 (skip scalars, coords).

        When *fields* is given (canonical names — same as dataset names
        when no ``field_map``), only matching datasets are read.
        """
        result: dict[str, FloatArray] = {}
        for name in group:
            ds = group[name]
            if not isinstance(ds, h5py.Dataset):
                continue
            if ds.ndim < 2:
                continue
            if fields is not None and name not in fields:
                continue
            result[name] = np.asarray(ds, dtype=np.float64)
        return result

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
        self,
        f: h5py.File,
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

    def available_auxiliary(self, path: Path) -> list[str]:
        """Return names of available auxiliary datasets.

        Default returns ``[]``.  Subclasses may override.

        Parameters
        ----------
        path : Path
            Simulation output directory.

        Returns
        -------
        list[str]
        """
        return []

    def load_auxiliary(self, path: Path, name: str) -> TabularData:
        """Load a named auxiliary dataset.

        Default raises ``KeyError``.  Subclasses may override.

        Parameters
        ----------
        path : Path
            Simulation output directory.
        name : str
            Dataset name.

        Raises
        ------
        KeyError
            Always, unless overridden by a subclass.
        """
        msg = f"No auxiliary dataset {name!r}"
        raise KeyError(msg)


def can_read_confidence(path: Path) -> float:
    """Estimate confidence that *path* contains canonical HDF5 output.

    Low confidence by design — specific readers (iPIC3D, BATSRUS)
    should win when their signatures are present.

    Parameters
    ----------
    path : Path
        Directory to check.

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
        log.debug("Failed to read %s", h5_file, exc_info=True)

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
) -> Simulation:
    """Open a directory of HDF5 files.

    Returns a `Simulation` object::

        sim = open_simple(path, field_map={...})
        sim.steps               # [0, 100, 200]
        ds = sim.read(step=100)

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
    Simulation
        Wraps the reader, config, and path.
    """
    # Deferred to avoid circular import: _registry imports _simple at module level
    import pathlib

    from pypic.readers._registry import Simulation
    from pypic.readers.config import load_config

    path_obj = pathlib.Path(path)

    if path_obj.is_file():
        msg = f"Expected a directory, got file: {path_obj}. Pass the parent directory."
        raise ValueError(msg)

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
        return Simulation(reader, config, path_obj)

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
    return Simulation(reader, auto_config, path_obj)


# Self-register with the reader registry
from pypic.readers._registry import register_reader as _register_reader  # noqa: E402

_register_reader("simple", can_read_confidence, open_simple)  # type: ignore[arg-type]
