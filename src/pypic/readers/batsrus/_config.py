"""Parse BATSRUS ``PARAM.in`` configuration files."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from pypic.containers import SimulationConfig
from pypic.coordinates import CARTESIAN, GEOMETRY_BY_NAME
from pypic.grid import GridInfo
from pypic.readers._config_helpers import merge_simulation_toml
from pypic.units import Normalization, PhysicsParams

if TYPE_CHECKING:
    from pathlib import Path

    from pypic.readers.batsrus._header import BATSRUSHeader


@dataclass(frozen=True, slots=True)
class BATSRUSConfig:
    """Parsed BATSRUS ``PARAM.in`` configuration."""

    description: str = ""
    coord_system: str = "simulation"
    n_root_blocks: tuple[int, int, int] = (1, 1, 1)
    domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    domain_max: tuple[float, float, float] = (1.0, 1.0, 1.0)
    gamma: float = 5.0 / 3.0
    io_units: str = ""
    normalization_type: str = ""
    body_radius: float | None = None
    body_density_dim: float | None = None
    body_temp_dim: float | None = None
    solar_wind: dict[str, float] = field(default_factory=dict)
    start_time: dict[str, int] = field(default_factory=dict)
    dt_fixed: float | None = None
    geometry: str = "cartesian"
    use_splitb: bool = False
    divb_method: str = ""
    outer_boundary: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Wrap mutable dicts in read-only proxies
        _freeze = MappingProxyType
        object.__setattr__(self, "solar_wind", _freeze(dict(self.solar_wind)))
        object.__setattr__(self, "start_time", _freeze(dict(self.start_time)))
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata)))


def parse_param_in(path: Path) -> BATSRUSConfig:
    """Parse a BATSRUS ``PARAM.in`` file.

    Parameters
    ----------
    path
        Path to the ``PARAM.in`` file.

    Returns
    -------
    BATSRUSConfig
        Frozen dataclass with extracted configuration.
    """
    text = path.read_text()
    lines = text.splitlines()

    description = ""
    coord_system = "simulation"
    n_root = [1, 1, 1]
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]
    gamma = 5.0 / 3.0
    io_units = ""
    normalization_type = ""
    body_radius: float | None = None
    body_density_dim: float | None = None
    body_temp_dim: float | None = None
    solar_wind: dict[str, float] = {}
    start_time: dict[str, int] = {}
    dt_fixed: float | None = None
    use_splitb = False
    divb_method = ""
    outer_boundary: tuple[str, ...] = ()
    metadata: dict[str, Any] = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "#DESCRIPTION" and i + 1 < len(lines):
            description = lines[i + 1].strip()
            i += 2
            continue

        if line == "#COORDSYSTEM" and i + 1 < len(lines):
            coord_system = _first_token(lines[i + 1])
            i += 2
            continue

        if line == "#IOUNITS" and i + 1 < len(lines):
            io_units = _first_token(lines[i + 1])
            i += 2
            continue

        if line == "#NORMALIZATION" and i + 1 < len(lines):
            normalization_type = _first_token(lines[i + 1])
            i += 2
            continue

        if line == "#GAMMA" and i + 1 < len(lines):
            gamma = float(_first_token(lines[i + 1]))
            i += 2
            continue

        if line == "#GRID" and i + 9 < len(lines):
            n_root[0] = int(_first_token(lines[i + 1]))
            n_root[1] = int(_first_token(lines[i + 2]))
            n_root[2] = int(_first_token(lines[i + 3]))
            domain_min[0] = float(_first_token(lines[i + 4]))
            domain_max[0] = float(_first_token(lines[i + 5]))
            domain_min[1] = float(_first_token(lines[i + 6]))
            domain_max[1] = float(_first_token(lines[i + 7]))
            domain_min[2] = float(_first_token(lines[i + 8]))
            domain_max[2] = float(_first_token(lines[i + 9]))
            i += 10
            continue

        if line == "#BODY" and i + 1 < len(lines):
            use_body = _first_token(lines[i + 1])
            if use_body == "T" and i + 4 < len(lines):
                body_radius = float(_first_token(lines[i + 2]))
                # skip rCurrents
                body_density_dim = float(_first_token(lines[i + 4]))
                body_temp_dim = float(_first_token(lines[i + 5]))
                i += 6
            else:
                i += 2
            continue

        if line == "#SOLARWIND" and i + 8 < len(lines):
            solar_wind = {
                "rho_dim": float(_first_token(lines[i + 1])),
                "t_dim": float(_first_token(lines[i + 2])),
                "ux_dim": float(_first_token(lines[i + 3])),
                "uy_dim": float(_first_token(lines[i + 4])),
                "uz_dim": float(_first_token(lines[i + 5])),
                "bx_dim": float(_first_token(lines[i + 6])),
                "by_dim": float(_first_token(lines[i + 7])),
                "bz_dim": float(_first_token(lines[i + 8])),
            }
            i += 9
            continue

        if line == "#STARTTIME" and i + 7 < len(lines):
            start_time = {
                "year": int(_first_token(lines[i + 1])),
                "month": int(_first_token(lines[i + 2])),
                "day": int(_first_token(lines[i + 3])),
                "hour": int(_first_token(lines[i + 4])),
                "minute": int(_first_token(lines[i + 5])),
                "second": int(_first_token(lines[i + 6])),
            }
            i += 8
            continue

        if line == "#FIXEDTIMESTEP" and i + 2 < len(lines):
            use_fixed = _first_token(lines[i + 1])
            if use_fixed == "T":
                dt_fixed = float(_first_token(lines[i + 2]))
            i += 3
            continue

        if line == "#SCHEME" and i + 1 < len(lines):
            n_order = int(_first_token(lines[i + 1]))
            metadata["scheme_order"] = n_order
            if i + 2 < len(lines):
                metadata["flux_type"] = _first_token(lines[i + 2])
            i += 3 + max(0, n_order - 1)
            continue

        if line == "#SPLITB" and i + 1 < len(lines):
            use_splitb = _first_token(lines[i + 1]) == "T"
            i += 2
            continue

        if line == "#DIVB" and i + 1 < len(lines):
            divb_method = _first_token(lines[i + 1])
            i += 2
            continue

        if line == "#OUTERBOUNDARY":
            # Unlike every other command here the face count is variable
            # (4 in 2D, 6 in 3D), so read to the next blank or command line.
            faces: list[str] = []
            j = i + 1
            while j < len(lines) and len(faces) < 6:
                entry = lines[j].strip()
                if not entry or entry.startswith("#"):
                    break
                faces.append(_first_token(entry).lower())
                j += 1
            outer_boundary = tuple(faces)
            i = j
            continue

        i += 1

    return BATSRUSConfig(
        description=description,
        coord_system=coord_system,
        n_root_blocks=tuple(n_root),  # type: ignore[arg-type]
        domain_min=tuple(domain_min),  # type: ignore[arg-type]
        domain_max=tuple(domain_max),  # type: ignore[arg-type]
        gamma=gamma,
        io_units=io_units,
        normalization_type=normalization_type,
        body_radius=body_radius,
        body_density_dim=body_density_dim,
        body_temp_dim=body_temp_dim,
        solar_wind=solar_wind,
        start_time=start_time,
        dt_fixed=dt_fixed,
        geometry="cartesian",
        use_splitb=use_splitb,
        divb_method=divb_method,
        outer_boundary=outer_boundary,
        metadata=metadata,
    )


def to_simulation_config(
    config: BATSRUSConfig,
    header: BATSRUSHeader | None = None,
    *,
    grid: GridInfo | None = None,
    sim_dir: Path | None = None,
) -> SimulationConfig:
    """Build a `SimulationConfig` from BATSRUS config and header.

    If a ``simulation.toml`` exists in *sim_dir*, its normalization, frame,
    transforms, and metadata are merged in via
    `pypic.readers._config_helpers.merge_simulation_toml`.

    Parameters
    ----------
    config
        Parsed ``PARAM.in``.
    header
        Parsed ``.h`` header (provides grid dimensions from actual output).
    grid
        Pre-built GridInfo (overrides header-derived grid).
    sim_dir
        Simulation directory to scan for ``simulation.toml``. ``None`` skips
        the merge.

    Returns
    -------
    SimulationConfig
    """
    geometry = GEOMETRY_BY_NAME.get(config.geometry, CARTESIAN)

    if grid is None:
        if header is not None:
            ndim = header.ndim
            domain_min = header.domain_min
            domain_max = header.domain_max
            dims = _compute_grid_dims(header)
            spacing = tuple(
                (mx - mn) / d
                for mn, mx, d in zip(domain_min, domain_max, dims, strict=True)
            )
        else:
            ndim = 3 if config.n_root_blocks[2] > 1 else 2
            domain_min = config.domain_min[:ndim]
            domain_max = config.domain_max[:ndim]
            # BATSRUS AMR: 8 cells per root block per axis
            dims = tuple(8 * n for n in config.n_root_blocks[:ndim])
            spacing = tuple(
                (mx - mn) / d
                for mn, mx, d in zip(domain_min, domain_max, dims, strict=True)
            )

        grid = GridInfo(
            dimensions=dims,
            spacing=spacing,
            origin=domain_min,
            geometry=geometry,
            dt=config.dt_fixed,
            boundary=boundary_tags(config, header, len(dims)),
        )
    elif grid.boundary is None:
        grid = copy.replace(
            grid, boundary=boundary_tags(config, header, len(grid.dimensions))
        )

    extra: dict[str, Any] = {}
    if config.use_splitb:
        extra["use_splitb"] = True
    if config.divb_method:
        extra["divb_method"] = config.divb_method
    if config.solar_wind:
        extra["solar_wind"] = config.solar_wind
    physics = PhysicsParams(gamma=config.gamma, extra=extra)

    meta: dict[str, Any] = dict(config.metadata)
    if config.outer_boundary:
        # GridInfo holds one tag per axis; the per-face pairs only survive here.
        meta["outer_boundary"] = list(config.outer_boundary)
    if config.start_time:
        meta["start_time"] = config.start_time
    if config.description:
        meta["description"] = config.description

    base = SimulationConfig(
        model_name="BATSRUS",
        model_type="MHD",
        grid=grid,
        # PARAM.in carries no unit block; a simulation.toml supplies one.
        normalization=Normalization.undeclared(),
        physics=physics,
        frame=config.coord_system,
        metadata=meta,
    )
    return merge_simulation_toml(sim_dir, base)


def _first_token(line: str) -> str:
    """Extract the first whitespace-delimited token from a PARAM.in line."""
    return line.split()[0]


def boundary_tags(
    config: BATSRUSConfig | None,
    header: BATSRUSHeader | None,
    ndim: int,
) -> tuple[str, ...] | None:
    """Per-axis boundary tags for `GridInfo`, or None when neither source fits.

    Two sources describe the same thing at different fidelities. The ``.h``
    header's ``#PERIODIC`` is written by the run into the same file that
    supplies the extents, so it cannot disagree about the axis count, but it
    only distinguishes periodic from not. ``PARAM.in``'s ``#OUTERBOUNDARY``
    carries the richer vocabulary (``outflow``, ``float``, ``inflow``, ...)
    and is the only source for the HDF5 and ``.out`` paths, but it is an
    input deck: per-face, and always 3D even when the plot is a 2D slice.

    So the header decides *periodic*, the deck supplies the tag otherwise,
    and a disagreement resolves to ``"open"`` — the run is the truth, and
    claiming periodic would raise a false warning in `pypic.compute`.

    Faces collapse two-to-one per axis, since `GridInfo.boundary` holds one
    tag per axis. An asymmetric pair (``inflow`` against ``outflow``, routine
    in magnetosphere decks) becomes ``"mixed"`` rather than picking a side;
    BATSRUS requires both faces periodic for a wrapped axis, so nothing
    periodic is lost. A source whose length does not match *ndim* is dropped
    rather than truncated — a 3D deck describing a 2D cut has no honest
    mapping onto the surviving axes.
    """
    per_axis: tuple[str, ...] = ()
    if config is not None and len(config.outer_boundary) == 2 * ndim:
        faces = config.outer_boundary
        per_axis = tuple(
            faces[2 * k] if faces[2 * k] == faces[2 * k + 1] else "mixed"
            for k in range(ndim)
        )

    periodic: tuple[bool, ...] = ()
    if header is not None and len(header.is_periodic) == ndim:
        periodic = header.is_periodic

    if not per_axis and not periodic:
        return None

    tags: list[str] = []
    for axis in range(ndim):
        if periodic:
            if periodic[axis]:
                tags.append("periodic")
                continue
            # The run says this axis does not wrap; never echo a deck that
            # claims otherwise.
            tag = per_axis[axis] if per_axis else "open"
            tags.append("open" if tag == "periodic" else tag)
            continue
        tags.append(per_axis[axis] if per_axis else "open")
    return tuple(tags)


def _compute_grid_dims(header: BATSRUSHeader) -> tuple[int, ...]:
    """Compute uniform grid dimensions from header metadata.

    For a uniform grid, dimensions = domain extent / minimum cell size.
    """
    dims: list[int] = []
    for mn, mx, cs in zip(
        header.domain_min, header.domain_max, header.cell_size_min, strict=True
    ):
        dims.append(round((mx - mn) / cs))
    return tuple(dims)
