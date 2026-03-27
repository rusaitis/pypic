"""iPIC3D configuration parsing (.inp files and settings.hdf)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import h5py  # type: ignore[import-untyped]

from pypic.coordinates.geometry import CARTESIAN
from pypic.readers.base import GridInfo, SimulationConfig
from pypic.units import Normalization, SpeciesInfo

if TYPE_CHECKING:
    from pathlib import Path

    from pypic.types import Vector3

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IPic3DConfig:
    """Native iPIC3D simulation parameters.

    Stores the raw values from an ``.inp`` file or ``settings.hdf``,
    before any conversion to the canonical pypic schema.

    Parameters
    ----------
    nxc, nyc, nzc : int
        Number of cells along each axis.
    lx, ly, lz : float
        Domain size along each axis (code units).
    dx, dy, dz : float
        Cell spacing (code units). Computed as ``L / N``.
    dt : float
        Timestep in code units.
    xlen, ylen, zlen : int
        MPI topology (processors per axis).
    c : float
        Speed of light in code units.
    th : float
        Implicitness parameter (0.5 = Crank-Nicolson).
    b0 : tuple[float, float, float]
        Background magnetic field ``(B0x, B0y, B0z)``.
    ns : int
        Number of particle species.
    qom : tuple[float, ...]
        Charge-to-mass ratio per species.
    uth, vth, wth : tuple[float, ...]
        Thermal velocities per species (x, y, z components).
    u0, v0, w0 : tuple[float, ...]
        Drift velocities per species (x, y, z components).
    rho_init : tuple[float, ...]
        Initial number density per species (code units).
    npcelx, npcely, npcelz : tuple[int, ...]
        Particles per cell per species (x, y, z).
    periodic_x, periodic_y, periodic_z : bool
        Periodicity per axis.
    write_method : str
        Output format (``"phdf5"`` or ``"shdf5"``).
    field_output_cycle : int
        Field output frequency (cycles between dumps).
    field_output_tag : str
        Space-separated tags controlling which fields are written.
    particles_output_cycle : int
        Particle output frequency (cycles between dumps; <=0 = disabled).
    case : str
        Simulation case identifier.
    simulation_name : str
        Human-readable simulation name.
    """

    nxc: int
    nyc: int
    nzc: int
    lx: float
    ly: float
    lz: float
    dx: float
    dy: float
    dz: float
    dt: float
    xlen: int
    ylen: int
    zlen: int
    c: float
    th: float
    b0: tuple[float, float, float]
    ns: int
    qom: tuple[float, ...]
    uth: tuple[float, ...]
    vth: tuple[float, ...]
    wth: tuple[float, ...]
    u0: tuple[float, ...]
    v0: tuple[float, ...]
    w0: tuple[float, ...]
    rho_init: tuple[float, ...]
    npcelx: tuple[int, ...]
    npcely: tuple[int, ...]
    npcelz: tuple[int, ...]
    periodic_x: bool
    periodic_y: bool
    periodic_z: bool
    write_method: str
    field_output_cycle: int
    field_output_tag: str
    particles_output_cycle: int
    case: str
    simulation_name: str
    extra: dict[str, Any] = field(default_factory=dict)  # frozen via __post_init__

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))


def _parse_array_float(raw: str) -> tuple[float, ...]:
    """Parse a whitespace-separated array of floats."""
    return tuple(float(x) for x in raw.split())


def _parse_array_int(raw: str) -> tuple[int, ...]:
    """Parse a whitespace-separated array of ints."""
    return tuple(int(float(x)) for x in raw.split())


def parse_inp(path: Path) -> IPic3DConfig:
    """Parse an iPIC3D ``.inp`` configuration file.

    Parameters
    ----------
    path : Path
        Path to the ``.inp`` file.

    Returns
    -------
    IPic3DConfig
        Parsed configuration.

    Raises
    ------
    ExceptionGroup
        If required keys are missing.
    """
    kv: dict[str, str] = {}
    text = path.read_text()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"(\w+)\s*=\s*(.+)", line)
        if match:
            raw_value = match.group(2)
            if "#" in raw_value:
                raw_value = raw_value[: raw_value.index("#")]
            kv[match.group(1)] = raw_value.strip()

    errors: list[Exception] = []

    def require(key: str) -> str:
        if key not in kv:
            errors.append(KeyError(f"Missing required key: {key!r}"))
            return ""
        return kv[key]

    # Grid
    nxc_s = require("nxc")
    nyc_s = require("nyc")
    nzc_s = require("nzc")
    lx_s = require("Lx")
    ly_s = require("Ly")
    lz_s = require("Lz")

    # Time / physics
    dt_s = require("dt")
    c_s = require("c")
    th_s = require("th")

    # MPI
    xlen_s = require("XLEN")
    ylen_s = require("YLEN")
    zlen_s = require("ZLEN")

    # Species
    ns_s = require("ns")
    qom_s = require("qom")

    if errors:
        raise ExceptionGroup("Missing keys in iPIC3D .inp file", errors)

    nxc = int(nxc_s)
    nyc = int(nyc_s)
    nzc = int(nzc_s)
    lx = float(lx_s)
    ly = float(ly_s)
    lz = float(lz_s)
    ns = int(ns_s)

    qom = _parse_array_float(qom_s)
    if len(qom) < ns:
        errors.append(ValueError(f"qom has {len(qom)} entries, expected ns={ns}"))
    elif len(qom) > ns:
        log.warning("qom has %d entries but ns=%d; truncating", len(qom), ns)
        qom = qom[:ns]
    _per_species_keys = (
        "uth", "vth", "wth", "u0", "v0", "w0",
        "rhoINIT", "rhoINJECT", "npcelx", "npcely", "npcelz",
    )
    for key in _per_species_keys:
        if key in kv:
            n = len(kv[key].split())
            if n < ns:
                errors.append(
                    ValueError(f"{key} has {n} entries, expected ns={ns}")
                )
            elif n > ns:
                log.warning(
                    "%s has %d entries but ns=%d; truncating", key, n, ns
                )
                kv[key] = " ".join(kv[key].split()[:ns])
    if errors:
        raise ExceptionGroup("Validation errors in iPIC3D .inp file", errors)

    return IPic3DConfig(
        nxc=nxc,
        nyc=nyc,
        nzc=nzc,
        lx=lx,
        ly=ly,
        lz=lz,
        dx=lx / nxc,
        dy=ly / nyc,
        dz=lz / nzc,
        dt=float(dt_s),
        xlen=int(xlen_s),
        ylen=int(ylen_s),
        zlen=int(zlen_s),
        c=float(c_s),
        th=float(th_s),
        b0=(
            float(kv.get("B0x", "0.0")),
            float(kv.get("B0y", "0.0")),
            float(kv.get("B0z", "0.0")),
        ),
        ns=ns,
        qom=qom,
        uth=_parse_array_float(kv.get("uth", " ".join(["0.0"] * ns))),
        vth=_parse_array_float(kv.get("vth", " ".join(["0.0"] * ns))),
        wth=_parse_array_float(kv.get("wth", " ".join(["0.0"] * ns))),
        u0=_parse_array_float(kv.get("u0", " ".join(["0.0"] * ns))),
        v0=_parse_array_float(kv.get("v0", " ".join(["0.0"] * ns))),
        w0=_parse_array_float(kv.get("w0", " ".join(["0.0"] * ns))),
        rho_init=_parse_array_float(kv.get("rhoINIT", " ".join(["1.0"] * ns))),
        npcelx=_parse_array_int(kv.get("npcelx", " ".join(["0"] * ns))),
        npcely=_parse_array_int(kv.get("npcely", " ".join(["0"] * ns))),
        npcelz=_parse_array_int(kv.get("npcelz", " ".join(["0"] * ns))),
        periodic_x=kv.get("PERIODICX", "0") == "1",
        periodic_y=kv.get("PERIODICY", "0") == "1",
        periodic_z=kv.get("PERIODICZ", "0") == "1",
        write_method=kv.get("WriteMethod", "phdf5"),
        field_output_cycle=int(kv.get("FieldOutputCycle", "0")),
        field_output_tag=kv.get("FieldOutputTag", ""),
        particles_output_cycle=int(kv.get("ParticlesOutputCycle", "0")),
        case=kv.get("Case", ""),
        simulation_name=kv.get("SimulationName", ""),
    )


def parse_settings_hdf(path: Path) -> IPic3DConfig:
    """Parse an iPIC3D ``settings.hdf`` file (serial format metadata).

    Parameters
    ----------
    path : Path
        Path to the ``settings.hdf`` file.

    Returns
    -------
    IPic3DConfig
        Parsed configuration (equivalent to `parse_inp` output).
    """
    with h5py.File(path, "r") as f:
        col = f["collective"]
        topo = f["topology"]

        nxc = int(col["Nxc"][()].item())
        nyc = int(col["Nyc"][()].item())
        nzc = int(col["Nzc"][()].item())
        lx = float(col["Lx"][()].item())
        ly = float(col["Ly"][()].item())
        lz = float(col["Lz"][()].item())
        ns = int(col["Ns"][()].item())
        xlen = int(topo["XLEN"][()].item())
        ylen = int(topo["YLEN"][()].item())
        zlen = int(topo["ZLEN"][()].item())

        def _read_scalar(
            group: Any,  # noqa: ANN401 — h5py Group lacks typed protocol
            key: str,
            default: float = 0.0,
        ) -> float:
            if key in group:
                return float(group[key][()].item())
            return default

        qom: list[float] = []
        uth: list[float] = []
        vth: list[float] = []
        wth: list[float] = []
        u0: list[float] = []
        v0: list[float] = []
        w0: list[float] = []
        npcelx: list[int] = []
        npcely: list[int] = []
        npcelz: list[int] = []

        for s in range(ns):
            sp = col[f"species_{s}"]
            qom.append(float(sp["qom"][()].item()))
            uth.append(float(sp["uth"][()].item()))
            vth.append(float(sp["vth"][()].item()))
            wth.append(float(sp["wth"][()].item()))
            u0.append(float(sp["u0"][()].item()))
            v0.append(float(sp["v0"][()].item()))
            w0.append(float(sp["w0"][()].item()))
            npcelx.append(int(sp["Npcelx"][()].item()))
            npcely.append(int(sp["Npcely"][()].item()))
            npcelz.append(int(sp["Npcelz"][()].item()))

        is_periodic_x = int(topo["periodicX"][()].item()) == 1
        is_periodic_y = int(topo["periodicY"][()].item()) == 1
        is_periodic_z = int(topo["periodicZ"][()].item()) == 1

        dt = _read_scalar(col, "Dt")
        c = _read_scalar(col, "c", 1.0)
        th = _read_scalar(col, "Th", 0.5)
        b0 = (
            _read_scalar(col, "Bx0"),
            _read_scalar(col, "By0"),
            _read_scalar(col, "Bz0"),
        )

    return IPic3DConfig(
        nxc=nxc,
        nyc=nyc,
        nzc=nzc,
        lx=lx,
        ly=ly,
        lz=lz,
        dx=lx / nxc,
        dy=ly / nyc,
        dz=lz / nzc,
        dt=dt,
        xlen=xlen,
        ylen=ylen,
        zlen=zlen,
        c=c,
        th=th,
        b0=b0,
        ns=ns,
        qom=tuple(qom),
        uth=tuple(uth),
        vth=tuple(vth),
        wth=tuple(wth),
        u0=tuple(u0),
        v0=tuple(v0),
        w0=tuple(w0),
        rho_init=(1.0,) * ns,  # not stored in settings.hdf
        npcelx=tuple(npcelx),
        npcely=tuple(npcely),
        npcelz=tuple(npcelz),
        periodic_x=is_periodic_x,
        periodic_y=is_periodic_y,
        periodic_z=is_periodic_z,
        write_method="shdf5",  # settings.hdf only exists for serial output
        field_output_cycle=0,  # not stored in settings.hdf
        field_output_tag="",  # not stored in settings.hdf
        particles_output_cycle=0,  # not stored in settings.hdf
        case="",
        simulation_name="",
    )


def _build_species(cfg: IPic3DConfig) -> tuple[SpeciesInfo, ...]:
    """Build SpeciesInfo objects from iPIC3D config.

    iPIC3D defines species by ``qom`` (charge-to-mass ratio). Convention:
    ``|q| = 1``, ``m = 1/|qom|``, ``sign(q) = sign(qom)``.
    """
    species: list[SpeciesInfo] = []
    for s in range(cfg.ns):
        thermal_vel: Vector3 = (cfg.uth[s], cfg.vth[s], cfg.wth[s])
        drift_vel: Vector3 = (cfg.u0[s], cfg.v0[s], cfg.w0[s])

        species.append(
            SpeciesInfo(
                name=f"species_{s}",
                charge_to_mass=cfg.qom[s],
                thermal_velocity=thermal_vel,
                drift_velocity=drift_vel,
                density=cfg.rho_init[s],
                particles_per_cell=(cfg.npcelx[s], cfg.npcely[s], cfg.npcelz[s]),
            )
        )
    return tuple(species)


def to_simulation_config(cfg: IPic3DConfig) -> SimulationConfig:
    """Convert iPIC3D config to the canonical SimulationConfig.

    Uses the node-centered origin offset trick: ``origin = -dx/2`` so that
    ``coordinate_arrays()`` produces exact node positions ``0, dx, 2dx, ..., L``.

    Parameters
    ----------
    cfg : IPic3DConfig
        Native iPIC3D configuration.

    Returns
    -------
    SimulationConfig
        Canonical simulation configuration.
    """
    boundary_map = {True: "periodic", False: "open"}
    grid = GridInfo(
        dimensions=(cfg.nxc + 1, cfg.nyc + 1, cfg.nzc + 1),
        spacing=(cfg.dx, cfg.dy, cfg.dz),
        origin=(-cfg.dx / 2, -cfg.dy / 2, -cfg.dz / 2),
        geometry=CARTESIAN,
        dt=cfg.dt,
        boundary=(
            boundary_map[cfg.periodic_x],
            boundary_map[cfg.periodic_y],
            boundary_map[cfg.periodic_z],
        ),
    )

    physics: dict[str, Any] = {
        "theta": cfg.th,
        "c": cfg.c,
        "b0": cfg.b0,
    }
    physics.update(cfg.extra)

    metadata: dict[str, Any] = {
        "grid_centering": "node",
        "write_method": cfg.write_method,
    }
    if cfg.case:
        metadata["case"] = cfg.case
    if cfg.simulation_name:
        metadata["simulation_name"] = cfg.simulation_name
    if cfg.field_output_cycle > 0:
        metadata["field_output_cycle"] = cfg.field_output_cycle
    if cfg.field_output_tag:
        metadata["field_output_tag"] = cfg.field_output_tag
    if cfg.particles_output_cycle > 0:
        metadata["particles_output_cycle"] = cfg.particles_output_cycle

    return SimulationConfig(
        model_name="iPIC3D",
        model_type="PIC",
        grid=grid,
        normalization=Normalization.identity(),
        species=_build_species(cfg),
        physics=physics,
        metadata=metadata,
    )


def to_toml(cfg: IPic3DConfig) -> str:
    """Generate a ``simulation.toml`` string from iPIC3D config.

    Parameters
    ----------
    cfg : IPic3DConfig
        Native iPIC3D configuration.

    Returns
    -------
    str
        TOML content conforming to SCHEMA.md.
    """
    sim_config = to_simulation_config(cfg)
    lines: list[str] = []

    lines.append("[model]")
    lines.append('name = "iPIC3D"')
    lines.append('type = "PIC"')
    if cfg.simulation_name:
        lines.append(f'description = "{cfg.simulation_name}"')
    lines.append("")

    g = sim_config.grid
    d = g.dimensions
    lines.append("[grid]")
    lines.append(f"dimensions = [{d[0]}, {d[1]}, {d[2]}]")
    lines.append(f"spacing = [{g.spacing[0]}, {g.spacing[1]}, {g.spacing[2]}]")
    lines.append(f"origin = [{g.origin[0]}, {g.origin[1]}, {g.origin[2]}]")
    lines.append(f"dt = {cfg.dt}")
    assert g.boundary is not None
    bnd = ", ".join(f'"{b}"' for b in g.boundary)
    lines.append(f"boundary = [{bnd}]")
    lines.append("")

    lines.append("[units]")
    lines.append('system = "PIC"')
    lines.append("")

    lines.append("[coordinates]")
    lines.append('geometry = "cartesian"')
    lines.append('frame = "simulation"')
    lines.append("")

    for s in range(cfg.ns):
        sp = sim_config.species[s]
        lines.append("[[species]]")
        lines.append(f'name = "{sp.name}"')
        lines.append(f"charge_to_mass = {cfg.qom[s]}")
        tv = (cfg.uth[s], cfg.vth[s], cfg.wth[s])
        lines.append(f"thermal_velocity = [{tv[0]}, {tv[1]}, {tv[2]}]")
        dv = (cfg.u0[s], cfg.v0[s], cfg.w0[s])
        lines.append(f"drift_velocity = [{dv[0]}, {dv[1]}, {dv[2]}]")
        lines.append(f"density = {cfg.rho_init[s]}")
        ppc = (cfg.npcelx[s], cfg.npcely[s], cfg.npcelz[s])
        lines.append(f"particles_per_cell = [{ppc[0]}, {ppc[1]}, {ppc[2]}]")
        lines.append("")

    lines.append("[physics.pic]")
    lines.append(f"theta = {cfg.th}")
    lines.append(f"speed_of_light = {cfg.c}")
    lines.append("")

    lines.append("[initial_conditions]")
    if cfg.case:
        lines.append(f'type = "{cfg.case}"')
    lines.append(f"B0 = [{cfg.b0[0]}, {cfg.b0[1]}, {cfg.b0[2]}]")
    lines.append("")

    return "\n".join(lines) + "\n"
