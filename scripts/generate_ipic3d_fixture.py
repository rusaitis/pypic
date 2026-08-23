"""Generate synthetic iPIC3D test fixtures for all three reader formats.

Creates a tiny 4×4×2 grid, 2 species (electron + ion), with analytically
known Double Harris values where all 4π corrections and sign flips are
exactly predictable.  Output committed to ``tests/data/ipic3d-synthetic/``
for fast CI tests with machine-precision assertions.

Embedded values match iPIC3D's internal storage convention:
- EM fields: stored as-is (no 4π)
- rho, J: stored as value/(4π) (Gaussian convention)
- Pressure: stored as value/(4π), with charge sign on diagonal for electrons

Usage::

    uv run python scripts/generate_ipic3d_fixture.py
"""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = _REPO_ROOT / "tests" / "data" / "ipic3d-synthetic"

NXC, NYC, NZC = 4, 4, 2
LX, LY, LZ = 4.0, 4.0, 2.0
DX, DY, DZ = LX / NXC, LY / NYC, LZ / NZC
DT = 0.1
C = 1.0
TH = 0.5
B0X = 0.1
NS = 2
QOM = (-64.0, 1.0)
UTH = (0.04, 0.005)
VTH = (0.02, 0.005)
WTH = (0.02, 0.005)
U0 = (0.0, 0.0)
V0 = (0.0, 0.0)
W0 = (0.001, -0.064)
RHO_INIT = (1.0, 1.0)
NPCELX = (3, 3)
NPCELY = (3, 3)
NPCELZ = (1, 1)

FOUR_PI = 4.0 * math.pi

# Node-centered shape
NX, NY, NZ = NXC + 1, NYC + 1, NZC + 1


def _node_coords() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Node coordinates (the reader reconstructs these from origin+spacing)."""
    x = np.linspace(0, LX, NX)
    y = np.linspace(0, LY, NY)
    z = np.linspace(0, LZ, NZ)
    return x, y, z


def _analytical_fields() -> dict[str, np.ndarray]:
    """Build analytically known fields in SI-rationalized normalization.

    These are the *correct* values after all reader corrections.
    """
    x, y, z = _node_coords()
    _, Y, _ = np.meshgrid(x, y, z, indexing="ij")

    fields: dict[str, np.ndarray] = {}

    # B field: tanh profile (Double Harris)
    delta = LY / 8.0
    y_mid = LY / 2.0
    fields["B1"] = B0X * np.tanh((Y - y_mid) / delta)
    fields["B2"] = np.zeros((NX, NY, NZ))
    fields["B3"] = np.zeros((NX, NY, NZ))

    # E field: zero at t=0
    fields["E1"] = np.zeros((NX, NY, NZ))
    fields["E2"] = np.zeros((NX, NY, NZ))
    fields["E3"] = np.zeros((NX, NY, NZ))

    # Per-species density: uniform background
    # Species 0 = electrons (qom < 0), species 1 = ions (qom > 0)
    # Charge density rho_c = n * q, and with |q|=1: rho_c_e = -1, rho_c_i = +1
    fields["rho_c_s0"] = np.full((NX, NY, NZ), -RHO_INIT[0])
    fields["rho_c_s1"] = np.full((NX, NY, NZ), RHO_INIT[1])

    # Per-species current density (small drift)
    for s in range(NS):
        sign = -1.0 if QOM[s] < 0 else 1.0
        rho = sign * RHO_INIT[s]
        fields[f"J1_s{s}"] = np.full((NX, NY, NZ), rho * U0[s])
        fields[f"J2_s{s}"] = np.full((NX, NY, NZ), rho * V0[s])
        fields[f"J3_s{s}"] = np.full((NX, NY, NZ), rho * W0[s])

    # Pressure tensor: P_ij = rho_c * v_th_i * v_th_j (charge-weighted)
    # After reader correction, diagonal should be positive (|rho_c| * v_th²)
    for s in range(NS):
        sign = -1.0 if QOM[s] < 0 else 1.0
        abs_rho = RHO_INIT[s]  # magnitude
        vths = (UTH[s], VTH[s], WTH[s])
        fields[f"P11_s{s}"] = np.full((NX, NY, NZ), abs_rho * vths[0] ** 2)
        fields[f"P22_s{s}"] = np.full((NX, NY, NZ), abs_rho * vths[1] ** 2)
        fields[f"P33_s{s}"] = np.full((NX, NY, NZ), abs_rho * vths[2] ** 2)
        fields[f"P12_s{s}"] = np.zeros((NX, NY, NZ))
        fields[f"P13_s{s}"] = np.zeros((NX, NY, NZ))
        fields[f"P23_s{s}"] = np.zeros((NX, NY, NZ))

    # Energy flux: EF = rho * v * v_th² (per-species, simplified)
    for s in range(NS):
        abs_rho = RHO_INIT[s]
        vths = (UTH[s], VTH[s], WTH[s])
        drifts = (U0[s], V0[s], W0[s])
        for i, (v_th, drift) in enumerate(zip(vths, drifts, strict=True)):
            fields[f"EF{i + 1}_s{s}"] = np.full((NX, NY, NZ), abs_rho * drift * v_th**2)

    # Totals
    fields["rho_c"] = fields["rho_c_s0"] + fields["rho_c_s1"]
    for comp in ("J1", "J2", "J3"):
        fields[comp] = fields[f"{comp}_s0"] + fields[f"{comp}_s1"]

    return fields


def _to_gaussian_storage(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convert SI-rationalized values to iPIC3D Gaussian storage format.

    EM fields: unchanged
    rho, J: divide by 4π
    Pressure diagonal: divide by 4π, then negate for electrons (qom < 0)
    Pressure off-diag: divide by 4π
    """
    stored: dict[str, np.ndarray] = {}

    # EM fields stored as-is
    for name in ("B1", "B2", "B3", "E1", "E2", "E3"):
        stored[name] = fields[name].copy()

    for s in range(NS):
        # Density and current: store as value / 4π
        stored[f"rho_c_s{s}"] = fields[f"rho_c_s{s}"] / FOUR_PI
        for comp in ("J1", "J2", "J3"):
            stored[f"{comp}_s{s}"] = fields[f"{comp}_s{s}"] / FOUR_PI

        # Pressure: store as value / 4π, with sign flip on diagonal for electrons
        for pcomp in ("P11", "P22", "P33"):
            val = fields[f"{pcomp}_s{s}"] / FOUR_PI
            if QOM[s] < 0:
                val = -val  # electrons store negative diagonal
            stored[f"{pcomp}_s{s}"] = val
        for pcomp in ("P12", "P13", "P23"):
            stored[f"{pcomp}_s{s}"] = fields[f"{pcomp}_s{s}"] / FOUR_PI

        # Energy flux: store as value / 4π
        for i in range(1, 4):
            stored[f"EF{i}_s{s}"] = fields[f"EF{i}_s{s}"] / FOUR_PI

    return stored


def _write_inp(path: Path) -> None:
    """Write a minimal .inp config file."""
    lines = [
        "Case                           = SyntheticTest",
        "SimulationName                 = SyntheticFixture",
        "WriteMethod                    = phdf5",
        "FieldOutputCycle               = 1",
        "FieldOutputTag                 = B + E + rho + J + J_s + rho_s + pressure + E_flux",
        "ParticlesOutputCycle           = 10",
        f"B0x                            = {B0X}",
        "B0y                            = 0.0",
        "B0z                            = 0.0",
        f"dt                             = {DT}",
        "ncycles                        = 1",
        f"th                             = {TH}",
        f"c                              = {C}",
        f"Lx                             = {LX}",
        f"Ly                             = {LY}",
        f"Lz                             = {LZ}",
        f"nxc                            = {NXC}",
        f"nyc                            = {NYC}",
        f"nzc                            = {NZC}",
        "XLEN                           = 1",
        "YLEN                           = 1",
        "ZLEN                           = 1",
        f"ns                  = {NS}",
        f"rhoINIT             = {' '.join(str(r) for r in RHO_INIT)}",
        f"npcelx              = {' '.join(str(n) for n in NPCELX)}",
        f"npcely              = {' '.join(str(n) for n in NPCELY)}",
        f"npcelz              = {' '.join(str(n) for n in NPCELZ)}",
        f"qom                 = {' '.join(str(q) for q in QOM)}",
        f"uth                 = {' '.join(str(u) for u in UTH)}",
        f"vth                 = {' '.join(str(v) for v in VTH)}",
        f"wth                 = {' '.join(str(w) for w in WTH)}",
        f"u0                  = {' '.join(str(u) for u in U0)}",
        f"v0                  = {' '.join(str(v) for v in V0)}",
        f"w0                  = {' '.join(str(w) for w in W0)}",
        "PERIODICX                      = 1",
        "PERIODICY                      = 1",
        "PERIODICZ                      = 1",
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_phdf5(base: Path, stored: dict[str, np.ndarray]) -> None:
    """Write phdf5-format output: Fields_00000/ and Moments_00000/ dirs."""
    ipic_b_map = {"B1": "Bx", "B2": "By", "B3": "Bz"}
    ipic_e_map = {"E1": "Ex", "E2": "Ey", "E3": "Ez"}
    ipic_j_map = {"J1": "Jx", "J2": "Jy", "J3": "Jz"}
    ipic_p_map = {
        "P11": "pXX",
        "P12": "pXY",
        "P13": "pXZ",
        "P22": "pYY",
        "P23": "pYZ",
        "P33": "pZZ",
    }

    fields_dir = base / "Fields_00000"
    moments_dir = base / "Moments_00000"
    fields_dir.mkdir(parents=True, exist_ok=True)
    moments_dir.mkdir(parents=True, exist_ok=True)

    # B field file
    with h5py.File(fields_dir / "B_00000.h5", "w") as f:
        g = f.create_group("Fields")
        for canon, ipic in ipic_b_map.items():
            g.create_dataset(ipic, data=stored[canon])

    # E field file
    with h5py.File(fields_dir / "E_00000.h5", "w") as f:
        g = f.create_group("Fields")
        for canon, ipic in ipic_e_map.items():
            g.create_dataset(ipic, data=stored[canon])

    # Per-species moments
    for s in range(NS):
        # Current density
        with h5py.File(moments_dir / f"J_species_{s}_00000.h5", "w") as f:
            g = f.create_group(f"Moments/species_{s}")
            for canon, ipic in ipic_j_map.items():
                g.create_dataset(ipic, data=stored[f"{canon}_s{s}"])

        # Charge density
        with h5py.File(moments_dir / f"rho_species_{s}_00000.h5", "w") as f:
            g = f.create_group(f"Moments/species_{s}")
            g.create_dataset("rho", data=stored[f"rho_c_s{s}"])

        # Pressure tensor
        with h5py.File(moments_dir / f"Pressure_species_{s}_00000.h5", "w") as f:
            g = f.create_group(f"Moments/species_{s}")
            for canon, ipic in ipic_p_map.items():
                g.create_dataset(ipic, data=stored[f"{canon}_s{s}"])

        # Energy flux
        with h5py.File(moments_dir / f"E_flux_species_{s}_00000.h5", "w") as f:
            g = f.create_group(f"Moments/species_{s}")
            ipic_ef_map = {"EF1": "EFx", "EF2": "EFy", "EF3": "EFz"}
            for canon, ipic in ipic_ef_map.items():
                g.create_dataset(ipic, data=stored[f"{canon}_s{s}"])


def _write_shdf5(base: Path, stored: dict[str, np.ndarray]) -> None:
    """Write shdf5-format output: 2 proc files (XLEN=2, YLEN=1, ZLEN=1).

    Also writes settings.hdf with topology and species metadata.
    """
    xlen, ylen, zlen = 2, 1, 1
    # Each proc gets half the x-domain

    for proc in range(xlen * ylen * zlen):
        ix = proc % xlen
        iy = (proc // xlen) % ylen
        iz = proc // (xlen * ylen)

        x0 = ix * (NXC // xlen)
        nx_loc = NXC // xlen + 1 if ix < xlen - 1 else NX - x0

        with h5py.File(base / f"proc{proc}.hdf", "w") as f:
            # Topology
            topo = f.create_group("topology")
            topo.create_dataset("cartesian_coord", data=np.array([ix, iy, iz]))

            # Extract local patch
            local_slice = slice(x0, x0 + nx_loc)

            # EM fields
            ipic_map = {
                "B1": "Bx",
                "B2": "By",
                "B3": "Bz",
                "E1": "Ex",
                "E2": "Ey",
                "E3": "Ez",
            }
            for canon, ipic in ipic_map.items():
                g = f.create_group(f"fields/{ipic}")
                g.create_dataset("cycle_0", data=stored[canon][local_slice, :, :])

            # Per-species moments
            for s in range(NS):
                for comp, ipic in {"J1": "Jx", "J2": "Jy", "J3": "Jz"}.items():
                    g = f.create_group(f"moments/species_{s}/{ipic}")
                    g.create_dataset(
                        "cycle_0", data=stored[f"{comp}_s{s}"][local_slice, :, :]
                    )
                g = f.create_group(f"moments/species_{s}/rho")
                g.create_dataset(
                    "cycle_0", data=stored[f"rho_c_s{s}"][local_slice, :, :]
                )

                # Pressure tensor
                ipic_p = {
                    "P11": "pXX",
                    "P12": "pXY",
                    "P13": "pXZ",
                    "P22": "pYY",
                    "P23": "pYZ",
                    "P33": "pZZ",
                }
                for canon, ipic in ipic_p.items():
                    g = f.create_group(f"moments/species_{s}/{ipic}")
                    g.create_dataset(
                        "cycle_0", data=stored[f"{canon}_s{s}"][local_slice, :, :]
                    )

                # Energy flux
                ipic_ef = {"EF1": "EFx", "EF2": "EFy", "EF3": "EFz"}
                for canon, ipic in ipic_ef.items():
                    g = f.create_group(f"moments/species_{s}/{ipic}")
                    g.create_dataset(
                        "cycle_0", data=stored[f"{canon}_s{s}"][local_slice, :, :]
                    )

    # Write settings.hdf
    with h5py.File(base / "settings.hdf", "w") as f:
        col = f.create_group("collective")
        col.create_dataset("Nxc", data=np.array([NXC]))
        col.create_dataset("Nyc", data=np.array([NYC]))
        col.create_dataset("Nzc", data=np.array([NZC]))
        col.create_dataset("Lx", data=np.array([LX]))
        col.create_dataset("Ly", data=np.array([LY]))
        col.create_dataset("Lz", data=np.array([LZ]))
        col.create_dataset("Ns", data=np.array([NS]))
        col.create_dataset("Dt", data=np.array([DT]))
        col.create_dataset("c", data=np.array([C]))
        col.create_dataset("Th", data=np.array([TH]))
        col.create_dataset("Bx0", data=np.array([B0X]))
        col.create_dataset("By0", data=np.array([0.0]))
        col.create_dataset("Bz0", data=np.array([0.0]))

        for s in range(NS):
            sp = col.create_group(f"species_{s}")
            sp.create_dataset("qom", data=np.array([QOM[s]]))
            sp.create_dataset("uth", data=np.array([UTH[s]]))
            sp.create_dataset("vth", data=np.array([VTH[s]]))
            sp.create_dataset("wth", data=np.array([WTH[s]]))
            sp.create_dataset("u0", data=np.array([U0[s]]))
            sp.create_dataset("v0", data=np.array([V0[s]]))
            sp.create_dataset("w0", data=np.array([W0[s]]))
            sp.create_dataset("Npcelx", data=np.array([NPCELX[s]]))
            sp.create_dataset("Npcely", data=np.array([NPCELY[s]]))
            sp.create_dataset("Npcelz", data=np.array([NPCELZ[s]]))

        topo = f.create_group("topology")
        topo.create_dataset("XLEN", data=np.array([xlen]))
        topo.create_dataset("YLEN", data=np.array([ylen]))
        topo.create_dataset("ZLEN", data=np.array([zlen]))
        topo.create_dataset("periodicX", data=np.array([1]))
        topo.create_dataset("periodicY", data=np.array([1]))
        topo.create_dataset("periodicZ", data=np.array([1]))

    # Also write a .inp for the serial format (XLEN=2)
    lines = [
        "Case                           = SyntheticTest",
        "SimulationName                 = SyntheticFixture",
        "WriteMethod                    = shdf5",
        "FieldOutputCycle               = 1",
        f"B0x                            = {B0X}",
        "B0y                            = 0.0",
        "B0z                            = 0.0",
        f"dt                             = {DT}",
        "ncycles                        = 1",
        f"th                             = {TH}",
        f"c                              = {C}",
        f"Lx                             = {LX}",
        f"Ly                             = {LY}",
        f"Lz                             = {LZ}",
        f"nxc                            = {NXC}",
        f"nyc                            = {NYC}",
        f"nzc                            = {NZC}",
        f"XLEN                           = {xlen}",
        f"YLEN                           = {ylen}",
        f"ZLEN                           = {zlen}",
        f"ns                  = {NS}",
        f"rhoINIT             = {' '.join(str(r) for r in RHO_INIT)}",
        f"npcelx              = {' '.join(str(n) for n in NPCELX)}",
        f"npcely              = {' '.join(str(n) for n in NPCELY)}",
        f"npcelz              = {' '.join(str(n) for n in NPCELZ)}",
        f"qom                 = {' '.join(str(q) for q in QOM)}",
        f"uth                 = {' '.join(str(u) for u in UTH)}",
        f"vth                 = {' '.join(str(v) for v in VTH)}",
        f"wth                 = {' '.join(str(w) for w in WTH)}",
        f"u0                  = {' '.join(str(u) for u in U0)}",
        f"v0                  = {' '.join(str(v) for v in V0)}",
        f"w0                  = {' '.join(str(w) for w in W0)}",
        "PERIODICX                      = 1",
        "PERIODICY                      = 1",
        "PERIODICZ                      = 1",
    ]
    (base / "synthetic_serial.inp").write_text("\n".join(lines) + "\n")


def _write_h5hut(base: Path, stored: dict[str, np.ndarray]) -> None:
    """Write H5hut-format output: single file with ZYX ordering, float32.

    Also includes Vfx/divB fields and per-species pressure.
    """
    ipic_b = {"B1": "Bx", "B2": "By", "B3": "Bz"}
    ipic_e = {"E1": "Ex", "E2": "Ey", "E3": "Ez"}
    ipic_j = {"J1": "Jx", "J2": "Jy", "J3": "Jz"}
    ipic_p = {
        "P11": "Pxx",
        "P12": "Pxy",
        "P13": "Pxz",
        "P22": "Pyy",
        "P23": "Pyz",
        "P33": "Pzz",
    }

    h5_path = base / "SyntheticFixture-Fields_000000.h5"
    with h5py.File(h5_path, "w") as f:
        step = f.create_group("Step#0")
        step.attrs["nspec"] = np.array([NS])
        block = step.create_group("Block")

        def _write_field(
            name: str, data: np.ndarray, *, use_float32: bool = True
        ) -> None:
            # H5hut stores in ZYX order
            transposed = data.transpose(2, 1, 0)
            if use_float32:
                transposed = transposed.astype(np.float32)
            g = block.create_group(name)
            g.create_dataset("0", data=transposed)

        # EM fields (stored as-is, no 4π)
        for canon, ipic in ipic_b.items():
            _write_field(ipic, stored[canon])
        for canon, ipic in ipic_e.items():
            _write_field(ipic, stored[canon])

        # Per-species moments (stored as value / 4π)
        for s in range(NS):
            _write_field(f"rho_{s}", stored[f"rho_c_s{s}"])
            for canon, ipic in ipic_j.items():
                _write_field(f"{ipic}_{s}", stored[f"{canon}_s{s}"])
            for canon, ipic in ipic_p.items():
                _write_field(f"{ipic}_{s}", stored[f"{canon}_s{s}"])
            # Energy flux
            for i, ef_ipic in enumerate(("EFx", "EFy", "EFz"), 1):
                _write_field(f"{ef_ipic}_{s}", stored[f"EF{i}_s{s}"])

        # H5hut-specific fields: Vfx/Vfy/Vfz and divB
        # Fluid velocity = J / rho_c (or just use known values)
        vfx = np.zeros((NX, NY, NZ))
        vfy = np.zeros((NX, NY, NZ))
        vfz = np.zeros((NX, NY, NZ))
        _write_field("Vfx", vfx)
        _write_field("Vfy", vfy)
        _write_field("Vfz", vfz)

        # divB should be ~0 for our tanh profile
        _write_field("divB", np.zeros((NX, NY, NZ)))

    # Write .inp for h5hut (write_method = h5hut, but detection is file-based)
    lines = [
        "Case                           = SyntheticTest",
        "SimulationName                 = SyntheticFixture",
        "WriteMethod                    = h5hut  # uses H5hut format",
        "FieldOutputCycle               = 1",
        f"B0x                            = {B0X}",
        "B0y                            = 0.0",
        "B0z                            = 0.0",
        f"dt                             = {DT}",
        "ncycles                        = 1",
        f"th                             = {TH}",
        f"c                              = {C}",
        f"Lx                             = {LX}",
        f"Ly                             = {LY}",
        f"Lz                             = {LZ}",
        f"nxc                            = {NXC}",
        f"nyc                            = {NYC}",
        f"nzc                            = {NZC}",
        "XLEN                           = 1",
        "YLEN                           = 1",
        "ZLEN                           = 1",
        f"ns                  = {NS}",
        f"rhoINIT             = {' '.join(str(r) for r in RHO_INIT)}",
        f"npcelx              = {' '.join(str(n) for n in NPCELX)}",
        f"npcely              = {' '.join(str(n) for n in NPCELY)}",
        f"npcelz              = {' '.join(str(n) for n in NPCELZ)}",
        f"qom                 = {' '.join(str(q) for q in QOM)}",
        f"uth                 = {' '.join(str(u) for u in UTH)}",
        f"vth                 = {' '.join(str(v) for v in VTH)}",
        f"wth                 = {' '.join(str(w) for w in WTH)}",
        f"u0                  = {' '.join(str(u) for u in U0)}",
        f"v0                  = {' '.join(str(v) for v in V0)}",
        f"w0                  = {' '.join(str(w) for w in W0)}",
        "PERIODICX                      = 1",
        "PERIODICY                      = 1",
        "PERIODICZ                      = 1",
    ]
    (base / "SyntheticFixture.inp").write_text("\n".join(lines) + "\n")


def _write_phdf5_particles(base: Path) -> None:
    """Write phdf5 particle output: Particles_00000/species_{s}_00000.h5.

    Creates 18 particles per species (3x3x2 sub-grid within the first cell),
    with deterministic positions and velocities from a fixed seed.
    """
    rng = np.random.default_rng(seed=42)
    particles_dir = base / "phdf5" / "Particles_00000"
    particles_dir.mkdir(parents=True, exist_ok=True)

    n_particles = 18  # 3 x 3 x 2

    for s in range(NS):
        # Deterministic positions: uniform sub-grid within first cell
        px = np.linspace(0.1, DX - 0.1, 3)
        py = np.linspace(0.1, DY - 0.1, 3)
        pz = np.linspace(0.1, DZ - 0.1, 2)
        gx, gy, gz = np.meshgrid(px, py, pz, indexing="ij")
        position = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

        # Deterministic velocities: small random from seed
        velocity = rng.standard_normal((n_particles, 3)) * UTH[s]

        # Charge: scalar value per species (sign(qom) * 1.0)
        q_scalar = -1.0 if QOM[s] < 0 else 1.0

        # Integer tracking IDs
        particle_ids = np.arange(s * n_particles, (s + 1) * n_particles, dtype=np.int64)

        h5_path = particles_dir / f"species_{s}_00000.h5"
        with h5py.File(h5_path, "w") as f:
            g = f.create_group(f"Particles/species_{s}")
            g.create_dataset("position", data=position)
            g.create_dataset("velocity", data=velocity)
            g.create_dataset("q", data=np.array([[q_scalar]]))
            g.create_dataset("ID", data=particle_ids)

    return n_particles  # type: ignore[return-value]  # used for verification


def _write_conserved_quantities(base: Path) -> None:
    """Write ConservedQuantities test files in both Format A and Format B."""
    # Format A: Roman numeral header (single file)
    format_a_dir = base / "phdf5"
    format_a_dir.mkdir(parents=True, exist_ok=True)
    format_a = [
        "I.    Cycle  II.   Ex    III.  Ey    IV.   Ez    V.    E_total",
        "VI.   Bx    VII.  By    VIII. Bz    IX.   B_total  X.    KE",
        "XI.   Total_E  XII.  E_var  XIII. Momentum",
        "--------------------------------------------------------------",
        "  Cycle     E_total    Ex         Ey         Ez         B_total    Bx         By         Bz         KE         Total_E    E_var      Momentum",
        "       0  1.000E+00  1.000E-01  2.000E-01  3.000E-01  4.000E+00  3.000E+00  1.000E+00  0.000E+00  5.000E-01  5.500E+00  0.000E+00  1.000E-03",
        "       5  1.100E+00  1.100E-01  2.100E-01  3.100E-01  3.900E+00  2.900E+00  1.000E+00  0.000E+00  6.000E-01  5.600E+00  1.818E-02  1.100E-03",
        "      10  1.200E+00  1.200E-01  2.200E-01  3.200E-01  3.800E+00  2.800E+00  1.000E+00  0.000E+00  7.000E-01  5.700E+00  3.636E-02  1.200E-03",
    ]
    (format_a_dir / "ConservedQuantities.txt").write_text("\n".join(format_a) + "\n")

    # Format B: comment header (multi-file)
    format_b_dir = base / "h5hut" / "info-conserved"
    format_b_dir.mkdir(parents=True, exist_ok=True)

    header_b = [
        "#(1-> Cycle",
        "#(2-> Total_Energy",
        "#(3-> Energy_Variation",
        "#(4-> Electric_energy",
        "#(5-> B_Local_energy",
        "#(6-> Kinetic_energy",
        "#(7-> Momentum",
        "#(8-> B_Total_energy",
        "#(9-> B_internal_energy",
        "#(10-> KE_removed",
        "#(11-> E_removed",
        "#(12-> npart_s0",
        "#(13-> charge_s0",
        "#(14-> KE_s0",
        "#(15-> npart_s1",
        "#(16-> charge_s1",
        "#(17-> KE_s1",
        "-----------------------------------------------",
    ]

    # File 0: cycles 0-10
    # Columns (0-based): cycle(0) total_E(1) E_var(2) E_elec(3) B_local(4) KE(5) mom(6) B_total(7) B_int(8) KE_rm(9) E_rm(10) npart_s0(11) charge_s0(12) KE_s0(13) npart_s1(14) charge_s1(15) KE_s1(16)
    data_b0 = [
        "0   5.5   0.0   1.0   3.5   0.5   0.001   4.0   0.5   0.0   0.0   1000   -1000   0.2   1000   1000   0.3",
        "5   5.6   0.018   1.1   3.4   0.6   0.0011   3.9   0.5   0.0   0.0   1000   -1000   0.25   1000   1000   0.35",
        "10  5.7   0.036   1.2   3.3   0.7   0.0012   3.8   0.5   0.0   0.0   1000   -1000   0.3   1000   1000   0.4",
    ]
    text0 = "\n".join(header_b + data_b0) + "\n"
    (format_b_dir / "ConservedQuantities0.txt").write_text(text0)

    # File 1: cycles 10-20 (cycle 10 duplicated, later file wins)
    data_b1 = [
        "10  5.71  0.038   1.21   3.31   0.71   0.0012   3.81   0.5   0.0   0.0   1000   -1000   0.31   1000   1000   0.41",
        "15  5.8   0.055   1.3   3.2   0.8   0.0013   3.7   0.5   0.0   0.0   1000   -1000   0.35   1000   1000   0.45",
        "20  5.9   0.073   1.4   3.1   0.9   0.0014   3.6   0.5   0.0   0.0   1000   -1000   0.4   1000   1000   0.5",
    ]
    text1 = "\n".join(header_b + data_b1) + "\n"
    (format_b_dir / "ConservedQuantities1.txt").write_text(text1)


def _write_species_quantities(base: Path) -> None:
    """Write a synthetic SpeciesQuantities.txt file."""
    sq_dir = base / "phdf5"
    sq_dir.mkdir(parents=True, exist_ok=True)
    # Format: cycle species momentum total_ke bulk_ke thermal_ke
    lines = [
        "  0  0  1.00e-03  0.200  0.050  0.150",
        "  0  1  2.00e-03  0.300  0.080  0.220",
        "  5  0  1.10e-03  0.210  0.052  0.158",
        "  5  1  2.10e-03  0.310  0.082  0.228",
        " 10  0  1.20e-03  0.220  0.054  0.166",
        " 10  1  2.20e-03  0.320  0.084  0.236",
    ]
    (sq_dir / "SpeciesQuantities.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Generate all synthetic fixtures."""
    print("Generating iPIC3D synthetic fixtures...")

    fields = _analytical_fields()
    stored = _to_gaussian_storage(fields)

    # phdf5
    phdf5_dir = OUTPUT_DIR / "phdf5"
    phdf5_dir.mkdir(parents=True, exist_ok=True)
    _write_inp(phdf5_dir / "synthetic.inp")
    _write_phdf5(phdf5_dir, stored)
    print(f"  phdf5: {phdf5_dir}")

    # shdf5
    shdf5_dir = OUTPUT_DIR / "shdf5"
    shdf5_dir.mkdir(parents=True, exist_ok=True)
    _write_shdf5(shdf5_dir, stored)
    print(f"  shdf5: {shdf5_dir}")

    # H5hut
    h5hut_dir = OUTPUT_DIR / "h5hut"
    h5hut_dir.mkdir(parents=True, exist_ok=True)
    _write_h5hut(h5hut_dir, stored)
    print(f"  h5hut: {h5hut_dir}")

    # Particles
    _write_phdf5_particles(OUTPUT_DIR)
    print(f"  particles: {OUTPUT_DIR / 'phdf5' / 'Particles_00000'}")

    # ConservedQuantities
    _write_conserved_quantities(OUTPUT_DIR)
    print("  ConservedQuantities: Format A + B")

    # SpeciesQuantities
    _write_species_quantities(OUTPUT_DIR)
    print("  SpeciesQuantities: phdf5")

    # Round-trip verification
    print("\nVerifying round-trip...")
    from pypic.readers.ipic3d import (
        IPic3DH5hutReader,
        IPic3DParallelReader,
        IPic3DSerialReader,
        load_conserved_quantities,
        parse_inp,
        parse_settings_hdf,
    )

    # phdf5
    cfg_p = parse_inp(phdf5_dir / "synthetic.inp")
    reader_p = IPic3DParallelReader(cfg_p)
    ds_p = reader_p.read_timestep(phdf5_dir, 0)
    np.testing.assert_allclose(ds_p["B1"], fields["B1"], atol=1e-14)
    np.testing.assert_allclose(ds_p["rho_c_s0"], fields["rho_c_s0"], atol=1e-12)
    np.testing.assert_allclose(ds_p["rho_c_s1"], fields["rho_c_s1"], atol=1e-12)
    np.testing.assert_allclose(ds_p["P11_s0"], fields["P11_s0"], atol=1e-12)
    np.testing.assert_allclose(ds_p["P11_s1"], fields["P11_s1"], atol=1e-12)
    print("  phdf5: OK")

    # shdf5
    cfg_s = parse_inp(shdf5_dir / "synthetic_serial.inp")
    reader_s = IPic3DSerialReader(cfg_s)
    ds_s = reader_s.read_timestep(shdf5_dir, 0)
    np.testing.assert_allclose(ds_s["B1"], fields["B1"], atol=1e-14)
    np.testing.assert_allclose(ds_s["rho_c_s0"], fields["rho_c_s0"], atol=1e-12)
    np.testing.assert_allclose(ds_s["P11_s0"], fields["P11_s0"], atol=1e-12)
    np.testing.assert_allclose(ds_s["P11_s1"], fields["P11_s1"], atol=1e-12)
    print("  shdf5: OK")

    # settings.hdf
    cfg_h = parse_settings_hdf(shdf5_dir / "settings.hdf")
    assert cfg_h.nxc == NXC
    assert cfg_h.qom == QOM
    print("  settings.hdf: OK")

    # H5hut (float32 → float64 introduces rounding)
    cfg_hut = parse_inp(h5hut_dir / "SyntheticFixture.inp")
    reader_hut = IPic3DH5hutReader(cfg_hut)
    ds_hut = reader_hut.read_timestep(h5hut_dir, 0)
    np.testing.assert_allclose(ds_hut["B1"], fields["B1"], atol=1e-6)
    np.testing.assert_allclose(ds_hut["rho_c_s0"], fields["rho_c_s0"], atol=1e-5)
    np.testing.assert_allclose(ds_hut["P11_s0"], fields["P11_s0"], atol=1e-5)
    assert ds_hut["B1"].dtype == np.float64, "H5hut should promote to float64"
    print("  h5hut: OK")

    # Particles (phdf5)
    from pypic.readers.ipic3d._particles import (
        detect_particle_steps,
        read_phdf5_particles,
    )

    p_steps = detect_particle_steps(phdf5_dir)
    assert p_steps == [0], f"Expected [0], got {p_steps}"
    pcl = read_phdf5_particles(phdf5_dir, 0, 0, cfg_p)
    assert pcl.n_particles == 18
    assert pcl.position is not None
    assert pcl.velocity is not None
    assert pcl.weight is not None
    assert pcl.weight.dtype == np.float64
    assert pcl.species_charge == -1.0  # electrons
    pcl_i = read_phdf5_particles(phdf5_dir, 0, 1, cfg_p)
    assert pcl_i.species_charge == 1.0  # ions
    assert pcl.id is not None, "Particle IDs should be loaded"
    assert pcl.id.dtype == np.int64
    np.testing.assert_array_equal(pcl.id, np.arange(18, dtype=np.int64))
    print("  particles: OK")

    # ConservedQuantities Format A
    cq_a = load_conserved_quantities(OUTPUT_DIR / "phdf5" / "ConservedQuantities.txt")
    assert len(cq_a.cycle) == 3
    assert cq_a.cycle[0] == 0
    print("  CQ Format A: OK")

    # ConservedQuantities Format B
    cq_b = load_conserved_quantities(OUTPUT_DIR / "h5hut" / "info-conserved")
    assert len(cq_b.cycle) == 5  # 0, 5, 10(deduped), 15, 20
    assert len(cq_b.species_npart) == 2
    print("  CQ Format B: OK")

    # Energy flux (phdf5)
    np.testing.assert_allclose(ds_p["EF1_s0"], fields["EF1_s0"], atol=1e-12)
    print("  energy flux (phdf5): OK")

    # FieldOutputTag
    assert cfg_p.field_output_tag != ""
    assert "pressure" in cfg_p.field_output_tag
    assert cfg_p.particles_output_cycle == 10
    print("  FieldOutputTag: OK")

    # SpeciesQuantities
    from pypic.readers.ipic3d._conserved import load_species_quantities

    sq = load_species_quantities(OUTPUT_DIR / "phdf5" / "SpeciesQuantities.txt")
    assert len(sq) == 3  # 3 cycles
    assert "momentum_s0" in sq
    assert "thermal_ke_s1" in sq
    print("  SpeciesQuantities: OK")

    # Total size
    total = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*") if f.is_file())
    print(f"\nTotal fixture size: {total:,} bytes ({total / 1024:.1f} KB)")
    print("Done!")


if __name__ == "__main__":
    main()
