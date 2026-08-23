"""Visual Poincaré section plots for manual inspection.

Run with::

    uv run python scripts/visual/visual_poincare.py

Output goes to ``tests/output/poincare_*.png``. Generates four scenes:

1. Nested magnetic islands (closed circles, axis-aligned cut)
2. Helical drift (vortex + axial flow, axis-aligned cut)
3. Magnetotail-style X-line with islands (sheared field + perturbation)
4. The same X-line field cut by a *tilted* plane (arbitrary normal)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt

from pypic import PoincareSurface, poincare_section
from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.plotting import plot_poincare_section, use_theme
from pypic.units import Normalization

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = _REPO_ROOT / "tests" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def _build_dataset(
    fields: dict[str, np.ndarray], extent: tuple[float, float, float], n: int
) -> FieldDataset:
    lx, ly, lz = extent
    return FieldDataset.from_arrays(
        fields,
        GridInfo(
            dimensions=(n, n, n),
            spacing=(2 * lx / (n - 1), 2 * ly / (n - 1), 2 * lz / (n - 1)),
            origin=(-lx, -ly, -lz),
        ),
        Normalization.identity(),
    )


def scene_nested_islands(theme: str = "dark") -> Path:
    """Pure closed circles — nested rings, one per seed radius."""
    n = 128
    extent = 1.5
    coords = np.linspace(-extent, extent, n)
    X, Y, _ = np.meshgrid(coords, coords, coords, indexing="ij")
    data = _build_dataset(
        {"B_1": -Y, "B_2": X, "B_3": np.zeros_like(X)},
        (extent, extent, extent),
        n,
    )

    surf = PoincareSurface.from_axis("y", 0.0, name="y = 0 (closed orbits)")
    radii = np.linspace(0.2, 1.2, 8)
    seeds = np.column_stack([radii, np.zeros_like(radii), np.zeros_like(radii)])

    section = poincare_section(
        data,
        seeds,
        surf,
        max_steps=4000,
        direction="forward",
        atol=1e-10,
        rtol=1e-10,
        step_size_init=0.005,
        max_step=0.02,
    )

    with use_theme(theme):
        fig, ax = plot_poincare_section(section, marker_size=8.0, alpha=0.9)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(
            "Nested magnetic islands\n"
            r"$\mathbf{B} = (-y, x, 0)$, fan of seeds, $y = 0$ section"
        )
    out = OUTPUT_DIR / f"poincare_islands_{theme}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def scene_helical_drift(theme: str = "dark") -> Path:
    """Helix with axial drift — puncture cloud drifts monotonically in u (= -z)."""
    n = 96
    extent = 2.5
    coords = np.linspace(-extent, extent, n)
    X, Y, _ = np.meshgrid(coords, coords, coords, indexing="ij")
    data = _build_dataset(
        {"B_1": -Y, "B_2": X, "B_3": 0.1 * np.ones_like(X)},
        (extent, extent, extent),
        n,
    )

    surf = PoincareSurface.from_axis("y", 0.0, name="y = 0 (helical drift)")
    seeds = np.array([[r, 0.0, -2.0] for r in [0.4, 0.7, 1.0, 1.3]])

    section = poincare_section(
        data,
        seeds,
        surf,
        max_steps=10_000,
        direction="forward",
        atol=1e-10,
        rtol=1e-10,
        step_size_init=0.01,
        max_step=0.05,
    )

    with use_theme(theme):
        fig, ax = plot_poincare_section(section, marker_size=10.0, alpha=0.85)
        ax.set_title(
            "Helical drift\n"
            r"$\mathbf{B} = (-y, x, 0.1)$, seeds at $z = -2$, $y = 0$ section"
        )
        ax.set_xlabel(r"$u = -z$")
        ax.set_ylabel(r"$v = -x$")
    out = OUTPUT_DIR / f"poincare_helix_{theme}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def _abc_field(
    n: int = 192,
    periods: int = 4,
    a: float = 1.0,
    b: float = 1.0 / np.sqrt(2),
    c: float = 1.0 / np.sqrt(3),
) -> FieldDataset:
    r"""Arnold–Beltrami–Childress (ABC) flow, sampled on multiple periods.

    $\mathbf{B} = (A \sin z + C \cos y,\;
                   B \sin x + A \cos z,\;
                   C \sin y + B \cos x)$

    The canonical 3D chaotic-flow test field. Default parameters
    $(A, B, C) = (1, 1/\sqrt{2}, 1/\sqrt{3})$ follow Dombre et al.
    (1986) — breaking the threefold symmetry gives cleaner KAM
    tori interleaved with chaotic regions. Sampled on
    $[-2\pi, 2\pi]^3$ (``periods=4``) so traces have room to wander
    many revolutions before reaching the box boundary.
    """
    extent = periods * np.pi
    coords = np.linspace(-extent, extent, n)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    Bx = a * np.sin(Z) + c * np.cos(Y)
    By = b * np.sin(X) + a * np.cos(Z)
    Bz = c * np.sin(Y) + b * np.cos(X)
    return FieldDataset.from_arrays(
        {"B_1": Bx, "B_2": By, "B_3": Bz},
        GridInfo(
            dimensions=(n, n, n),
            spacing=(2 * extent / (n - 1),) * 3,
            origin=(-extent, -extent, -extent),
        ),
        Normalization.identity(),
    )


def scene_abc_flow(theme: str = "dark") -> Path:
    r"""ABC flow Poincaré section: classic KAM tori + chaotic sea.

    Seeds along the $y = \\pi/2$ midline at $z = 0$; punctures gathered
    on the $z = 0$ plane (a period midplane). Different starting $x$
    land on different invariant tori or in the chaotic region.
    """
    data = _abc_field(n=192, periods=4)

    surf = PoincareSurface.from_axis("z", 0.0, name="z = 0 (ABC midplane)")

    n_seeds = 36
    seed_x = np.linspace(0.1, 2 * np.pi - 0.1, n_seeds)
    seeds = np.column_stack(
        [
            seed_x,
            np.full(n_seeds, np.pi),
            np.zeros(n_seeds),
        ]
    )

    section = poincare_section(
        data,
        seeds,
        surf,
        max_steps=50_000,
        direction="both",
        atol=1e-9,
        rtol=1e-9,
        step_size_init=0.02,
        max_step=0.1,
    )

    with use_theme(theme):
        fig, ax = plot_poincare_section(
            section,
            marker_size=1.2,
            alpha=0.55,
            color_by_seed=True,
        )
        ax.set_title(
            "Arnold–Beltrami–Childress (ABC) flow\n"
            r"$(A, B, C) = (1, 1/\sqrt{2}, 1/\sqrt{3})$, "
            r"$z = 0$ Poincaré section"
        )
    out = OUTPUT_DIR / f"poincare_abc_{theme}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def scene_tilted_plane(theme: str = "dark") -> Path:
    """ABC flow with a *tilted* plane — showcase arbitrary normal."""
    data = _abc_field(n=192, periods=4)

    surf = PoincareSurface(
        normal=(0.3, 0.0, 1.0),
        point=(0.0, 0.0, 0.0),
        name="tilted plane (17° from xy)",
    )

    n_seeds = 36
    seed_x = np.linspace(0.1, 2 * np.pi - 0.1, n_seeds)
    seeds = np.column_stack(
        [
            seed_x,
            np.full(n_seeds, np.pi),
            np.zeros(n_seeds),
        ]
    )

    section = poincare_section(
        data,
        seeds,
        surf,
        max_steps=50_000,
        direction="both",
        atol=1e-9,
        rtol=1e-9,
        step_size_init=0.02,
        max_step=0.1,
    )

    with use_theme(theme):
        fig, ax = plot_poincare_section(
            section,
            marker_size=1.2,
            alpha=0.55,
            color_by_seed=True,
        )
        ax.set_title(
            "ABC flow, tilted Poincaré plane\n"
            r"normal $\propto (0.3, 0, 1)$, Gram–Schmidt $(u, v)$ basis"
        )
    out = OUTPUT_DIR / f"poincare_tilted_{theme}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    """Render every Poincaré scene into OUTPUT_DIR."""
    theme = "dark"
    print("Generating Poincaré section visuals...")
    for scene_fn in (
        scene_nested_islands,
        scene_helical_drift,
        scene_abc_flow,
        scene_tilted_plane,
    ):
        try:
            out = scene_fn(theme=theme)
            print(f"  {out.name}")
        except Exception as err:
            print(f"  FAILED {scene_fn.__name__}: {err!r}")
            raise


if __name__ == "__main__":
    main()
