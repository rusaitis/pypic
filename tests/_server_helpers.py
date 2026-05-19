"""Shared fixture builders for ``pypic.server`` tests.

Both the HTTP-route and WebSocket-stream test modules need to spin up a
minimal on-disk simulation tree (one ``simulation.toml`` plus a few
HDF5 timesteps) to drive ``create_app`` against. Centralizing the
helper keeps the two test files from drifting on schema or shape
conventions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


SIM_TOML = """\
[schema]
version = "1.0"

[model]
name = "test_sim"
type = "MHD"

[run]
name = "server_test_run"

[time]
scheme = "fixed"
dt = 0.1
t_start = 0.0
t_end = 1.0
n_steps = 3

[grid]
dimensions = [4, 4, 4]
spacing = [1.0, 1.0, 1.0]
lower = [0.0, 0.0, 0.0]
upper = [4.0, 4.0, 4.0]

[units]
system = "SI"

[coordinates]
geometry = "cartesian"
frame = "simulation"

[physics.mhd]
gamma = 1.6667

[[species]]
name = "p"
charge = 1.0
mass = 1.0
"""


def make_sim_dir(parent: Path, name: str = "run0", *, n_steps: int = 3) -> Path:
    """Create a synthetic single-sim tree under *parent* with *n_steps* HDF5 outputs."""
    d = parent / name
    d.mkdir()
    (d / "simulation.toml").write_text(SIM_TOML, encoding="utf-8")
    rng = np.random.default_rng(42)
    shape = (4, 4, 4)
    for i in range(n_steps):
        with h5py.File(d / f"output_{i:06d}.h5", "w") as f:
            grp = f.create_group("fields")
            grp.create_dataset("B_1", data=rng.standard_normal(shape))
            grp.create_dataset("B_2", data=rng.standard_normal(shape))
            grp.create_dataset("B_3", data=rng.standard_normal(shape))
            f.attrs["model"] = "test_sim"
            f.attrs["step"] = i
    return d
