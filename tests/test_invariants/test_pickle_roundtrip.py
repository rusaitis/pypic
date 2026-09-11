# Source: src/pypic/_frozen.py (the MappingProxyType reducer) +
#         docs/architecture.md § "Python conventions" (frozen dataclasses
#         expose internal dicts as MappingProxyType; concurrent.futures for
#         parallelism).
# Claim: every container that holds a read-only mapping survives pickle,
#        copy.deepcopy and the pickler process pools use, and comes back
#        equal with every mapping still read-only. A mappingproxy has no
#        reduction of its own, so without the reducer each of them raises and
#        a dataset can neither cross a ProcessPoolExecutor nor land in a
#        joblib.Memory cache.
"""Read-only containers round-trip through pickle, deepcopy and process pools."""

from __future__ import annotations

import copy
import dataclasses
import pickle
from collections.abc import Mapping
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from pypic import (
    CARTESIAN,
    FieldDataset,
    FieldLine,
    GridInfo,
    Normalization,
    ParticleData,
    ParticleTrace,
    PhysicsParams,
    PoincareSection,
    PoincareSurface,
    SimulationConfig,
    StaggerInfo,
    TabularData,
    parse_inp,
)
from pypic.readers.openggcm._grid import OpenGGCMGrid

if TYPE_CHECKING:
    from collections.abc import Callable

_IPIC3D_INP = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "ipic3d-synthetic"
    / "phdf5"
    / "synthetic.inp"
)

_CLONES: dict[str, Callable[[object], object]] = {
    "pickle": lambda obj: pickle.loads(pickle.dumps(obj)),
    "deepcopy": copy.deepcopy,
    "process-pool pickler": lambda obj: pickle.loads(ForkingPickler.dumps(obj)),
}

_DATASET_STATE = (
    "grid",
    "normalization",
    "species",
    "physics",
    "metadata",
    "frame",
    "transforms",
    "aliases",
)


def _holders() -> dict[str, object]:
    rng = np.random.default_rng(7)
    normalization = Normalization.identity()
    physics = PhysicsParams(extra={"eta": 0.01})
    grid = GridInfo((4, 3, 2), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), CARTESIAN)
    points = rng.standard_normal((5, 3))
    field_line = FieldLine(
        points=points,
        field_name="B",
        seed_point=(0.0, 0.0, 0.0),
        normalization=normalization,
        scalars={"|B|": np.ones(5)},
        metadata={"n_steps": 5},
    )
    axis = np.linspace(0.0, 1.0, 2)
    return {
        "PhysicsParams": physics,
        "StaggerInfo": StaggerInfo(
            convention="staggered",
            field_locations={"B": "face"},
            position={"B_1": (0.5, 0.0, 0.0)},
        ),
        "SimulationConfig": SimulationConfig(
            model_name="t",
            model_type="PIC",
            grid=grid,
            normalization=normalization,
            physics=physics,
            metadata={"step": 3},
        ),
        "TabularData": TabularData(
            name="probe", columns={"t": np.arange(4.0)}, metadata={"cadence": 1.0}
        ),
        "ParticleData": ParticleData(
            species_index=0,
            species_name="e",
            position=rng.standard_normal((3, 3)),
            velocity=rng.standard_normal((3, 3)),
            n_particles=3,
            metadata={"step": 0},
        ),
        "FieldLine": field_line,
        "ParticleTrace": ParticleTrace(
            points=points,
            time=np.arange(5.0),
            velocity=rng.standard_normal((5, 3)),
            species_name="e",
            normalization=normalization,
            scalars={"gamma": np.ones(5)},
            metadata={"pusher": "boris"},
        ),
        "PoincareSection": PoincareSection(
            surface=PoincareSurface.from_axis("z", 0.0),
            seeds=np.zeros((1, 3)),
            direction="forward",
            punctures_3d=(np.zeros((2, 3)),),
            punctures_2d=(np.zeros((2, 2)),),
            field_lines=(field_line,),
            metadata={"turns": 2},
        ),
        "IPic3DConfig": parse_inp(_IPIC3D_INP),
        "OpenGGCMGrid": OpenGGCMGrid(
            nx=2,
            ny=2,
            nz=2,
            x=axis,
            y=axis,
            z=axis,
            stagger=MappingProxyType({"bx": (axis, axis, axis)}),
            metadata=MappingProxyType({"source": "test"}),
        ),
        "FieldDataset": FieldDataset.from_arrays(
            {"B_1": rng.standard_normal((4, 3, 2))},
            grid,
            normalization,
            physics=physics,
        ),
    }


def _same(a: object, b: object) -> bool:
    # Types must match all the way down, so a read-only mapping that comes
    # back as a plain dict counts as a difference.
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, tuple | list) and isinstance(b, tuple | list):
        return len(a) == len(b) and all(map(_same, a, b))
    if isinstance(a, FieldDataset) and isinstance(b, FieldDataset):
        return a.xr.identical(b.xr) and all(
            _same(getattr(a, name), getattr(b, name)) for name in _DATASET_STATE
        )
    if dataclasses.is_dataclass(a):
        return all(
            _same(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a)
        )
    return bool(a == b)


def test_read_only_containers_survive_pickle_deepcopy_and_process_pools() -> None:
    failures = []
    for name, holder in _holders().items():
        for how, clone in _CLONES.items():
            try:
                again = clone(holder)
            except (TypeError, pickle.PicklingError) as exc:
                failures.append(f"{name} via {how}: {exc}")
                continue
            if not _same(holder, again):
                failures.append(f"{name} via {how}: differs or lost a read-only map")
    assert not failures, "\n".join(failures)
