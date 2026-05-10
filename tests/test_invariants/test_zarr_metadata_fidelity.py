# Source: TASKS.md Step 24 ("pypic metadata (grid, normalization, species,
#         physics, frame) serialized to xr.Dataset.attrs as JSON-compatible
#         dicts") + TASKS.md Step 25 precision note ("float64→float32 loses
#         ~7 decimal digits, well below PIC numerical accuracy. Default:
#         preserve source dtype") + docs/schema.md § 4 ("Every file contains
#         enough metadata to convert back to SI without the original
#         simulation.toml").
# Claim: Zarr round-trip with dtype="float32" downcasts *field arrays*
#        (expected, lossy by ~7 decimal digits) but leaves *metadata*
#        bit-exact: Normalization refs, StaggerInfo, species scalars,
#        PhysicsParams, GridInfo dt/spacing/origin, frame, transforms.
# Backlog #6 in autoresearcher-pypic.md.
"""Zarr float32 downcast preserves metadata bit-exact, arrays within 1e-6."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pypic.containers import StaggerInfo
from pypic.dataset import FieldDataset
from pypic.grid import GridInfo
from pypic.io.zarr import from_zarr, to_zarr
from pypic.units import Normalization, PhysicsParams, SpeciesInfo
from tests.strategies import normalizations


def _species_scalar() -> st.SearchStrategy[float]:
    """Finite nonzero float with full float64 mantissa exercised."""
    return st.floats(
        min_value=1e-10,
        max_value=1e10,
        allow_nan=False,
        allow_infinity=False,
        exclude_min=True,
    ).flatmap(lambda x: st.sampled_from([x, -x]))


def _species_info() -> st.SearchStrategy[SpeciesInfo]:
    return st.builds(
        SpeciesInfo,
        name=st.sampled_from(["electrons", "ions", "alpha"]),
        charge=_species_scalar(),
        mass=st.floats(
            1e-10, 1e10, allow_nan=False, allow_infinity=False, exclude_min=True
        ),
    )


def _stagger_info() -> st.SearchStrategy[StaggerInfo]:
    return st.builds(
        StaggerInfo,
        convention=st.sampled_from(["cell", "node", "staggered"]),
        interpolation_order=st.one_of(st.none(), st.integers(1, 4)),
        notes=st.one_of(st.none(), st.text(min_size=0, max_size=30)),
    )


def _physics_params() -> st.SearchStrategy[PhysicsParams]:
    return st.builds(
        PhysicsParams,
        gamma=st.floats(1.01, 3.0, allow_nan=False, allow_infinity=False),
        c=st.floats(1.0, 1e10, allow_nan=False, allow_infinity=False),
        relativistic=st.booleans(),
    )


@given(
    norm=normalizations(),
    species=st.lists(_species_info(), min_size=0, max_size=2),
    stagger=_stagger_info(),
    physics=_physics_params(),
    dt=st.floats(1e-6, 1.0, allow_nan=False, allow_infinity=False),
)
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_float32_preserves_metadata_bit_exact(
    norm: Normalization,
    species: list[SpeciesInfo],
    stagger: StaggerInfo,
    physics: PhysicsParams,
    dt: float,
) -> None:
    """Metadata survives a float32 field downcast with bit-exact fidelity.

    Field arrays lose ~7 decimal digits (float64 → float32 is lossy by
    design — the whole point of the downcast is to halve storage).
    Every other piece of ``FieldDataset`` state is scalar or structural,
    so the round-trip must be exact.
    """
    grid = GridInfo(
        dimensions=(4, 3, 2),
        spacing=(1.0, 0.5, 2.0),
        origin=(0.0, 0.0, 0.0),
        dt=dt,
    )
    b1 = np.arange(24, dtype=np.float64).reshape(4, 3, 2) * 1.5
    fds = FieldDataset.from_arrays(
        {"B_1": b1},
        grid,
        norm,
        species=species,
        physics=physics,
        metadata={"stagger": stagger, "run_id": "test"},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        store = Path(tmpdir) / "f32.zarr"
        to_zarr(fds, store, dtype="float32")
        loaded = from_zarr(store)

        # Field array: precision dropped to float32. Expected-lossy check —
        # tolerates the ~6 decimal digits float32 keeps.
        assert loaded["B_1"].dtype == np.float32
        np.testing.assert_allclose(loaded["B_1"], b1, rtol=1e-6)

        # Normalization: all 8 reference values must survive bit-exact.
        assert loaded.normalization == norm

        # StaggerInfo round-trip via the tagged __pypic_class__ path.
        loaded_stagger = loaded.metadata["stagger"]
        assert isinstance(loaded_stagger, StaggerInfo)
        assert loaded_stagger.convention == stagger.convention
        assert loaded_stagger.interpolation_order == stagger.interpolation_order
        assert loaded_stagger.notes == stagger.notes

        # Species scalars — charge/mass are the ones that travel into
        # macro_charge / macro_mass (Step 25b), so any lossy cast here
        # would corrupt every per-particle moment downstream.
        assert len(loaded.species) == len(species)
        for sp_loaded, sp_original in zip(loaded.species, species, strict=True):
            assert sp_loaded.name == sp_original.name
            assert sp_loaded.charge == sp_original.charge
            assert sp_loaded.mass == sp_original.mass

        # PhysicsParams
        assert loaded.physics.gamma == physics.gamma
        assert loaded.physics.c == physics.c
        assert loaded.physics.relativistic == physics.relativistic

        # GridInfo structure
        assert loaded.grid.dimensions == grid.dimensions
        assert loaded.grid.spacing == grid.spacing
        assert loaded.grid.origin == grid.origin
        assert loaded.grid.dt == dt
