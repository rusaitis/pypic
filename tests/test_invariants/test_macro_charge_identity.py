# Source: docs/schema.md § "Computing per-particle quantities"
#         + src/pypic/containers.py:382 (macro_charge) / :389 (macro_mass)
#         + TASKS.md Step 25b ("macro_charge collapses to a one-liner:
#           species_charge × weight. No two-branch fallback.").
# Claim: ParticleData.macro_charge == species_charge * weight bit-exact;
#        macro_mass == species_mass * weight bit-exact. Also guards the
#        fail-loud contract: either scalar missing → ValueError.
# Backlog #5 in autoresearcher-pypic.md.
"""Algebraic identity for ``ParticleData.macro_charge`` / ``macro_mass``."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pypic.containers import ParticleData


def _weight_array(n: int) -> st.SearchStrategy[np.ndarray]:
    """Positive-finite float64 weights. PIC weights are strictly positive
    (number of physical particles per macroparticle); zero is excluded to
    match the schema semantics, not to avoid floating-point hazards.
    """
    return arrays(
        dtype=np.float64,
        shape=(n,),
        elements=st.floats(
            min_value=1e-30,
            max_value=1e30,
            allow_nan=False,
            allow_infinity=False,
            exclude_min=True,
        ),
    )


def _species_scalar() -> st.SearchStrategy[float]:
    """Nonzero finite scalar (charge or mass). Zero is excluded because a
    zero species_charge would collapse macro_charge to the zero vector and
    hide sign-convention bugs in the ``species_charge * weight`` product.
    """
    return st.floats(
        min_value=1e-20,
        max_value=1e20,
        allow_nan=False,
        allow_infinity=False,
        exclude_min=True,
    ).flatmap(lambda x: st.sampled_from([x, -x]))


def _make_particle_data(
    weight: np.ndarray | None,
    species_charge: float | None,
    species_mass: float | None,
) -> ParticleData:
    """Minimal ParticleData with a placeholder velocity — shape matches weight."""
    n = 0 if weight is None else weight.shape[0]
    # ParticleData requires at least one of position/velocity. A zero-filled
    # velocity array keeps the test focused on the scalar×array identity.
    velocity = np.zeros((n, 3), dtype=np.float64)
    return ParticleData(
        species_index=0,
        species_name="test",
        position=None,
        velocity=velocity,
        n_particles=n,
        metadata={},
        weight=weight,
        species_charge=species_charge,
        species_mass=species_mass,
    )


@given(
    n=st.integers(min_value=1, max_value=32),
    species_charge=_species_scalar(),
    species_mass=_species_scalar(),
    data=st.data(),
)
@settings(max_examples=60, deadline=None)
def test_macro_charge_and_mass_bit_exact(
    n: int,
    species_charge: float,
    species_mass: float,
    data: st.DataObject,
) -> None:
    """``macro_charge == species_charge * weight`` element-wise bit-exact;
    same for ``macro_mass``. Scalar × array multiplication is a single
    float64 multiply per element, so equality is exact — the test guards
    against any future refactor adding a lossy conversion (e.g. casting
    through float32, or reading a dropped per-particle charge column
    instead of recomputing from the scalar).
    """
    weight = data.draw(_weight_array(n))
    pcl = _make_particle_data(weight, species_charge, species_mass)

    np.testing.assert_array_equal(pcl.macro_charge, species_charge * weight)
    np.testing.assert_array_equal(pcl.macro_mass, species_mass * weight)
    # Shape + dtype preservation — macro_* must not silently reshape.
    assert pcl.macro_charge.shape == weight.shape
    assert pcl.macro_charge.dtype == np.float64


@given(weight=_weight_array(4))
@settings(max_examples=20, deadline=None)
def test_macro_raises_when_species_scalar_missing(weight: np.ndarray) -> None:
    """Missing ``species_charge`` or ``species_mass`` must raise
    ``ValueError`` with a message naming the missing fields — not
    silently return zero or degrade to a per-particle fallback. This
    is the Step 25b contract ("no two-branch fallback").
    """
    pcl_no_charge = _make_particle_data(weight, None, 1.0)
    with pytest.raises(ValueError, match="species_charge"):
        _ = pcl_no_charge.macro_charge

    pcl_no_mass = _make_particle_data(weight, 1.0, None)
    with pytest.raises(ValueError, match="species_mass"):
        _ = pcl_no_mass.macro_mass


@given(
    n=st.integers(min_value=1, max_value=16),
    species_charge=_species_scalar(),
    data=st.data(),
)
@settings(max_examples=40, deadline=None)
def test_macro_raises_when_weight_missing(
    n: int, species_charge: float, data: st.DataObject
) -> None:
    """Missing ``weight`` must raise ``ValueError`` naming ``weight``.

    The placeholder velocity ensures ``ParticleData.__post_init__`` accepts
    the instance — the failure must come from ``macro_charge``, not
    construction.
    """
    del data  # unused — kept to keep the signature stable across strategies
    pcl = ParticleData(
        species_index=0,
        species_name="test",
        position=None,
        velocity=np.zeros((n, 3), dtype=np.float64),
        n_particles=n,
        metadata={},
        weight=None,
        species_charge=species_charge,
        species_mass=1.0,
    )
    with pytest.raises(ValueError, match="weight"):
        _ = pcl.macro_charge
