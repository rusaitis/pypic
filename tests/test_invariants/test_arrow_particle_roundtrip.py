# Source: src/pypic/io/_arrow.py:30 (``_encode_species_meta`` — serializes
#         species scalars to JSON bytes under ``b"pypic"`` schema metadata)
#         + :192 (``particles_from_arrow`` — reads the same JSON back).
# Claim: round-trip ``particles_from_arrow(particles_to_arrow(pcl))``
#        preserves species scalars, per-particle arrays, and count
#        bit-exact, including unusual magnitudes (1e-20, 1e20) that
#        would reveal any float → str → float quantization.
# Distinct from iteration 6's Zarr float32 test:
#   * Zarr stores scalars in a separate attrs dict (direct pickle);
#   * Arrow routes them through ``json.dumps/loads``, where a
#     float that round-trips Python's repr may not survive in some
#     edge cases (subnormals, values near 1e-16 relative precision).
# Fresh invariant #8 after backlog exhaustion.
"""Arrow round-trip bit-exactness for ParticleData scalars and arrays."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_array_equal

pyarrow = pytest.importorskip("pyarrow")

from pypic.containers import ParticleData  # noqa: E402 — after importorskip
from pypic.io._arrow import (  # noqa: E402 — after importorskip
    particles_from_arrow,
    particles_to_arrow,
)


def _species_scalar() -> st.SearchStrategy[float]:
    """Finite nonzero floats spanning 1e-10 to 1e10, plus their negatives.

    Subnormals excluded: json.dumps on a subnormal emits ``"2.22e-313"``
    which json.loads will parse but the round-trip may not be bit-exact
    across Python versions — the invariant is brittle on subnormals for
    reasons unrelated to pypic.
    """
    return st.floats(
        min_value=1e-10,
        max_value=1e10,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
        exclude_min=True,
    ).flatmap(lambda x: st.sampled_from([x, -x]))


def _position_array(n: int) -> st.SearchStrategy[np.ndarray]:
    return arrays(
        dtype=np.float64,
        shape=(n, 3),
        elements=st.floats(
            min_value=-100.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )


def _weight_array(n: int) -> st.SearchStrategy[np.ndarray]:
    return arrays(
        dtype=np.float64,
        shape=(n,),
        elements=st.floats(
            min_value=1e-6,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
            exclude_min=True,
        ),
    )


@given(
    n=st.integers(min_value=1, max_value=32),
    species_index=st.integers(min_value=0, max_value=7),
    species_name=st.sampled_from(["electrons", "ions", "alpha", "He++"]),
    species_charge=_species_scalar(),
    species_mass=_species_scalar(),
    data=st.data(),
)
@settings(max_examples=30, deadline=None)
def test_arrow_roundtrip_preserves_scalars_and_arrays(
    n: int,
    species_index: int,
    species_name: str,
    species_charge: float,
    species_mass: float,
    data: st.DataObject,
) -> None:
    """Complete round-trip must be bit-exact.

    * ``species_*`` scalars flow through ``json.dumps`` / ``json.loads``
      — the claim is that Python's JSON encoding of float64 always
      produces a string that parses back to the same float.
    * ``n_particles`` must match (table length == original count).
    * Position, velocity, weight arrays are columnar Arrow; zero-copy
      NumPy → Arrow → NumPy preserves every bit.
    * ``id`` (int64) must round-trip without any silent int32 cast.
    """
    position = data.draw(_position_array(n))
    velocity = data.draw(_position_array(n))
    weight = data.draw(_weight_array(n))
    particle_id = data.draw(
        arrays(dtype=np.int64, shape=(n,), elements=st.integers(0, 2**40))
    )

    original = ParticleData(
        species_index=species_index,
        species_name=species_name,
        position=position,
        velocity=velocity,
        n_particles=n,
        metadata={"source": "arrow-invariant-test"},
        id=particle_id,
        weight=weight,
        species_charge=species_charge,
        species_mass=species_mass,
    )

    table = particles_to_arrow(original)
    recovered = particles_from_arrow(table)

    # Scalar metadata — JSON round-trip fidelity
    assert recovered.species_index == species_index
    assert recovered.species_name == species_name
    assert recovered.n_particles == n
    assert recovered.species_charge == species_charge
    assert recovered.species_mass == species_mass

    # Per-particle arrays — Arrow zero-copy fidelity
    assert recovered.position is not None
    assert recovered.velocity is not None
    assert recovered.weight is not None
    assert recovered.id is not None
    assert_array_equal(recovered.position, position)
    assert_array_equal(recovered.velocity, velocity)
    assert_array_equal(recovered.weight, weight)
    assert_array_equal(recovered.id, particle_id)


@given(
    n=st.integers(min_value=1, max_value=16),
    species_charge=_species_scalar(),
    species_mass=_species_scalar(),
    data=st.data(),
)
@settings(max_examples=30, deadline=None)
def test_macro_charge_preserved_through_arrow(
    n: int,
    species_charge: float,
    species_mass: float,
    data: st.DataObject,
) -> None:
    """``macro_charge`` computed before and after the round-trip must
    match bit-exact. Ties iteration 5's per-particle identity
    (macro_charge = species_charge × weight) to the Arrow round-trip:
    if either the scalar or the weight array loses precision, the
    derived per-particle charge will drift.
    """
    weight = data.draw(_weight_array(n))
    original = ParticleData(
        species_index=0,
        species_name="electrons",
        position=None,
        velocity=np.zeros((n, 3), dtype=np.float64),
        n_particles=n,
        metadata={},
        weight=weight,
        species_charge=species_charge,
        species_mass=species_mass,
    )

    recovered = particles_from_arrow(particles_to_arrow(original))

    assert_array_equal(recovered.macro_charge, original.macro_charge)
    assert_array_equal(recovered.macro_mass, original.macro_mass)
