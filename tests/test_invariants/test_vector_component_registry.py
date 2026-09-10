# Source: docs/schema.md § 3 "Dual naming convention" (vector components
#         are ``<field>[_s<N>]_<i>`` with i in 1..3) + src/pypic/fields.py
#         (vector_component) + src/pypic/coordinates/transforms.py
#         (find_vector_triplets, which FieldDataset.transform_to uses to
#         decide which arrays are rotated as vectors).
# Claim: every registered name of the form ``<base>_<i>`` is one component
#        of a complete, registered triplet, and vector_component detects it.
#        transform_to rotates exactly the triplets this function reports, so
#        a registered vector with a missing sibling, or a component the
#        detector misses, would be reoriented in space while keeping
#        old-frame components — wrong physics with no error.
"""Every registered vector component belongs to a detectable, complete triplet."""

from __future__ import annotations

import re

from pypic._field_table import _FIELD_INFO
from pypic.coordinates.transforms import find_vector_triplets
from pypic.fields import vector_component

_COMPONENT = re.compile(r"^(?P<base>.+)_(?P<c>[123])$")


def test_registered_components_form_complete_detectable_triplets() -> None:
    bases: dict[str, set[int]] = {}
    undetected: list[str] = []
    for name in _FIELD_INFO:
        m = _COMPONENT.match(name)
        if m is None:
            continue
        bases.setdefault(m["base"], set()).add(int(m["c"]))
        if vector_component(name) != (m["base"], int(m["c"])):
            undetected.append(name)
    incomplete = sorted(b for b, cs in bases.items() if cs != {1, 2, 3})
    assert not undetected, f"components vector_component misses: {undetected}"
    assert not incomplete, f"registered vectors with missing siblings: {incomplete}"
    triplets = {t[0] for t in find_vector_triplets(_FIELD_INFO)}
    assert triplets == {f"{b}_1" for b in bases}, "find_vector_triplets disagrees"
