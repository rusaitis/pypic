"""Pickle support for the read-only mappings pypic's frozen containers hold.

`MappingProxyType` has no reduction of its own, so every frozen dataclass that
stores one would fail `pickle`, `copy.deepcopy` and process pools. One reducer
for the type covers every holder, present and future. The registration is
process-wide: once pypic is imported, any mappingproxy pickles as a snapshot
of its mapping instead of raising.
"""

import copyreg
from collections.abc import Callable
from types import MappingProxyType


def _rebuild_mappingproxy[K, V](mapping: dict[K, V]) -> MappingProxyType[K, V]:
    return MappingProxyType(mapping)


def _reduce_mappingproxy[K, V](
    proxy: MappingProxyType[K, V],
) -> tuple[Callable[[dict[K, V]], MappingProxyType[K, V]], tuple[dict[K, V]]]:
    # Pickle stores the constructor by qualified name, and the type itself is
    # not importable as builtins.mappingproxy.
    return _rebuild_mappingproxy, (dict(proxy),)


copyreg.pickle(MappingProxyType, _reduce_mappingproxy)
