"""Typed schema models rebuilt from the §4.1 HDF5 metadata groups.

pypic writes no HDF5; the layout in `docs/schema.md` §4.1 is a contract
for the codes that *emit* it. This module reads the parts of it that map
onto typed schema objects, so a file that carries its own provenance does
not need a `simulation.toml` beside it to be attributable.

Only the canonical-layout reader (`readers/_simple.py`) calls this.
Native-format readers — iPIC3D, BATSRUS, OpenGGCM — describe file
layouts that have no `/run/` group, so probing them would buy a
guaranteed miss on every read.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import ValidationError

from pypic.schema import Run

if TYPE_CHECKING:
    import h5py

__all__ = ["read_run_group"]

_log = logging.getLogger(__name__)


def _scalar(value: Any) -> Any:  # noqa: ANN401
    """Coerce one HDF5 attribute to something pydantic accepts."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [_scalar(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _group_to_payload(group: h5py.Group) -> dict[str, Any] | list[Any]:
    """Walk an HDF5 group into plain JSON-ish Python.

    A group whose children are all digit-named (``0``, ``1``, ...) decodes
    to a list in numeric order — that is how §4.1 spells TOML arrays of
    tables (`[[run.references]]`, `[run.authors]`), by analogy with its
    existing ``/species/{s0,s1,...}/``.
    """
    payload: dict[str, Any] = {k: _scalar(v) for k, v in group.attrs.items()}
    children = list(group)
    if children and all(name.isdigit() for name in children):
        return [_group_to_payload(group[name]) for name in sorted(children, key=int)]
    for name in children:
        child = group[name]
        if hasattr(child, "attrs") and not hasattr(child, "shape"):
            payload[name] = _group_to_payload(child)
    return payload


def read_run_group(f: h5py.File | h5py.Group) -> Run | None:
    """Rebuild `pypic.schema.Run` from a §4.1 ``/run/`` group.

    Returns ``None`` when the group is absent, or when it is present but
    does not validate. A malformed header must not cost a reader its
    field data — the arrays are still good, and a `simulation.toml`
    beside the file may still carry a usable record — so this logs and
    declines rather than raising.

    Parameters
    ----------
    f : h5py.File or h5py.Group
        An open handle whose root may contain a ``run`` group.

    Returns
    -------
    Run or None
        The validated provenance record, or None.
    """
    if "run" not in f:
        return None
    payload = _group_to_payload(f["run"])
    if not isinstance(payload, dict):
        _log.warning("HDF5 /run/ decoded to a list, not a record; ignoring it")
        return None
    try:
        return Run.model_validate(payload)
    except ValidationError as exc:
        _log.warning("HDF5 /run/ group failed schema validation, ignoring it: %s", exc)
        return None
