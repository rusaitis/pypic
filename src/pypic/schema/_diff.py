"""Diff helpers for JSON Schema documents.

Pure-Python, no dependencies beyond stdlib + :mod:`pypic.schema._export`
for stable serialization. Used by ``pypic schema diff`` and importable
directly by library consumers that want to compare two checked-in
schemas (e.g., to draft migration notes when v1.1 ships).

Two output shapes:

- :func:`schema_text_diff` — unified text diff over pretty-printed JSON.
  Human-readable, fit for migration docs and PR descriptions.
- :func:`schema_structured_diff` — ``{added, removed, changed}`` with
  RFC 6901 JSON Pointer paths. Machine-readable for tooling.

Both inputs are serialized through :func:`pypic.schema._export.dump_schema`
with ``pretty=True`` so key ordering is identical on each side and
differences reflect real shape changes, not formatting noise.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

from pypic.schema._export import dump_schema

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["schema_structured_diff", "schema_text_diff"]


def schema_text_diff(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    fromfile: str = "a",
    tofile: str = "b",
    n_context: int = 3,
) -> str:
    """Return a unified text diff of two JSON Schema documents.

    Returns the empty string when the documents are byte-equivalent
    after pretty-printing with sorted keys.

    Parameters
    ----------
    a, b
        JSON Schema documents (or any JSON-serializable mappings) to
        compare. Order is significant: lines prefixed ``-`` are in *a*,
        lines prefixed ``+`` are in *b*.
    fromfile, tofile
        Labels for the unified-diff header lines.
    n_context
        Number of context lines around each hunk (``difflib`` default 3).
    """
    a_lines = dump_schema(a, pretty=True).splitlines(keepends=True)
    b_lines = dump_schema(b, pretty=True).splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            a_lines,
            b_lines,
            fromfile=fromfile,
            tofile=tofile,
            n=n_context,
        )
    )


def schema_structured_diff(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
) -> dict[str, Any]:
    r"""Return a structured ``{added, removed, changed}`` diff.

    Semantics
    ---------
    - ``added``: sorted list of paths present in *b* but not *a*.
    - ``removed``: sorted list of paths present in *a* but not *b*.
    - ``changed``: dict mapping path → ``{"before": ..., "after": ...}``
      for leaf-value mismatches or list-length differences. Lists of
      equal length recurse element-wise (each index gets its own path);
      lists of differing lengths record the entire lists at the parent
      path without per-element add/remove decomposition.

    Paths use RFC 6901 JSON Pointer syntax (``/$defs/Run``,
    ``/properties/time/properties/dt``) with ``~`` → ``~0`` and
    ``/`` → ``~1`` escapes for object keys.

    Examples
    --------
    >>> schema_structured_diff({"a": 1}, {"a": 1})
    {'added': [], 'removed': [], 'changed': {}}
    >>> schema_structured_diff({"a": 1, "b": 2}, {"a": 1, "c": 3})
    {'added': ['/c'], 'removed': ['/b'], 'changed': {}}
    """
    added: list[str] = []
    removed: list[str] = []
    changed: dict[str, dict[str, Any]] = {}
    _walk_diff(a, b, "", added, removed, changed)
    added.sort()
    removed.sort()
    return {
        "added": added,
        "removed": removed,
        "changed": dict(sorted(changed.items())),
    }


def _walk_diff(
    a: Any,  # noqa: ANN401
    b: Any,  # noqa: ANN401
    path: str,
    added: list[str],
    removed: list[str],
    changed: dict[str, dict[str, Any]],
) -> None:
    """Depth-first walk emitting differences into the accumulators."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in a.keys() | b.keys():
            child_path = f"{path}/{_escape_pointer_token(key)}"
            if key not in b:
                removed.append(child_path)
            elif key not in a:
                added.append(child_path)
            else:
                _walk_diff(a[key], b[key], child_path, added, removed, changed)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            changed[path] = {"before": a, "after": b}
            return
        for i, (av, bv) in enumerate(zip(a, b, strict=True)):
            _walk_diff(av, bv, f"{path}/{i}", added, removed, changed)
        return
    if a != b:
        changed[path] = {"before": a, "after": b}


def _escape_pointer_token(key: object) -> str:
    """Escape an object key per RFC 6901 (``~`` → ``~0``, ``/`` → ``~1``)."""
    return str(key).replace("~", "~0").replace("/", "~1")
