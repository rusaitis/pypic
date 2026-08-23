"""Shared optional-dependency guard for the extras-gated subpackages.

``pypic.io``, ``pypic.plotting`` and ``pypic.server`` each sit behind an
extra.  They all need the same thing: check a handful of imports once, and
fail with an install hint instead of a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

from importlib import import_module

# Import name -> None when satisfied, else the message to re-raise.  Keeps
# the wording identical on the first failure and every call after it.
_CHECKED: dict[str, str | None] = {}


def require(*modules: str, extra: str, feature: str) -> None:
    """Raise ``ImportError`` if any of *modules* is missing.

    Parameters
    ----------
    *modules : str
        Import names *feature* needs, reported in the order given.
    extra : str
        Name of the pyproject extra that ships them.
    feature : str
        What needs them, used as the subject of the message.

    Raises
    ------
    ImportError
        If any module is missing. The message names the missing modules
        and the install command, with the extra quoted for shells that
        glob brackets.

    Examples
    --------
    >>> require("math", extra="core", feature="arithmetic")
    >>> require("nope", extra="zarr", feature="Zarr I/O")
    Traceback (most recent call last):
        ...
    ImportError: Zarr I/O requires nope. Install with: pip install "pypic-plasma[zarr]"
    """
    key = f"{extra}:{','.join(modules)}:{feature}"
    if key in _CHECKED:
        cached = _CHECKED[key]
        if cached is None:
            return
        raise ImportError(cached) from None
    missing: list[str] = []
    for name in modules:
        try:
            import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        msg = (
            f"{feature} requires {', '.join(missing)}. "
            f'Install with: pip install "pypic-plasma[{extra}]"'
        )
        _CHECKED[key] = msg
        raise ImportError(msg) from None
    _CHECKED[key] = None
