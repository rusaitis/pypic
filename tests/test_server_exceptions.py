"""Unit tests for the typed pypic / server exception hierarchy.

The boundary behavior (an unknown sim turns into an
``ErrorFrame(kind="unknown_sim")`` over the WebSocket, a 404 over
HTTP, etc.) lives in :mod:`tests.test_server_stream` and
:mod:`tests.test_server_routes`.  Tests here pin the *contract* the
hierarchy promises so the boundary code can rely on it: every subclass
routes to the right wire kind and HTTP status through
:func:`pypic.server.exceptions.error_routing`, and satisfies
``isinstance`` against both :class:`PypicError` and the appropriate
standard base, so existing ``except KeyError`` /
``except NotImplementedError`` callers in the library and in
third-party code keep working.
"""

from __future__ import annotations

import pydantic
import pytest

from pypic.exceptions import (
    GeometryUnsupportedError,
    PypicError,
    UndeclaredNormalizationError,
    UnknownFieldError,
    UnknownSimulationError,
    UnknownStepError,
)
from pypic.server.exceptions import ValidationFailedError, error_routing
from pypic.server.protocol import ErrorKind

# Every mapped exception type with the pair it must route to and the
# standard base a library caller would have caught instead.
_ROUTING_CONTRACT: tuple[tuple[type[PypicError], str, int, type[Exception]], ...] = (
    (UnknownSimulationError, "unknown_sim", 404, KeyError),
    (UnknownFieldError, "unknown_field", 404, KeyError),
    (UnknownStepError, "unknown_step", 404, KeyError),
    (GeometryUnsupportedError, "geometry_unsupported", 400, NotImplementedError),
    (UndeclaredNormalizationError, "undeclared_normalization", 400, ValueError),
    (ValidationFailedError, "validation", 422, ValueError),
)


def test_bare_pypic_error_routes_to_the_catch_all() -> None:
    """An unmapped error still reaches the wire as internal/500."""
    assert error_routing(PypicError("oops")) == ("internal", 500)


def test_an_unmapped_subclass_inherits_its_parents_routing() -> None:
    """The MRO walk, not an exact-type lookup, is what keeps the wire valid.

    `ErrorKind` is a closed vocabulary, so a subclass added without a
    routing entry must degrade to its nearest mapped base rather than
    put an unknown kind on the wire.
    """

    class _NarrowerFieldError(UnknownFieldError):
        pass

    assert error_routing(_NarrowerFieldError("oops")) == ("unknown_field", 404)


@pytest.mark.parametrize(
    ("exc_cls", "kind", "status", "stdlib_base"), _ROUTING_CONTRACT
)
def test_subclass_routing_contract(
    exc_cls: type[PypicError],
    kind: str,
    status: int,
    stdlib_base: type[Exception],
) -> None:
    """Each subclass routes to its wire kind and HTTP status, and stays catchable.

    The :mod:`pypic.server.app` global handler and
    :func:`pypic.server.stream._handle_one` both go through
    ``error_routing``; the multi-inheritance with ``stdlib_base`` is the
    contract that lets library callers keep their existing
    ``except KeyError`` / ``except NotImplementedError`` catches.
    """
    exc: PypicError = exc_cls("oops")
    assert error_routing(exc) == (kind, status)
    assert isinstance(exc, PypicError)
    assert isinstance(exc, stdlib_base)
    # ``exc.detail`` is what the HTTP body and the WebSocket
    # :class:`ErrorFrame.message` carry; it must round-trip the message
    # without ``KeyError``'s ``repr``-quoting.
    assert exc.detail == "oops"


def test_every_pypic_error_subclass_is_routed() -> None:
    """A new subclass cannot ship without deciding how it reaches the wire.

    Walks the hierarchy rather than trusting the parametrize list above,
    which a new subclass would otherwise silently bypass — it would
    route to ``internal`` / 500 and nobody would notice.
    """

    def descendants(cls: type[PypicError]) -> set[type[PypicError]]:
        subs = set(cls.__subclasses__())
        return subs.union(*(descendants(sub) for sub in subs)) if subs else subs

    declared = {entry[0] for entry in _ROUTING_CONTRACT}
    # Test-local subclasses from the MRO-walk test above are not
    # production types and are expected to inherit their routing.
    unrouted = {
        cls
        for cls in descendants(PypicError) - declared
        if not cls.__name__.startswith("_")
    }
    assert not unrouted, (
        f"unrouted PypicError subclasses: {sorted(c.__name__ for c in unrouted)}"
    )


def test_every_routed_kind_is_on_the_wire_vocabulary() -> None:
    """Routing cannot name a kind `ErrorFrame` would reject."""
    from typing import get_args

    valid = set(get_args(ErrorKind.__value__))
    routed = {kind for _, kind, _, _ in _ROUTING_CONTRACT} | {"internal"}
    assert routed <= valid


def test_detail_unwraps_keyerror_quotes() -> None:
    """``str()`` on KeyError-inheriting subclasses requotes; ``detail`` doesn't.

    Pins the contract the HTTP-handler / WS-handler call sites rely on
    after dropping the brittle ``str(exc).strip("'")`` workaround.
    """
    exc = UnknownFieldError("missing")
    assert str(exc) == "'missing'"  # KeyError.__str__ behavior
    assert exc.detail == "missing"


def test_detail_preserves_embedded_quotes() -> None:
    """Regression: the old ``.strip("'")`` workaround silently mangled this case.

    When the message itself contains a single quote, ``KeyError.__str__``
    switches to ``repr`` with surrounding double quotes, so a blanket
    ``strip("'")`` no longer matches — junk like
    ``"'foo' is reserved"`` (with literal ``"``) reached the wire.
    """
    exc = UnknownFieldError("'foo' is reserved")
    assert exc.detail == "'foo' is reserved"


def test_validation_failed_error_preserves_cause() -> None:
    """``ValidationFailedError`` keeps the pydantic ``ValidationError`` on __cause__."""

    class _Model(pydantic.BaseModel):
        x: int

    with pytest.raises(pydantic.ValidationError) as exc_info:
        _Model.model_validate({"x": "not_an_int"})
    original = exc_info.value

    with pytest.raises(ValidationFailedError) as wrapped_info:
        raise ValidationFailedError(str(original)) from original
    wrapped = wrapped_info.value
    assert wrapped.__cause__ is original
    assert isinstance(wrapped, PypicError)
    assert isinstance(wrapped, ValueError)
