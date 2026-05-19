"""Unit tests for the typed pypic / server exception hierarchy.

The boundary behavior (an unknown sim turns into an
``ErrorFrame(kind="unknown_sim")`` over the WebSocket, a 404 over
HTTP, etc.) lives in :mod:`tests.test_server_stream` and
:mod:`tests.test_server_routes`.  Tests here pin the *contract* the
hierarchy promises so the boundary code can rely on it: every
subclass has the right ``kind`` / ``status_code`` classvars and
satisfies ``isinstance`` against both :class:`PypicError` and the
appropriate standard base, so existing ``except KeyError`` /
``except NotImplementedError`` callers in the library and in
third-party code keep working.
"""

from __future__ import annotations

import pytest

from pypic.exceptions import (
    GeometryUnsupportedError,
    PypicError,
    UnknownFieldError,
    UnknownSimulationError,
    UnknownStepError,
)
from pypic.server.exceptions import ValidationFailedError


def test_pypic_error_defaults_to_internal() -> None:
    """Bare PypicError routes to the catch-all internal/500 kind."""
    assert PypicError.kind == "internal"
    assert PypicError.status_code == 500


@pytest.mark.parametrize(
    ("exc_cls", "kind", "status", "stdlib_base"),
    [
        (UnknownSimulationError, "unknown_sim", 404, KeyError),
        (UnknownFieldError, "unknown_field", 404, KeyError),
        (UnknownStepError, "unknown_step", 404, KeyError),
        (GeometryUnsupportedError, "geometry_unsupported", 400, NotImplementedError),
        (ValidationFailedError, "validation", 422, ValueError),
    ],
)
def test_subclass_routing_contract(
    exc_cls: type[PypicError],
    kind: str,
    status: int,
    stdlib_base: type[Exception],
) -> None:
    """Each subclass advertises the wire kind, HTTP status, and inheritance.

    The :mod:`pypic.server.app` global handler and
    :func:`pypic.server.stream._handle_one` both read ``kind`` and
    ``status_code`` directly; the multi-inheritance with ``stdlib_base``
    is the contract that lets library callers keep their existing
    ``except KeyError`` / ``except NotImplementedError`` catches.
    """
    assert exc_cls.kind == kind
    assert exc_cls.status_code == status
    exc: PypicError = exc_cls("oops")
    assert isinstance(exc, PypicError)
    assert isinstance(exc, stdlib_base)
    # ``exc.detail`` is what the HTTP body and the WebSocket
    # :class:`ErrorFrame.message` carry; it must round-trip the message
    # without ``KeyError``'s ``repr``-quoting.
    assert exc.detail == "oops"


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
    pydantic = pytest.importorskip("pydantic")

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
