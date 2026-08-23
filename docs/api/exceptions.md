# Exceptions

Every error pypic raises deliberately derives from `PypicError`, and each
subclass also inherits the built-in exception a caller would naturally reach
for — so `except KeyError` and `except NotImplementedError` keep working while
the typed class carries the extra routing metadata.

| Exception | Also a | `kind` | HTTP |
|---|---|---|---|
| `UnknownSimulationError` | `KeyError` | `unknown_simulation` | 404 |
| `UnknownFieldError` | `KeyError` | `unknown_field` | 404 |
| `UnknownStepError` | `KeyError` | `unknown_step` | 404 |
| `GeometryUnsupportedError` | `NotImplementedError` | `geometry_unsupported` | 400 |

`kind` and `status_code` exist so one dispatcher can serve both transports in
[`pypic.server`](server.md): `status_code` becomes the HTTP response code, and
`kind` becomes the `ErrorFrame.kind` on the WebSocket stream. Raising a bare
`KeyError` where one of these applies costs that routing — the request
degrades to a generic 500 / `kind="internal"`.

::: pypic.exceptions
