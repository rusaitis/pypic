# Exceptions

Every error pypic raises deliberately derives from `PypicError`, and each
subclass also inherits the built-in exception a caller would naturally reach
for — so `except KeyError` and `except NotImplementedError` keep working while
the typed class lets a caller dispatch on type instead of on message text.

| Exception | Also a | `kind` | HTTP |
|---|---|---|---|
| `UnknownSimulationError` | `KeyError` | `unknown_sim` | 404 |
| `UnknownFieldError` | `KeyError` | `unknown_field` | 404 |
| `UnknownStepError` | `KeyError` | `unknown_step` | 404 |
| `GeometryUnsupportedError` | `NotImplementedError` | `geometry_unsupported` | 400 |

The `kind` and HTTP columns are not properties of these classes — they live in
`pypic.server.exceptions.error_routing`, the one table
[`pypic.server`](server.md) consults so a single dispatcher serves both
transports: the status becomes the HTTP response code, the kind becomes
`ErrorFrame.kind` on the WebSocket stream. Raising a bare `KeyError` where one
of these applies costs that routing — the request degrades to a generic
500 / `kind="internal"`.

::: pypic.exceptions
