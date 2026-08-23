"""Starlette/FastAPI server that streams pypic data to webpic.

The server exposes:

* JSON HTTP routes for discovery (``/health``, ``/sims`` and children).
* A WebSocket endpoint (``/sims/{sim}/stream``) that consumes a JSON
  ``SubscribeRequest`` frame and replies with one Arrow IPC binary frame
  carrying the requested [`FieldDataset`][pypic.dataset.FieldDataset] slice.

The Arrow-IPC choice is deliberate: it has a browser-native reader in
the ``apache-arrow`` npm package (``tableFromIPC``), a Rust reader in
``arrow-rs``, and no gRPC-Web / Envoy proxy overhead.  Arrow Flight has
no shipping JS client, so it is not used here.

Usage::

    from pypic.server import create_app
    import uvicorn

    app = create_app(root="/data/runs")
    uvicorn.run(app, host="127.0.0.1", port=8000)

Or via the bundled CLI: ``pypic serve /data/runs --port 8000``.

The server package is optional — install with ``pip install "pypic-plasma[server]"``.
"""

from __future__ import annotations

__all__ = ["create_app", "serve"]


def __getattr__(name: str) -> object:
    # Lazy import so ``from pypic import server`` doesn't require the
    # FastAPI / uvicorn extras to be installed unless the user actually
    # constructs an app or runs the server.
    if name == "create_app":
        from pypic.server.app import create_app

        return create_app
    if name == "serve":
        from pypic.server.app import serve

        return serve
    msg = f"module 'pypic.server' has no attribute {name!r}"
    raise AttributeError(msg)
