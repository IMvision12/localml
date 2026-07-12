"""FastAPI application factory.

Serves the compiled React build (the desktop app's interface) plus the /api and
/v1 routers. Routers are registered BEFORE the catch-all static mount so `/api/*`
and `/v1/*` win over the greedy `"/"` mount.

The UI is gated to the Electron shell - see `_shell_only_ui` below.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_PY_DIR = Path(__file__).resolve().parents[1]
if str(_PY_DIR) not in sys.path:
    sys.path.insert(0, str(_PY_DIR))

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from server import __version__
from server.paths import webui_dir

@asynccontextmanager
async def _lifespan(app: FastAPI):
    from server.routes.system import hw_poller
    poller = asyncio.create_task(hw_poller())
    try:
        yield
    finally:
        poller.cancel()

def create_app() -> FastAPI:
    app = FastAPI(title="InferML", version=__version__, lifespan=_lifespan)

    # The UI is for the desktop shell, not for browsers.
    #
    # Electron renders web content - that is simply what Chromium does - so the
    # interface is HTML served over loopback. Left open, that means anyone can
    # point a browser at localhost and drive the whole app. So the shell passes a
    # per-launch secret (INFERML_UI_TOKEN, generated in src/main/sidecar.js) and
    # the server hands the UI to holders of that secret only. A browser can reach
    # the port but cannot guess the secret, so it gets a 403.
    #
    # /api and /v1 are intentionally NOT gated: the OpenAI-compatible endpoint is
    # meant to be called by agent frameworks, and the MCP server talks to /api.
    # Both are localhost-only, and because no CORS headers are sent, a webpage you
    # visit still cannot read their responses.
    #
    # With no token in the environment (running `python -m server.cli` by hand to
    # debug the backend) the gate is off, so that path keeps working.
    ui_token = os.environ.get("INFERML_UI_TOKEN")

    @app.middleware("http")
    async def _shell_only_ui(request, call_next):
        path = request.url.path
        is_api = path.startswith("/api") or path.startswith("/v1")
        if ui_token and not is_api:
            if request.headers.get("x-inferml-shell") != ui_token:
                return PlainTextResponse(
                    "InferML is a desktop app - its interface isn't served to browsers.\n"
                    "Open the InferML app instead.\n\n"
                    "(The OpenAI-compatible API is still available at /v1.)",
                    status_code=403,
                )
        return await call_next(request)

    # Never let the shell cache the UI.
    #
    # The window is a Chromium instance with a normal HTTP cache, and StaticFiles
    # serves the UI with an ETag/Last-Modified that invites caching. That's a trap
    # here: an app update swaps in a new webui/ on disk, but the renderer can keep
    # painting the *previous* build's CSS/JS from cache - producing a UI that
    # doesn't match the code, with no obvious way for a user to force a reload
    # (the app has no menu, so no Ctrl+R).
    #
    # There is nothing to gain by caching either: this is localhost, off a local
    # disk. So the UI is always served fresh. /api and /v1 are untouched - they're
    # dynamic and uncached anyway.
    @app.middleware("http")
    async def _no_store_for_ui(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if not (path.startswith("/api") or path.startswith("/v1")):
            response.headers["Cache-Control"] = "no-store, must-revalidate"
            # Starlette's MutableHeaders has no .pop(); del is the way. Drop the
            # validators too, so a cached copy can't be revalidated into reuse.
            for stale in ("etag", "last-modified"):
                if stale in response.headers:
                    del response.headers[stale]
        return response

    @app.get("/api/health")
    def health():
        return {"ok": True, "name": "inferml", "version": __version__}

    # Updates are handled by the Electron shell (electron-updater against GitHub
    # Releases), not by the server - it can't replace the app bundle it runs in.
    from server.routes import inference, hf, store, system
    from server.openai_api import routes as openai_routes
    app.include_router(inference.router)
    app.include_router(hf.router)
    app.include_router(store.router)
    app.include_router(system.router)
    app.include_router(openai_routes.router)

    webui = webui_dir()
    index = webui / "index.html"
    if index.exists():
        app.mount("/", StaticFiles(directory=str(webui), html=True), name="webui")
    else:
        @app.get("/")
        def _needs_build():
            return JSONResponse(
                status_code=503,
                content={
                    "ok": False,
                    "error": (
                        "Frontend not built. Run `npm run build:renderer` - "
                        f"expected a compiled build at {webui}."
                    ),
                },
            )

    return app

app = create_app()
