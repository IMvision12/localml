"""FastAPI application factory.

Phase 1: serve the compiled React build as static files at the root, plus a
health probe. API and OpenAI-compatible routers mount here in later phases -
they must be registered BEFORE the catch-all static mount so `/api/*` and
`/v1/*` win over the greedy `"/"` mount.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PY_DIR = Path(__file__).resolve().parents[1]
if str(_PY_DIR) not in sys.path:
    sys.path.insert(0, str(_PY_DIR))

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
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

    @app.get("/api/health")
    def health():
        return {"ok": True, "name": "inferml", "version": __version__}

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
