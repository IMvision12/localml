"""The local HTTP API - the one thing InferML deliberately exposes.

This is an *optional* listener bolted onto the side of the engine. The desktop
app does not use it and never talks to it: the window is a local file driven over
stdio, so the UI works whether this is running or not. Turning it on adds exactly
one capability - other programs on this machine can drive your loaded models.

What it serves
--------------
  /v1/*     OpenAI-compatible. Point LangChain / the OpenAI SDK here.
  /api/*    Six endpoints, and only the six the MCP server calls.

What it does NOT serve
----------------------
The app's interface. There is no static mount, no index.html, nothing to render.
Pointing a browser at the root gets a 404 and always will - this file has no way
to hand out the UI even if someone asked it to. That is the difference from the
old web build, where the same port served both the API and the whole application.

Security posture
----------------
  - binds 127.0.0.1 only, never 0.0.0.0. It is not reachable from your network.
  - sends no CORS headers, so a web page you visit cannot *read* a response from
    it, even though it can technically issue a request.
  - exposes no token/credential route. `/api/hf/token` existed on the old server
    and is deliberately not here: any local process could have read your
    HuggingFace token out of it.

It is still unauthenticated, which means any process running as you can use it.
That is the same trade Ollama makes, and it is why this is a setting the user
turns on rather than something that is simply always listening.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading

from fastapi import Body, FastAPI
from fastapi.responses import StreamingResponse

import runtime
from api.routes import router as v1_router
from engine import DownloadCancelled, actionable_error
from services import hf_service as hf
from services import store_service as store

HOST = "127.0.0.1"
DEFAULT_PORT = 11500


def create_app() -> FastAPI:
    app = FastAPI(title="InferML API", docs_url=None, redoc_url=None)

    app.include_router(v1_router)

    @app.get("/api/health")
    async def health():
        # The version is Electron's (package.json), handed down at spawn - the
        # MCP server's `inferml_status` tool reads it from here.
        return {"ok": True, "name": "inferml", "version": os.environ.get("INFERML_VERSION")}

    @app.get("/api/status")
    async def status():
        return runtime.probe_status()

    @app.get("/api/models")
    async def models():
        eng = runtime.engine()
        return {"loaded": eng.loaded_model_ids(), "currentLlm": eng.current_llm_id()}

    @app.get("/api/hf/installed")
    async def installed():
        return store.list_installed()

    @app.get("/api/hf/search")
    async def search(q: str | None = None, task: str | None = None):
        try:
            return await runtime.run_blocking(hf.search, q, task)
        except Exception as e:
            return {"error": str(e)}

    @app.post("/api/download")
    async def download(payload: dict = Body(...)):
        """Fetch a model, streaming progress as SSE.

        MCP leans on this for more than convenience: on Windows, diffusers builds
        its own download patterns with os.path.join, so `from_pretrained` skips
        every component config.json and the load fails from a cold cache. This
        route's patterns are forward-slash only, so pre-warming through it is the
        supported way to make a diffusion model loadable at all.
        """
        model_id = payload.get("modelId")
        if not model_id:
            return {"ok": False, "error": "modelId required"}

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancel = threading.Event()

        def worker():
            def on_progress(evt: dict) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", **evt})
            try:
                info = runtime.engine().download(model_id, on_progress=on_progress, cancel_event=cancel)
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "result", "ok": True, "info": info})
            except DownloadCancelled:
                loop.call_soon_threadsafe(queue.put_nowait,
                                          {"type": "result", "ok": False, "cancelled": True, "error": "cancelled"})
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait,
                                          {"type": "result", "ok": False, "error": actionable_error(e)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "__done__"})

        threading.Thread(target=worker, name=f"api-download-{model_id}", daemon=True).start()

        async def stream():
            while True:
                msg = await queue.get()
                if msg.get("type") == "__done__":
                    break
                yield "data: " + json.dumps({"modelId": model_id, **msg}) + "\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


class APIServer:
    """uvicorn on a background thread, startable and stoppable at runtime.

    It runs inside the engine process on purpose. The API has to reach live
    adapters - to stream tokens out of a loaded model, and to reuse a model the
    user already opened in the app instead of loading a second multi-GB copy of
    it. A separate process could not do either.
    """

    def __init__(self) -> None:
        self._server = None
        self._thread = None
        self._port = None
        self._error = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and self._server and self._server.started)

    @property
    def url(self) -> str | None:
        return f"http://{HOST}:{self._port}" if self._port else None

    def status(self) -> dict:
        return {
            "running": self.running,
            "port": self._port,
            "url": self.url if self.running else None,
            "error": self._error,
        }

    def start(self, port: int = DEFAULT_PORT) -> dict:
        if self.running:
            return self.status()

        import uvicorn

        self._error = None
        self._port = int(port or DEFAULT_PORT)

        config = uvicorn.Config(
            create_app(),
            host=HOST,
            port=self._port,
            log_level="warning",
            # uvicorn's default handlers would write to stdout - which is the
            # stdio protocol's wire. Anything it logs must go to stderr instead.
            log_config=None,
            access_log=False,
        )
        self._server = uvicorn.Server(config)

        def serve():
            try:
                asyncio.run(self._server.serve())
            except Exception as e:                      # port in use is the common one
                self._error = str(e)

        self._thread = threading.Thread(target=serve, name="api-server", daemon=True)
        self._thread.start()

        # Give it a moment to bind, so the caller learns about "port in use" now
        # rather than discovering it later from a silently-dead server.
        for _ in range(50):
            if self.running or self._error:
                break
            threading.Event().wait(0.1)

        if self._error:
            self._server = None
            self._port = None
        return self.status()

    def stop(self) -> dict:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None
        self._port = None
        return self.status()


SERVER = APIServer()
