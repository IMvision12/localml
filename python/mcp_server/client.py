"""Async HTTP client for a running InferML server.

The MCP server is a *client* of `inferml`, not a second copy of the engine.
Inference is serialized behind one lock in the server process and torch is not
thread-safe against itself; embedding the engine here would double-load torch,
compete for the same GPU memory, and leave the UI and MCP with separate model
caches. Talking HTTP means both share whatever model is already warm.

Endpoint errors arrive in two shapes - OpenAI's `{"error": {"message": ...}}`
on /v1/*, and `{"ok": false, "error": ...}` on /api/* - and both become an
`InferMLError` carrying the server's own (already actionable) message.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

DEFAULT_URL = "http://127.0.0.1:11500"

# First use of a model downloads weights and warms a pipeline; diffusion on CPU
# can take minutes. Status probes should fail fast instead.
INFERENCE_TIMEOUT = 1800.0
PROBE_TIMEOUT = 5.0


class InferMLError(RuntimeError):
    """A request the server rejected, or a server we couldn't reach."""


class InferMLClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = INFERENCE_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _client(self, timeout: float | None = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url,
                                 timeout=timeout or self.timeout)

    def _unreachable(self, e: Exception) -> InferMLError:
        # Almost always one thing: the API is off. It is opt-in by design - it is
        # the only surface InferML exposes - so "nothing is listening" is the
        # normal state, not a fault. Say precisely how to change it. (This used to
        # advise running `inferml --no-browser`, a command from the pipx web app
        # that no longer exists, which left the one recoverable failure MCP has
        # with instructions nobody could follow.)
        return InferMLError(
            f"Can't reach InferML at {self.base_url} ({type(e).__name__}).\n\n"
            "The local API is probably switched off - it is opt-in. Open the "
            "InferML app and turn it on under Settings -> API & MCP, then retry. "
            "If the app is not running, start it first.\n\n"
            "(If your API listens elsewhere, point this server at it with --url, "
            "or the INFERML_URL environment variable.)"
        )

    @staticmethod
    def _raise_for_body(body: Any) -> Any:
        """Turn InferML's two error envelopes into exceptions."""
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict) and err.get("message"):
                raise InferMLError(str(err["message"]))
            if err and body.get("ok") is False:
                raise InferMLError(str(err))
            if err and "ok" not in body:
                raise InferMLError(str(err))
        return body

    async def get(self, path: str, *, timeout: float | None = None) -> Any:
        try:
            async with self._client(timeout) as c:
                r = await c.get(path)
        except httpx.RequestError as e:
            raise self._unreachable(e) from e
        return self._raise_for_body(r.json())

    async def post(self, path: str, body: dict, *, timeout: float | None = None) -> Any:
        try:
            async with self._client(timeout) as c:
                r = await c.post(path, json=body)
        except httpx.RequestError as e:
            raise self._unreachable(e) from e
        # Media endpoints can answer with raw bytes (audio/speech -> audio/wav).
        ctype = r.headers.get("content-type", "")
        if not ctype.startswith("application/json"):
            if r.status_code >= 400:
                raise InferMLError(r.text[:500])
            return r.content
        return self._raise_for_body(r.json())

    async def post_file(self, path: str, *, filename: str, content: bytes,
                        mime: str, data: dict) -> Any:
        try:
            async with self._client() as c:
                r = await c.post(path, files={"file": (filename, content, mime)}, data=data)
        except httpx.RequestError as e:
            raise self._unreachable(e) from e
        ctype = r.headers.get("content-type", "")
        if not ctype.startswith("application/json"):
            if r.status_code >= 400:
                raise InferMLError(r.text[:500])
            return r.text
        return self._raise_for_body(r.json())

    async def post_sse(self, path: str, body: dict) -> AsyncIterator[dict]:
        """Yield the decoded `data:` payload of each SSE frame."""
        try:
            async with self._client() as c:
                async with c.stream("POST", path, json=body) as r:
                    if r.status_code >= 400:
                        raise InferMLError(f"{path} failed with HTTP {r.status_code}")
                    async for line in r.aiter_lines():
                        if line.startswith("data:"):
                            try:
                                yield json.loads(line[5:].strip())
                            except json.JSONDecodeError:
                                continue
        except httpx.RequestError as e:
            raise self._unreachable(e) from e

    async def health(self) -> dict:
        return await self.get("/api/health", timeout=PROBE_TIMEOUT)

    async def download(self, model_id: str) -> dict:
        """Run the server's snapshot download to completion. Idempotent: an
        already-cached model returns almost immediately.

        This is also the supported way to make a diffusion model loadable on
        Windows - diffusers builds its own download patterns with os.path.join,
        so `from_pretrained` silently skips every component config.json there,
        while this route's patterns are forward-slash only.
        """
        result: dict = {}
        async for evt in self.post_sse("/api/download", {"modelId": model_id}):
            if evt.get("type") == "result":
                result = evt
        if not result:
            raise InferMLError(f"Download of {model_id!r} ended without a result.")
        if not result.get("ok"):
            if result.get("cancelled"):
                raise InferMLError(f"Download of {model_id!r} was cancelled.")
            raise InferMLError(result.get("error") or f"Download of {model_id!r} failed.")
        return result.get("info") or {}
