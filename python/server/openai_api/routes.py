"""OpenAI-compatible routes: /v1/chat/completions and /v1/models.

Drop-in target for LangChain `ChatOpenAI`, the OpenAI SDK, LangGraph, etc.:
point `base_url` at http://localhost:PORT/v1 with any api_key. Routes to the
LLM currently loaded in InferML (or lazy-loads one named in `model`).
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse, StreamingResponse

from server import deps
from server.openai_api.llm import (
    LLMNotLoaded, resolve_llm, build_inputs, generate_full, stream_generate,
)

router = APIRouter(prefix="/v1")

def _rid() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]

def _now() -> int:
    return int(time.time())

def _finish_reason(completion_tokens: int, params: dict) -> str:
    return "length" if completion_tokens >= int(params.get("max_tokens") or 512) else "stop"

def _params_from(payload: dict) -> dict:
    return {
        "max_tokens": payload.get("max_tokens") or payload.get("max_completion_tokens"),
        "temperature": payload.get("temperature"),
        "top_p": payload.get("top_p"),
        "stop": payload.get("stop"),
    }

@router.get("/models")
async def list_models():
    eng = deps.engine()
    ids = list(eng.loaded_model_ids())
    cur = eng.current_llm_id()
    if cur and cur not in ids:
        ids.append(cur)
    try:
        from server.store_service import list_installed
        for mid, meta in list_installed().items():
            if (meta or {}).get("task") in ("text-generation", "conversational") and mid not in ids:
                ids.append(mid)
    except Exception:
        pass
    created = _now()
    return {
        "object": "list",
        "data": [{"id": mid, "object": "model", "created": created, "owned_by": "inferml"} for mid in ids],
    }

@router.post("/chat/completions")
async def chat_completions(payload: dict = Body(...)):
    messages = payload.get("messages") or []
    stream = bool(payload.get("stream"))
    model_req = payload.get("model")
    tools = payload.get("tools")
    tool_choice = payload.get("tool_choice")
    params = _params_from(payload)
    eng = deps.engine()

    if stream:
        return StreamingResponse(
            _stream_response(eng, model_req, messages, tools, tool_choice, params),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async with deps.INFERENCE_LOCK:
        deps.clear_stop()
        try:
            model, tokenizer, model_id = await deps.run_blocking(resolve_llm, eng, model_req)
        except LLMNotLoaded as e:
            return JSONResponse(status_code=400, content=_err(str(e), "invalid_request_error"))

        def _work():
            input_ids = build_inputs(tokenizer, messages, tools if tools else None)
            text, ptoks, ctoks = generate_full(model, tokenizer, input_ids, params)
            return text, ptoks, ctoks

        try:
            text, ptoks, ctoks = await deps.run_blocking(_work)
        except Exception as e:
            import traceback
            from engine import actionable_error
            print("[/v1/chat/completions] " + traceback.format_exc(), flush=True)
            return JSONResponse(status_code=500, content=_err(actionable_error(e) or repr(e), "server_error"))

    message = {"role": "assistant", "content": text}
    finish = _finish_reason(ctoks, params)
    if tools:
        from server.openai_api.tools import parse_tool_calls, ToolFormatUnknown
        try:
            tool_calls = parse_tool_calls(model_id, text)
        except ToolFormatUnknown as e:
            return JSONResponse(status_code=422, content=_err(str(e), "invalid_request_error"))
        if tool_calls:
            message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
            finish = "tool_calls"

    return {
        "id": _rid(),
        "object": "chat.completion",
        "created": _now(),
        "model": model_id,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": ptoks,
            "completion_tokens": ctoks,
            "total_tokens": ptoks + ctoks,
        },
    }

async def _stream_response(eng, model_req, messages, tools, tool_choice, params):
    rid = _rid()
    created = _now()
    include_usage = False

    async with deps.INFERENCE_LOCK:
        deps.clear_stop()
        try:
            model, tokenizer, model_id = await deps.run_blocking(resolve_llm, eng, model_req)
        except LLMNotLoaded as e:
            yield _sse_error(rid, created, "unknown", str(e))
            yield "data: [DONE]\n\n"
            return

        if tools:
            try:
                text, _p, ctoks = await deps.run_blocking(
                    lambda: generate_full(model, tokenizer, build_inputs(tokenizer, messages, tools), params)
                )
            except Exception as e:
                from engine import actionable_error
                yield _sse_error(rid, created, model_id, actionable_error(e))
                yield "data: [DONE]\n\n"
                return
            from server.openai_api.tools import parse_tool_calls, ToolFormatUnknown
            try:
                calls = parse_tool_calls(model_id, text)
            except ToolFormatUnknown as e:
                yield _sse_error(rid, created, model_id, str(e))
                yield "data: [DONE]\n\n"
                return
            yield _sse_chunk(rid, created, model_id, {"role": "assistant"})
            if calls:
                yield _sse_chunk(rid, created, model_id, {"tool_calls": calls})
                yield _sse_final(rid, created, model_id, "tool_calls")
            else:
                if text:
                    yield _sse_chunk(rid, created, model_id, {"content": text})
                yield _sse_final(rid, created, model_id, _finish_reason(ctoks, params))
            yield "data: [DONE]\n\n"
            return

        try:
            input_ids = await deps.run_blocking(build_inputs, tokenizer, messages, None)
        except Exception as e:
            yield _sse_error(rid, created, model_id, str(e))
            yield "data: [DONE]\n\n"
            return

        yield _sse_chunk(rid, created, model_id, {"role": "assistant"})
        produced = 0
        try:
            async for delta in _aiter_sync(
                lambda: stream_generate(model, tokenizer, input_ids, params, deps.stop_requested)
            ):
                produced += 1
                if delta:
                    yield _sse_chunk(rid, created, model_id, {"content": delta})
        except Exception as e:
            from engine import actionable_error
            yield _sse_error(rid, created, model_id, actionable_error(e))
            yield "data: [DONE]\n\n"
            return

        yield _sse_final(rid, created, model_id, "stop")
        yield "data: [DONE]\n\n"

async def _aiter_sync(gen_factory):
    """Drain a blocking generator (running on a worker thread) as an async
    iterator, so token production never blocks the event loop."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    def worker():
        try:
            for item in gen_factory():
                loop.call_soon_threadsafe(q.put_nowait, ("item", item))
        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, ("error", e))
        finally:
            loop.call_soon_threadsafe(q.put_nowait, ("done", _DONE))

    threading.Thread(target=worker, name="oai-stream", daemon=True).start()
    while True:
        kind, val = await q.get()
        if kind == "done":
            break
        if kind == "error":
            raise val
        yield val

def _chunk_obj(rid, created, model_id, delta, finish=None):
    return {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }

def _sse_chunk(rid, created, model_id, delta):
    return "data: " + json.dumps(_chunk_obj(rid, created, model_id, delta)) + "\n\n"

def _sse_final(rid, created, model_id, finish):
    return "data: " + json.dumps(_chunk_obj(rid, created, model_id, {}, finish)) + "\n\n"

def _sse_error(rid, created, model_id, message):
    obj = _chunk_obj(rid, created, model_id, {"content": f"[error] {message}"}, "stop")
    return "data: " + json.dumps(obj) + "\n\n"

def _err(message: str, etype: str) -> dict:
    return {"error": {"message": message, "type": etype, "code": None}}
