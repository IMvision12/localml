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

from fastapi import APIRouter, Body, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

import runtime as deps
from api.llm import (
    LLMNotLoaded, resolve_llm, build_inputs, generate_full, stream_generate,
)
from api.embeddings import embed, EmbeddingError
from api.media import (
    MediaError, resolve_media_model, installed_task,
    to_image_data_url, bytes_to_data_url, data_url_bytes, parse_size,
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
        from services.store_service import list_installed
        _servable = ("text-generation", "conversational",
                     "feature-extraction", "sentence-similarity",
                     "automatic-speech-recognition", "text-to-speech",
                     "object-detection", "zero-shot-object-detection",
                     "image-segmentation", "mask-generation", "text-to-image")
        for mid, meta in list_installed().items():
            if (meta or {}).get("task") in _servable and mid not in ids:
                ids.append(mid)
    except Exception:
        pass
    from api.embeddings import DEFAULT_EMBED_MODEL
    if DEFAULT_EMBED_MODEL not in ids:
        ids.append(DEFAULT_EMBED_MODEL)
    created = _now()
    return {
        "object": "list",
        "data": [{"id": mid, "object": "model", "created": created, "owned_by": "inferml"} for mid in ids],
    }

def _encode_embedding(row, enc_format: str):
    """OpenAI returns floats by default, or a base64 blob of little-endian
    float32 when `encoding_format` is "base64"."""
    if enc_format == "base64":
        import base64
        import numpy as np
        arr = np.asarray(row, dtype="<f4")
        return base64.b64encode(arr.tobytes()).decode("ascii")
    return [float(x) for x in row]

@router.post("/embeddings")
async def embeddings(payload: dict = Body(...)):
    inputs = payload.get("input")
    model_req = payload.get("model")
    enc_format = str(payload.get("encoding_format") or "float").lower()
    dimensions = payload.get("dimensions")
    if inputs is None:
        return JSONResponse(status_code=400, content=_err("`input` is required.", "invalid_request_error"))
    if enc_format not in ("float", "base64"):
        return JSONResponse(status_code=400,
                            content=_err(f"Unsupported encoding_format {enc_format!r}.", "invalid_request_error"))

    async with deps.INFERENCE_LOCK:
        try:
            result = await deps.run_blocking(embed, model_req, inputs, True, dimensions)
        except EmbeddingError as e:
            return JSONResponse(status_code=400, content=_err(str(e), "invalid_request_error"))
        except Exception as e:
            import traceback
            from engine import actionable_error
            print("[/v1/embeddings] " + traceback.format_exc(), flush=True)
            return JSONResponse(status_code=500, content=_err(actionable_error(e) or repr(e), "server_error"))

    data = [
        {"object": "embedding", "index": i, "embedding": _encode_embedding(row, enc_format)}
        for i, row in enumerate(result["vectors"])
    ]
    return {
        "object": "list",
        "data": data,
        "model": result["model_id"],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "total_tokens": result["prompt_tokens"],
        },
    }

# --- media endpoints ----------------------------------------------------------
# All of these reuse the engine's task handlers (python/tasks/, adapters) via
# ENGINE.run() - same code path as the UI workspaces - and only shape HTTP here.

async def _media_run(model_req, tasks: tuple, fallback, inputs: dict, params: dict,
                     task_hint: str | None = None):
    """Resolve `model`, run one inference behind the shared lock, and return
    (model_id, output_kinds dict). Raises; callers map errors via _media_err."""
    mid = resolve_media_model(model_req, tasks, fallback)
    hint = installed_task(mid) or task_hint
    eng = deps.engine()
    async with deps.INFERENCE_LOCK:
        deps.clear_stop()
        out = await deps.run_blocking(eng.run, mid, hint, inputs, params)
    return mid, out

def _media_err(e: Exception):
    if isinstance(e, (MediaError, ValueError)):
        return JSONResponse(status_code=400, content=_err(str(e), "invalid_request_error"))
    import traceback
    from engine import actionable_error
    print("[/v1 media] " + traceback.format_exc(), flush=True)
    return JSONResponse(status_code=500, content=_err(actionable_error(e) or repr(e), "server_error"))

def _audio_duration_s(raw: bytes):
    """Best-effort clip duration from the container header (for verbose_json)."""
    try:
        import io
        import soundfile as sf
        info = sf.info(io.BytesIO(raw))
        return round(float(info.frames) / float(info.samplerate or 1), 2)
    except Exception:
        return None

@router.post("/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float | None = Form(default=None),
):
    """OpenAI-compatible speech-to-text. Runs any ASR model (Whisper, Wav2Vec2,
    …). `language`/`prompt`/`temperature` are accepted for SDK compatibility but
    not forwarded - the ASR task doesn't take them."""
    fmt = (response_format or "json").strip().lower()
    if fmt not in ("json", "text", "verbose_json"):
        return JSONResponse(status_code=400, content=_err(
            f"response_format {fmt!r} is not supported - use json, text, or verbose_json.",
            "invalid_request_error"))
    raw = await file.read()
    if not raw:
        return JSONResponse(status_code=400, content=_err("`file` is empty.", "invalid_request_error"))
    mime = file.content_type or "audio/wav"
    try:
        mid, out = await _media_run(
            model, ("automatic-speech-recognition",), "openai/whisper-tiny",
            {"dataUrl": bytes_to_data_url(raw, mime)}, {},
            task_hint="automatic-speech-recognition")
    except Exception as e:
        return _media_err(e)
    text = out.get("text") or ""
    if fmt == "text":
        return PlainTextResponse(text)
    if fmt == "verbose_json":
        return {"task": "transcribe", "language": language or "",
                "duration": _audio_duration_s(raw), "text": text, "segments": []}
    return {"text": text}

# Arbitrary-but-stable mapping of OpenAI voice names onto CMU-Arctic x-vector
# speaker indices (SpeechT5 only; other TTS models ignore speaker_index).
_NAMED_VOICES = {"alloy": 7306, "echo": 6799, "fable": 6671,
                 "onyx": 3922, "nova": 7067, "shimmer": 5750}

@router.post("/audio/speech")
async def audio_speech(payload: dict = Body(...)):
    """OpenAI-compatible text-to-speech. Always returns WAV bytes (no local
    mp3/opus transcoder), whatever `response_format` asks for."""
    text = str(payload.get("input") or "").strip()
    if not text:
        return JSONResponse(status_code=400, content=_err("`input` is required.", "invalid_request_error"))
    params: dict = {}
    v = str(payload.get("voice") or "").strip().lower()
    if v.isdigit():
        params["speaker_index"] = int(v)
    elif v in _NAMED_VOICES:
        params["speaker_index"] = _NAMED_VOICES[v]
    try:
        mid, out = await _media_run(
            payload.get("model"), ("text-to-speech",), "microsoft/speecht5_tts",
            {"text": text}, params, task_hint="text-to-speech")
    except Exception as e:
        return _media_err(e)
    wav = data_url_bytes(out["dataUrl"])
    return Response(content=wav, media_type="audio/wav",
                    headers={"X-InferML-Model": mid})

@router.post("/image/detection")
async def image_detection(payload: dict = Body(...)):
    """Object detection. `image` is base64 or a data URL. Pass `labels`
    (list of strings) to run zero-shot detection (OWL-ViT, Grounding-DINO)
    instead of a fixed-vocabulary detector. Boxes are normalized [0,1]."""
    labels = payload.get("labels")
    if labels is not None and (not isinstance(labels, list) or not all(isinstance(x, str) for x in labels)):
        return JSONResponse(status_code=400, content=_err("`labels` must be an array of strings.", "invalid_request_error"))
    params: dict = {}
    if payload.get("threshold") is not None:
        params["threshold"] = float(payload["threshold"])
    if payload.get("nms_iou") is not None:
        params["nms_iou"] = float(payload["nms_iou"])
    try:
        inputs = {"dataUrl": to_image_data_url(payload.get("image"))}
        if labels:
            params["candidate_labels"] = [s.strip() for s in labels if s.strip()]
            inputs["text"] = ", ".join(params["candidate_labels"])
            tasks, fallback, hint = (("zero-shot-object-detection",),
                                     "IDEA-Research/grounding-dino-tiny",
                                     "zero-shot-object-detection")
        else:
            tasks, fallback, hint = (("object-detection",),
                                     "facebook/detr-resnet-50",
                                     "object-detection")
        mid, out = await _media_run(payload.get("model"), tasks, fallback, inputs, params, task_hint=hint)
    except Exception as e:
        return _media_err(e)
    data = [{"label": b.get("label"), "score": b.get("score"),
             "box": {"x": b["box"][0], "y": b["box"][1], "w": b["box"][2], "h": b["box"][3]}}
            for b in (out.get("boxes") or [])]
    resp = {"object": "list", "created": _now(), "model": mid, "data": data}
    if payload.get("annotated"):
        resp["annotated"] = out.get("annotated")
    return resp

@router.post("/image/segmentation")
async def image_segmentation(payload: dict = Body(...)):
    """Image segmentation - semantic / instance / panoptic (routed by the
    model, e.g. SegFormer / Mask2Former) and SAM auto-mask generation. Returns
    an RGBA overlay PNG (data URL) + a legend."""
    params = {k: payload[k] for k in
              ("overlay_alpha", "legend_min_pct", "points_per_batch", "min_mask_pct", "max_masks")
              if payload.get(k) is not None}
    try:
        inputs = {"dataUrl": to_image_data_url(payload.get("image"))}
        mid, out = await _media_run(
            payload.get("model"), ("image-segmentation", "mask-generation"),
            "nvidia/segformer-b0-finetuned-ade-512-512",
            inputs, params, task_hint="image-segmentation")
    except Exception as e:
        return _media_err(e)
    return {"model": mid, "created": _now(),
            "overlay": out.get("overlay"), "legend": out.get("legend") or []}

@router.post("/image/generation")
@router.post("/images/generations")  # OpenAI-compatible alias
async def image_generation(payload: dict = Body(...)):
    """Text-to-image via the diffusion adapters (SD, SDXL, FLUX, …). Returns
    b64_json PNGs regardless of `response_format` (there's no URL hosting).
    No default model - diffusion weights are GB-scale, so downloads must be
    explicit: pass an HF id or install a text-to-image model in InferML."""
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse(status_code=400, content=_err("`prompt` is required.", "invalid_request_error"))
    n = max(1, min(int(payload.get("n") or 1), 4))
    params: dict = dict(parse_size(payload.get("size")))
    steps = payload.get("steps") or payload.get("num_inference_steps")
    if steps:
        params["num_inference_steps"] = int(steps)
    if payload.get("guidance_scale") is not None:
        params["guidance_scale"] = float(payload["guidance_scale"])
    if payload.get("negative_prompt"):
        params["negative_prompt"] = str(payload["negative_prompt"])
    data = []
    mid = None
    try:
        for _ in range(n):
            mid, out = await _media_run(
                payload.get("model"), ("text-to-image",), None,
                {"text": prompt}, params, task_hint="text-to-image")
            data.append({"b64_json": out["dataUrl"].split(",", 1)[1]})
    except Exception as e:
        return _media_err(e)
    return {"created": _now(), "model": mid, "data": data}

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
        from api.tools import parse_tool_calls, ToolFormatUnknown
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
            from api.tools import parse_tool_calls, ToolFormatUnknown
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
