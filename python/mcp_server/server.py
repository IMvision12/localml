"""MCP server exposing InferML's local models to Claude and other LLM clients.

Wraps a running `inferml` server (see client.py for why it's a client, not an
engine embed) as MCP tools: detection, segmentation, transcription, speech,
image generation, text generation, embeddings, plus model discovery/install.

Run:
    inferml --no-browser          # terminal 1: the model server
    inferml-mcp                   # terminal 2: this, over stdio

Claude Code:
    claude mcp add inferml -- inferml-mcp

Conventions:
  - Media *inputs* are local file paths. LLM clients have paths, not base64.
  - Large binary *outputs* (audio, generated images) are written to
    --output-dir and the path is returned; images are additionally inlined so
    the model can actually look at them. Vectors are never inlined - a 768-dim
    float array is pure context bloat - they're summarized instead.
"""
from __future__ import annotations

import argparse
import base64
import io
import mimetypes
import os
import wave
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP, Image

from mcp_server.client import DEFAULT_URL, InferMLClient, InferMLError

INSTRUCTIONS = """Runs Hugging Face models locally through InferML.

Requires the InferML server to be running (`inferml --no-browser`). Call
`inferml_status` first if a tool fails - it reports whether the server is up
and which inference stack is installed.

Models are Hugging Face repo ids (e.g. "hustvl/yolos-tiny"). Most tools have a
small default model; `generate_image` has none, because diffusion weights are
gigabytes - pass an explicit `model`. Use `search_models` to find candidates
and `download_model` to fetch one ahead of time.
"""

mcp = FastMCP("inferml", instructions=INSTRUCTIONS)

_client = InferMLClient()
_output_dir = Path.home() / "inferml-outputs"


def _out_path(name: str, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    _output_dir.mkdir(parents=True, exist_ok=True)
    return _output_dir / name


def _read_media(path: str, expect: str) -> tuple[bytes, str]:
    """Load a local image/audio file. Returns (bytes, mime)."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise InferMLError(f"No such file: {p}")
    mime, _ = mimetypes.guess_type(str(p))
    mime = mime or ("image/png" if expect == "image" else "audio/wav")
    if not mime.startswith(expect):
        raise InferMLError(f"{p.name} looks like {mime}, expected {expect}/*.")
    return p.read_bytes(), mime


def _image_data_url(path: str) -> str:
    raw, mime = _read_media(path, "image")
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def _decode_data_url(u: str) -> bytes:
    return base64.b64decode(u.split(",", 1)[1] if "," in u else u)


def _wav_duration(raw: bytes) -> float | None:
    """Duration from the RIFF header.

    Must stay stdlib-only. Importing soundfile here hangs the stdio server
    indefinitely: the tool call never returns and the client sees a dead
    transport. Nothing in this process should reach for the inference stack -
    the `mcp` extra doesn't install it.
    """
    try:
        with wave.open(io.BytesIO(raw)) as w:
            return round(w.getnframes() / float(w.getframerate()), 2)
    except Exception:
        return None


# --- discovery ---------------------------------------------------------------

@mcp.tool()
async def inferml_status() -> dict:
    """Check that the InferML server is reachable and ready to run models.

    Reports the inference stack's readiness, any missing packages, the active
    accelerator (cpu/gpu), and which models are currently loaded in memory.
    Call this first when another tool fails unexpectedly.
    """
    health = await _client.health()
    status = await _client.get("/api/status", timeout=10.0)
    models = await _client.get("/api/models", timeout=10.0)
    return {
        "server": {"url": _client.base_url, "version": health.get("version")},
        "ready": status.get("ready"),
        "missing_packages": status.get("missing") or [],
        "accelerator": status.get("activeAccelerator"),
        "torch": status.get("torch"),
        "loaded_models": models.get("loaded") or [],
        "current_llm": models.get("currentLlm"),
    }


@mcp.tool()
async def list_models() -> dict:
    """List the models InferML can serve right now.

    `loaded` are resident in memory (fastest); `available` also includes models
    downloaded to the local Hugging Face cache.
    """
    loaded = await _client.get("/api/models", timeout=10.0)
    served = await _client.get("/v1/models", timeout=10.0)
    installed = await _client.get("/api/hf/installed", timeout=10.0)
    return {
        "loaded": loaded.get("loaded") or [],
        "current_llm": loaded.get("currentLlm"),
        "available": [m["id"] for m in (served.get("data") or [])],
        "installed": [
            {"id": mid, "task": (meta or {}).get("task")}
            for mid, meta in (installed or {}).items()
        ],
    }


@mcp.tool()
async def search_models(query: str = "", task: str = "", limit: int = 10) -> list[dict]:
    """Search Hugging Face for models InferML can actually run.

    Results are pre-filtered to architectures InferML supports, so anything
    returned here will load (unsupported runtimes like GGUF/Ultralytics are
    excluded). Narrow with `task`, e.g. "object-detection",
    "image-segmentation", "automatic-speech-recognition", "text-to-speech",
    "text-to-image", "text-generation", "feature-extraction".
    """
    params = []
    if query:
        params.append(f"q={query}")
    if task:
        params.append(f"task={task}")
    qs = ("?" + "&".join(params)) if params else ""
    body = await _client.get("/api/hf/search" + qs, timeout=60.0)
    items = body.get("items") if isinstance(body, dict) else body
    out = []
    for m in (items or [])[: max(1, min(limit, 50))]:
        out.append({
            "id": m.get("id"),
            "task": m.get("task"),
            "size": m.get("size"),
            "downloads": m.get("dl"),
            "installed": m.get("installed"),
            "description": (m.get("desc") or "")[:160],
        })
    return out


@mcp.tool()
async def download_model(model_id: str) -> dict:
    """Download a model's weights into the local cache.

    Idempotent - an already-cached model returns immediately. Large models take
    minutes. Downloading ahead of time is also the reliable way to prepare a
    diffusion model for `generate_image` on Windows.
    """
    info = await _client.download(model_id)
    return {
        "model": model_id,
        "downloaded_bytes": info.get("bytes"),
        "path": info.get("path"),
    }


# --- vision ------------------------------------------------------------------

@mcp.tool()
async def detect_objects(image_path: str, labels: list[str] | None = None,
                         threshold: float | None = None,
                         model: str = "") -> dict:
    """Detect objects in an image and return their labels, scores, and boxes.

    Pass `labels` (e.g. ["a cat", "a traffic light"]) to run *open-vocabulary*
    detection, which finds arbitrary things described in words. Omit it to use
    a fixed-vocabulary detector over the 80 COCO classes.

    Boxes are normalized to [0,1] as {x, y, w, h} with the origin top-left,
    so they're resolution-independent.
    """
    payload: dict[str, Any] = {"image": _image_data_url(image_path)}
    if model:
        payload["model"] = model
    if threshold is not None:
        payload["threshold"] = threshold
    if labels:
        payload["labels"] = labels
    body = await _client.post("/v1/image/detection", payload)
    return {
        "model": body.get("model"),
        "count": len(body.get("data") or []),
        "objects": body.get("data") or [],
    }


@mcp.tool()
async def segment_image(image_path: str, model: str = "",
                        output_path: str = "") -> list:
    """Segment an image into labelled regions and show the result.

    Handles semantic, instance, and panoptic segmentation (SegFormer,
    Mask2Former, OneFormer, ...) and SAM-style automatic mask generation - the
    model you pass decides which. SAM produces unlabelled "region N" masks
    because it has no class vocabulary.

    Returns a colour-coded overlay image plus the legend mapping colours to
    classes.
    """
    payload: dict[str, Any] = {"image": _image_data_url(image_path)}
    if model:
        payload["model"] = model
    body = await _client.post("/v1/image/segmentation", payload)

    overlay = _decode_data_url(body["overlay"])
    dest = _out_path(f"segmentation-{Path(image_path).stem}.png", output_path)
    dest.write_bytes(overlay)

    legend = body.get("legend") or []
    summary = "\n".join(f"  {e.get('color')}  {e.get('label')}" for e in legend)
    return [
        f"Segmented with {body.get('model')}: {len(legend)} regions.\n"
        f"Overlay saved to {dest}\n\nLegend:\n{summary}",
        Image(data=overlay, format="png"),
    ]


@mcp.tool()
async def generate_image(prompt: str, model: str, size: str = "",
                         steps: int | None = None,
                         guidance_scale: float | None = None,
                         negative_prompt: str = "",
                         output_path: str = "",
                         ensure_downloaded: bool = True) -> list:
    """Generate an image from a text prompt using a local diffusion model.

    `model` is required - diffusion weights are gigabytes, so there is no
    default. Try "stabilityai/sd-turbo" (fast, 1-4 steps) or search for
    text-to-image models.

    `size` is "WIDTHxHEIGHT" (e.g. "512x512"); omit it to use the model's
    native resolution. `ensure_downloaded` pre-fetches the weights through
    InferML's own downloader, which is required on Windows: diffusers builds
    its download patterns with os.path.join, so loading an uncached pipeline
    there silently skips every component config.json and fails.
    """
    if ensure_downloaded:
        await _client.download(model)

    payload: dict[str, Any] = {"prompt": prompt, "model": model}
    if size:
        payload["size"] = size
    if steps is not None:
        payload["steps"] = steps
    if guidance_scale is not None:
        payload["guidance_scale"] = guidance_scale
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    body = await _client.post("/v1/image/generation", payload)
    raw = base64.b64decode(body["data"][0]["b64_json"])
    stem = "".join(ch if ch.isalnum() else "-" for ch in prompt.lower())[:40].strip("-")
    dest = _out_path(f"{stem or 'image'}.png", output_path)
    dest.write_bytes(raw)
    return [f"Generated with {body.get('model')}. Saved to {dest}",
            Image(data=raw, format="png")]


# --- audio -------------------------------------------------------------------

@mcp.tool()
async def transcribe_audio(audio_path: str, model: str = "",
                           timestamps: bool = False) -> dict:
    """Transcribe speech from an audio file to text.

    Defaults to Whisper. Set `timestamps` to get a segment-by-segment
    transcript instead of one block of text.
    """
    raw, mime = _read_media(audio_path, "audio")
    data = {"response_format": "verbose_json" if timestamps else "json"}
    if model:
        data["model"] = model
    body = await _client.post_file("/v1/audio/transcriptions",
                                   filename=Path(audio_path).name,
                                   content=raw, mime=mime, data=data)
    out = {"text": body.get("text", "")}
    if timestamps:
        out["duration_seconds"] = body.get("duration")
        out["segments"] = body.get("segments") or []
    return out


@mcp.tool()
async def text_to_speech(text: str, voice: str = "", model: str = "",
                         output_path: str = "") -> dict:
    """Synthesize speech from text and save it as a WAV file.

    Returns the file path - audio is not inlined, since an LLM can't listen to
    it. `voice` accepts OpenAI's names (alloy, echo, fable, onyx, nova,
    shimmer) and only affects SpeechT5; other TTS models have a fixed voice.
    Output is always WAV.
    """
    payload: dict[str, Any] = {"input": text}
    if voice:
        payload["voice"] = voice
    if model:
        payload["model"] = model
    raw = await _client.post("/v1/audio/speech", payload)
    if not isinstance(raw, (bytes, bytearray)):
        raise InferMLError(f"Expected WAV bytes, got {type(raw).__name__}")
    stem = "".join(ch if ch.isalnum() else "-" for ch in text.lower())[:40].strip("-")
    dest = _out_path(f"{stem or 'speech'}.wav", output_path)
    dest.write_bytes(raw)
    return {"path": str(dest), "bytes": len(raw),
            "duration_seconds": _wav_duration(bytes(raw))}


# --- text --------------------------------------------------------------------

@mcp.tool()
async def generate_text(prompt: str, model: str = "", system: str = "",
                        max_tokens: int = 512,
                        temperature: float | None = None) -> dict:
    """Run a prompt through a local LLM loaded in InferML.

    With no `model`, uses whichever LLM is currently loaded. Useful for running
    a small on-device model, comparing its output against your own, or keeping
    a prompt entirely local.
    """
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    payload: dict[str, Any] = {"messages": messages, "max_tokens": max_tokens}
    if model:
        payload["model"] = model
    if temperature is not None:
        payload["temperature"] = temperature
    body = await _client.post("/v1/chat/completions", payload)
    choice = body["choices"][0]
    return {
        "model": body.get("model"),
        "text": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage"),
    }


@mcp.tool()
async def embed_text(texts: list[str], model: str = "") -> dict:
    """Embed text locally and report the vectors' shape and similarity.

    The raw vectors are deliberately not returned - hundreds of floats per
    string are unreadable and would flood the context. For two or more inputs
    this returns the pairwise cosine similarity matrix, which is what the
    vectors are usually for.
    """
    if not texts:
        raise InferMLError("`texts` must contain at least one string.")
    payload: dict[str, Any] = {"input": texts}
    if model:
        payload["model"] = model
    body = await _client.post("/v1/embeddings", payload)
    vectors = [d["embedding"] for d in body["data"]]
    out: dict[str, Any] = {
        "model": body.get("model"),
        "count": len(vectors),
        "dimensions": len(vectors[0]) if vectors else 0,
        "usage": body.get("usage"),
    }
    if len(vectors) > 1:
        # Embeddings come back L2-normalized, so the dot product is the cosine.
        out["cosine_similarity"] = [
            [round(sum(a * b for a, b in zip(u, v)), 4) for v in vectors]
            for u in vectors
        ]
    return out


def main() -> None:
    global _client, _output_dir
    p = argparse.ArgumentParser(
        prog="inferml-mcp",
        description="Expose a running InferML server to LLM clients over MCP (stdio).",
    )
    p.add_argument("--url", default=os.environ.get("INFERML_URL", DEFAULT_URL),
                   help=f"InferML server base URL (default {DEFAULT_URL}).")
    p.add_argument("--output-dir",
                   default=os.environ.get("INFERML_MCP_OUTPUT_DIR", str(_output_dir)),
                   help="Where generated images and audio are written.")
    args = p.parse_args()

    _client = InferMLClient(args.url)
    _output_dir = Path(args.output_dir).expanduser()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
