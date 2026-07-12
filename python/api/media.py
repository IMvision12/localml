"""Shared plumbing for the media endpoints (/v1/audio/*, /v1/image/*).

These endpoints don't reimplement inference - they call the engine's existing
task handlers (ASR, TTS, object-detection, segmentation/SAM, diffusion) and
only do the HTTP-shaped parts here: resolving the `model` field to a real
Hugging Face id, coercing uploads/base64 into the `dataUrl` inputs the tasks
already consume, and unpacking `output_kinds` dicts into response bodies.
"""
from __future__ import annotations

import base64


class MediaError(Exception):
    """A bad request the route should turn into a 400."""


# OpenAI's own model names (plus InferML placeholders) accepted as "whatever
# fits this endpoint": resolve to an installed model for the task, else the
# endpoint's default. Any real HF id is used verbatim.
GENERIC_NAMES = {
    "", "inferml", "default", "current",
    # audio/transcriptions
    "whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe",
    # audio/speech
    "tts-1", "tts-1-hd", "gpt-4o-mini-tts",
    # images
    "dall-e-2", "dall-e-3", "gpt-image-1",
}


def resolve_media_model(requested: str | None, tasks: tuple, fallback: str | None) -> str:
    """Return the HF model id to run. Named models pass through; generic names
    prefer a model the user already installed for one of `tasks`, then
    `fallback`. Raises MediaError when there's nothing to run."""
    name = (requested or "").strip()
    if name and name.lower() not in GENERIC_NAMES:
        return name
    try:
        from services.store_service import list_installed
        for mid, meta in list_installed().items():
            if (meta or {}).get("task") in tasks:
                return mid
    except Exception:
        pass
    if fallback:
        return fallback
    raise MediaError(
        f"No model available. Pass a Hugging Face id as `model`, or install a "
        f"{tasks[0]} model in InferML first."
    )


def installed_task(model_id: str) -> str | None:
    """The task the user installed this model under, if any - used as the
    routing hint when the Hub metadata lookup can't supply a pipeline_tag."""
    try:
        from services.store_service import list_installed
        return (list_installed().get(model_id) or {}).get("task") or None
    except Exception:
        return None


def to_image_data_url(s) -> str:
    """Accept a data URL or bare base64 for the `image` field."""
    s = (s or "").strip() if isinstance(s, str) else ""
    if not s:
        raise MediaError("`image` is required (base64 or data URL).")
    if s.startswith("data:"):
        return s
    return "data:image/png;base64," + s


def bytes_to_data_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


def data_url_bytes(data_url: str) -> bytes:
    """Decode the payload of a data URL produced by output_kinds."""
    payload = data_url.split(",", 1)[1] if "," in data_url else data_url
    return base64.b64decode(payload)


def parse_size(size) -> dict:
    """OpenAI-style `size` ("1024x1024") -> width/height params for diffusion.
    Unparseable values are ignored (the model's native size is used)."""
    try:
        w, h = str(size or "").lower().split("x")
        w, h = int(w), int(h)
        if w > 0 and h > 0:
            return {"width": w, "height": h}
    except Exception:
        pass
    return {}
