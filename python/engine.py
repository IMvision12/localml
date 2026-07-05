"""InferML inference engine - the reusable core.

Holds all model-loading and inference logic, driven in-process by the FastAPI
web server (`server/`). Running in-process is what lets the OpenAI-compatible
endpoint hold a live handle to the currently-loaded LLM and stream tokens from
it.

Design invariants:
  - One model = one loaded pipeline. Adapter instances are cached by
    (adapter_class_name, model_id); a second request reuses the loaded model.
  - Inference is NOT thread-safe against itself (torch). Callers serialize.
    The server runs `run()`/`download()` in a threadpool behind a single lock.
"""
from __future__ import annotations

import fnmatch
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import _win_compat  # noqa: F401, E402

from routing import inspect_model, pick_adapter, override_for  # noqa: E402
from io_utils import resolve_device  # noqa: E402

class DownloadCancelled(Exception):
    """Raised from the custom tqdm when the user dismisses a download."""

class Engine:
    """Holds the adapter cache and drives load / run / download / unload."""

    def __init__(self):
        self._adapter_cache: dict = {}
        self._current_llm_id: str | None = None
        self._log = _default_log

    def _get_adapter(self, info: dict):
        model_id = info["model_id"]
        adapter = pick_adapter(info)
        cache_key = (type(adapter).__name__, model_id)
        cached = self._adapter_cache.get(cache_key)
        if cached is not None:
            return cached
        self._log(f"loading {type(adapter).__name__} for {model_id}")
        dev = resolve_device()
        adapter.load(info, dev)
        self._adapter_cache[cache_key] = adapter
        return adapter

    def _resolve_info(self, model_id: str, task: str | None) -> dict:
        info = inspect_model(model_id)
        if not info.get("pipeline_tag") and task:
            info["pipeline_tag"] = task
        return info

    def run(self, model_id: str, task: str | None, inputs: dict, params: dict | None) -> dict:
        """Execute one inference. Returns an `output_kinds` dict.

        Overrides merge under request params, the model is inspected + routed,
        the adapter is loaded (or reused) and invoked. Records the model as the
        current LLM when it's a text generator so the OpenAI endpoint can find
        it.
        """
        if not model_id:
            raise ValueError("Missing 'modelId' - the session isn't bound to a model")

        inputs = inputs or {}

        override = override_for(model_id) or {}
        ovr_params = override.get("params") or {}
        req_params = params or {}
        merged_params = {**ovr_params, **req_params}

        info = self._resolve_info(model_id, task)
        adapter = self._get_adapter(info)

        out = adapter.run(inputs, merged_params)

        if _is_text_generation(info, task, out):
            self._current_llm_id = model_id
        return out

    def current_llm_id(self) -> str | None:
        return self._current_llm_id

    def loaded_model_ids(self) -> list[str]:
        seen = []
        for (_cls, model_id) in self._adapter_cache.keys():
            if model_id not in seen:
                seen.append(model_id)
        return seen

    def get_cached_adapter(self, model_id: str):
        """Return a loaded adapter instance for `model_id`, or None."""
        for (_cls, mid), adapter in self._adapter_cache.items():
            if mid == model_id:
                return adapter
        return None

    def ensure_loaded(self, model_id: str, task: str | None = None):
        """Load a model without running inference and return its adapter.

        Used by the OpenAI endpoint to lazy-load a model named in the request
        body when it isn't resident yet.
        """
        cached = self.get_cached_adapter(model_id)
        if cached is not None:
            return cached
        info = self._resolve_info(model_id, task)
        return self._get_adapter(info)

    def unload(self, model_id: str | None = None) -> int:
        """Drop cached adapter(s), freeing references (GPU memory). Returns the
        number of adapters unloaded. `None` unloads everything."""
        keys = [k for k in self._adapter_cache
                if model_id is None or k[1] == model_id]
        for k in keys:
            adapter = self._adapter_cache.pop(k, None)
            if adapter is not None:
                try:
                    adapter.unload()
                except Exception:
                    pass
            if self._current_llm_id == k[1]:
                self._current_llm_id = None
        _empty_torch_cache()
        return len(keys)

    def download(self, model_id: str, on_progress=None, cancel_event: "threading.Event | None" = None) -> dict:
        """Run `snapshot_download`, streaming byte-level progress via the
        `on_progress(dict)` callback. Picks exactly one weight format so multi-
        format repos don't download 4× the bytes. Raises DownloadCancelled if
        `cancel_event` is set mid-flight.

        Progress is delivered to the `on_progress` callback.
        """
        from huggingface_hub import snapshot_download, HfApi
        from tqdm.auto import tqdm as _BaseTqdm

        if not model_id:
            raise ValueError("Missing 'modelId'")

        on_progress = on_progress or (lambda _evt: None)

        WEIGHT_FORMAT_ORDER = ["safetensors", "bin", "pt", "ckpt", "msgpack", "h5", "onnx", "ot"]
        WEIGHT_EXT_RX = re.compile(r"\.(safetensors|bin|pt|ckpt|msgpack|h5|onnx|ot)$", re.IGNORECASE)

        siblings = []
        try:
            info = HfApi().model_info(model_id, files_metadata=True)
            siblings = list(info.siblings or [])
        except Exception:
            siblings = []

        chosen_ext = None
        by_ext: dict = {}
        for s in siblings:
            m = WEIGHT_EXT_RX.search((getattr(s, "rfilename", "") or "").lower())
            if not m:
                continue
            by_ext.setdefault(m.group(1), []).append(s)
        for ext in WEIGHT_FORMAT_ORDER:
            if by_ext.get(ext):
                chosen_ext = ext
                break

        ignore_patterns: list = []
        if chosen_ext:
            for ext in WEIGHT_FORMAT_ORDER:
                if ext != chosen_ext and by_ext.get(ext):
                    ignore_patterns.append(f"*.{ext}")
            if chosen_ext != "onnx":
                ignore_patterns.append("onnx/*")

        total_bytes = 0
        if siblings:
            for s in siblings:
                name = (getattr(s, "rfilename", "") or "").lower()
                if any(fnmatch.fnmatch(name, p) for p in ignore_patterns):
                    continue
                sz = getattr(s, "size", None) or 0
                if sz:
                    total_bytes += sz

        state = {"done": 0, "last_emit": 0.0}
        emit_lock = threading.Lock()

        def emit(final: bool = False) -> None:
            with emit_lock:
                done = state["done"]
                pct = (done / total_bytes * 100.0) if total_bytes else 0.0
                on_progress({
                    "done": int(done),
                    "total": int(total_bytes),
                    "pct": round(pct, 2),
                    "final": bool(final),
                })

        class ProgressTqdm(_BaseTqdm):
            def __init__(self, *args, **kwargs):
                self._is_bytes = kwargs.get("unit") == "B"
                kwargs["disable"] = True
                super().__init__(*args, **kwargs)

            def update(self, n=1):
                if cancel_event is not None and cancel_event.is_set():
                    raise DownloadCancelled()
                super().update(n)
                if self._is_bytes and n:
                    should_emit = False
                    with emit_lock:
                        state["done"] += n
                        now = time.time()
                        if (now - state["last_emit"]) >= 0.15:
                            state["last_emit"] = now
                            should_emit = True
                    if should_emit:
                        emit()

        emit()
        try:
            kwargs: dict = {"repo_id": model_id, "tqdm_class": ProgressTqdm}
            if ignore_patterns:
                kwargs["ignore_patterns"] = ignore_patterns
            path = snapshot_download(**kwargs)
        finally:
            emit(final=True)
        return {"path": path, "bytes": state["done"], "total_bytes": total_bytes}

def _default_log(msg: str) -> None:
    print(f"[engine] {msg}", file=sys.stderr, flush=True)

def _is_text_generation(info: dict, task: str | None, out: dict) -> bool:
    """Best-effort: did this run produce LLM text from a causal model?"""
    pt = (info.get("pipeline_tag") or task or "").lower()
    if pt in ("text-generation", "conversational"):
        return (out or {}).get("kind") == "text"
    return False

def _empty_torch_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def actionable_error(e: Exception) -> str:
    """Translate raw tracebacks into messages a user can act on.

    Maps common failure modes (OOM, gated repos, missing deps, ...) to guidance
    the UI and API can surface directly.
    """
    msg = str(e)
    lower = msg.lower()
    if "out of memory" in lower or "cuda out of memory" in lower:
        return "Out of memory - try a smaller model or switch to CPU (disable CUDA) in settings."
    if "cve-2025-32434" in lower or ("torch" in lower and "v2.6" in msg):
        return ("Your torch version is too old - transformers requires torch ≥ 2.6 to load this model's weights. "
                "Reinstall InferML's inference extra with a torch ≥ 2.6 wheel.")
    if "not a valid" in lower and "trust_remote_code" in lower:
        return ("This model requires `trust_remote_code=True`. Add an entry for it in "
                "python/model_overrides.json: { \"trust_remote_code\": true }.")
    is_gated = (
        "gatedrepoerror" in lower
        or "gated repo" in lower
        or "access to model" in lower and "restricted" in lower
        or "401" in msg and ("huggingface" in lower or "unauthorized" in lower)
        or "403" in msg and ("huggingface" in lower or "forbidden" in lower)
        or "must be authenticated" in lower
        or "you need to be logged in" in lower
    )
    if is_gated:
        return ("This model is gated or private - it requires a Hugging Face access token. "
                "Open Settings → HF Token, paste a token from "
                "https://huggingface.co/settings/tokens (Read access is enough), then retry.")
    if "no module named" in lower:
        mod = msg.split("'")[1] if "'" in msg else "unknown"
        return f"Missing Python package: `{mod}`. Install it into the InferML environment and retry."
    m = re.search(r"requires the (\S+) library", msg)
    if m:
        mod = m.group(1).strip("`'\".,")
        return f"Missing Python package: `{mod}`. Install it into the InferML environment and retry."
    if "could not load model" in lower or "not a recognized model" in lower:
        return (f"{msg}\n\nThis model doesn't fit any registered family. Add a folder "
                "under python/models/ for it, pin it in python/model_overrides.json, "
                "or drop a plugin file in python/plugins/.")
    return msg

ENGINE = Engine()
