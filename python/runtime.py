"""Process-wide engine state, shared by both front-ends.

InferML has two ways in, and they run on different concurrency models:

  - the **app**: `runner.py` reads JSON off stdin and dispatches onto a thread
    pool. Synchronous, `threading`.
  - the **API**: `api/` is FastAPI under uvicorn, in a thread with its own event
    loop. Asynchronous, `asyncio`.

Both drive the *same* `ENGINE`, and torch is not re-entrant - two inferences at
once corrupts state or crashes. So the lock that serialises them cannot live in
either front-end: it lives here, and it is a single `threading.Lock`, because
that is the only kind both worlds can honour.

`INFERENCE_LOCK` is an async *view* onto that one lock. `async with
INFERENCE_LOCK` acquires it on a worker thread, so an API request waits for a UI
inference (and vice versa) without ever blocking uvicorn's event loop. Getting
this wrong is subtle and nasty: two independent locks would look completely
correct, pass every test that exercises one path at a time, and then produce
garbage the first time someone hit /v1 while the app window was running a model.
"""
from __future__ import annotations

import asyncio
import os
import platform
import shutil
import sys
import threading
from pathlib import Path

from engine import ENGINE

# The one lock. `runner.py` takes it directly (`with INFERENCE_LOCK_RAW`);
# the API takes it through the async view below.
INFERENCE_LOCK_RAW = threading.Lock()


class _AsyncLockView:
    """`async with` over a threading.Lock, without blocking the event loop."""

    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock

    async def __aenter__(self):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._lock.acquire)
        return self

    async def __aexit__(self, *_exc):
        self._lock.release()
        return False


INFERENCE_LOCK = _AsyncLockView(INFERENCE_LOCK_RAW)


def engine():
    return ENGINE


# --- stop flag ---------------------------------------------------------------

_STOP = threading.Event()


def request_stop() -> None:
    _STOP.set()


def clear_stop() -> None:
    _STOP.clear()


def stop_requested() -> bool:
    return _STOP.is_set()


async def run_blocking(fn, *args, **kwargs):
    """Run a blocking engine call in the default threadpool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


# --- runtime probing ---------------------------------------------------------

# The inference stack is installed on demand (~2GB of CUDA wheels), so we must be
# able to answer "is it here yet?" without importing torch - which would cost
# seconds and defeat the point.
FULL_STACK = [
    "torch", "transformers", "PIL", "numpy", "huggingface_hub",
    "soundfile", "librosa", "accelerate", "timm", "diffusers",
    "sentencepiece", "scipy",
]


def probe_status() -> dict:
    import importlib.metadata
    import importlib.util

    missing = [m for m in FULL_STACK if importlib.util.find_spec(m) is None]
    ready = not missing

    is_mac = platform.system() == "Darwin"
    has_nvidia = shutil.which("nvidia-smi") is not None
    suggested = "gpu" if (is_mac or has_nvidia) else "cpu"

    try:
        torch_version = importlib.metadata.version("torch")
    except Exception:
        torch_version = None

    try:
        from services.hf_service import hf_cache_dir
        hf_cache = str(hf_cache_dir())
    except Exception:
        hf_cache = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")

    return {
        "ready": ready,
        "runtimeInstalled": ready,
        "missing": missing,
        "activeAccelerator": suggested,
        "installedAccelerator": suggested,
        "accelerators": {},
        "hasNvidia": has_nvidia,
        "suggestedAccelerator": suggested,
        "torch": torch_version,
        "platform": platform.system().lower(),
        "arch": platform.machine(),
        "runtimePath": sys.prefix,
        "hfCachePath": hf_cache,
        "sidecarRunning": True,
    }
