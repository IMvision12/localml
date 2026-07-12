"""The inference engine, driven over stdin/stdout.

InferML is an Electron app with a Python engine. This is the seam between them:
Electron spawns `python -u runner.py` as a child process and talks to it in
newline-delimited JSON. There is no HTTP server, no port, and nothing listening
on the network - the only way to reach the engine is to be its parent process.

Protocol
--------
In  (one JSON object per line on stdin):
    {"id": "7", "type": "hf.search", "q": "detr"}

Out (one JSON object per line on stdout), distinguished by which key is present:
    {"id": "7", "ok": true,  "result": {...}}   terminal - request finished
    {"id": "7", "ok": false, "error": "..."}    terminal - request failed
    {"id": "7", "progress": {...}}              streaming - more to come
    {"event": "hw:update", "data": {...}}       unsolicited broadcast

Every request gets exactly one terminal frame, so the caller can resolve a
promise per id and never leak one. Long operations (download, setup) emit any
number of `progress` frames first.

Why stdout is stolen
--------------------
torch, transformers and diffusers all print to stdout - version warnings,
progress bars, "Some weights were not initialized...". stdout is now the wire,
so a single stray print corrupts a JSON frame and desynchronises the protocol.
The real stdout is therefore captured here, before any of that is imported, and
`sys.stdout` is repointed at stderr. Anything that prints lands in the Electron
log where it belongs; only `_send` can reach the wire.

Concurrency
-----------
Requests run on a thread pool, so a slow HF search can't stall a hardware tick.
torch is not re-entrant, so inference serialises behind `_INFERENCE_LOCK`;
everything else is free to overlap. Writes to the wire are serialised by
`_WRITE_LOCK` - two threads finishing at once must not interleave half a line.
"""
from __future__ import annotations

import sys

# Before anything heavy is imported. See "Why stdout is stolen" above.
_WIRE = sys.stdout
sys.stdout = sys.stderr

import json
import os
import platform
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import runtime  # noqa: E402
from engine import ENGINE, DownloadCancelled, actionable_error  # noqa: E402
from services import hf_service as hf  # noqa: E402
from services import hw_service  # noqa: E402
from services import store_service as store  # noqa: E402
from services.appdata import installs_file, write_json  # noqa: E402
from services.events import HUB  # noqa: E402

# --- the wire ----------------------------------------------------------------

_WRITE_LOCK = threading.Lock()


def _send(frame: dict) -> None:
    line = json.dumps(frame, ensure_ascii=False, default=str)
    with _WRITE_LOCK:
        try:
            _WIRE.write(line + "\n")
            _WIRE.flush()
        except (BrokenPipeError, ValueError):
            # Electron is gone. There is nobody left to talk to and nothing to
            # clean up that the OS won't do for us.
            os._exit(0)


def _ok(rid, result=None):
    _send({"id": rid, "ok": True, "result": result})


def _err(rid, message):
    _send({"id": rid, "ok": False, "error": str(message)})


def _progress(rid, data: dict):
    _send({"id": rid, "progress": data})


HUB.set_sink(lambda event, data: _send({"event": event, "data": data}))

# --- shared state ------------------------------------------------------------

# The lock lives in runtime.py, NOT here. The API front-end has to take the same
# one - see the module docstring there for why two locks would be a silent
# correctness bug rather than a performance one.
_INFERENCE_LOCK = runtime.INFERENCE_LOCK_RAW
_STOP = runtime._STOP
_ACTIVE_DOWNLOADS: dict[str, threading.Event] = {}
_SETUP_RUNNING = threading.Event()

_INFERENCE_PKGS = [
    "transformers>=5.7.0", "diffusers", "accelerate", "timm", "pillow",
    "soundfile", "librosa", "numpy", "scipy", "sentencepiece", "protobuf",
    "huggingface_hub",
]


def _torch_index(accelerator: str) -> tuple[str, str | None]:
    """Which PyTorch wheel index to install from.

    macOS ships MPS support in the default PyPI wheel. On Windows/Linux the
    default wheel is the CUDA build, so a CPU install must be explicit or the
    user downloads gigabytes of CUDA they cannot use.
    """
    if platform.system() == "Darwin":
        return ("Apple Silicon / MPS", None)
    if accelerator == "gpu":
        return ("CUDA 12.4", "https://download.pytorch.org/whl/cu124")
    return ("CPU", "https://download.pytorch.org/whl/cpu")


def _pip_phases(accelerator: str):
    torch_pkgs = ["torch>=2.6", "torchvision", "torchaudio>=2.6"]
    pip = [sys.executable, "-m", "pip", "install"]
    label, index = _torch_index(accelerator)
    torch_cmd = pip + (["--index-url", index] if index else []) + torch_pkgs
    return [
        (f"Installing PyTorch ({label})", torch_cmd),
        ("Installing transformers, diffusers and supporting libraries", pip + _INFERENCE_PKGS),
    ]


# --- storage -----------------------------------------------------------------

_SIZE_CACHE: dict[str, tuple] = {}
_SIZE_TTL = 20.0


def _dir_size(paths) -> tuple[int, int]:
    total = files = 0
    for root in paths:
        p = Path(root)
        if not p.exists():
            continue
        for dirpath, _dirnames, filenames in os.walk(p):
            for fn in filenames:
                try:
                    total += os.stat(os.path.join(dirpath, fn)).st_size
                    files += 1
                except OSError:
                    pass
    return total, files


def _storage_size(key: str) -> dict:
    cached = _SIZE_CACHE.get(key)
    if cached and cached[1] > time.time():
        return cached[0]
    if key in ("hfCache", "hf"):
        root = hf.hf_cache_dir()
        b, f = _dir_size([root])
        out = {"ok": True, "bytes": b, "files": f, "paths": [str(root)]}
    elif key in ("pyRuntime", "py"):
        b, f = _dir_size([sys.prefix])
        out = {"ok": True, "bytes": b, "files": f, "paths": [sys.prefix]}
    else:
        return {"ok": False, "error": f"unknown storage key {key!r}"}
    _SIZE_CACHE[key] = (out, time.time() + _SIZE_TTL)
    return out


def _clear_hf_cache() -> dict:
    removed, freed, errors = 0, 0, []
    for root in hf._cache_roots():
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.name.startswith("models--"):
                try:
                    b, _ = _dir_size([child])
                    shutil.rmtree(child)
                    removed += 1
                    freed += b
                except Exception as e:
                    errors.append(str(e))
    try:
        write_json(installs_file(), {})
    except Exception:
        pass
    _SIZE_CACHE.pop("hfCache", None)
    HUB.publish("hf:installsChanged")
    return {"ok": True, "removed": removed, "bytes": freed, "errors": errors}


# --- long-running operations -------------------------------------------------

def _op_run(rid, p):
    _STOP.clear()
    with _INFERENCE_LOCK:
        try:
            output = ENGINE.run(p.get("modelId"), p.get("task"), p.get("input") or {}, p.get("params") or {})
            return {"ok": True, "output": output}
        except Exception as e:
            if _STOP.is_set():
                return {"ok": False, "error": "Stopped by user", "cancelled": True}
            print("[run] " + traceback.format_exc())
            return {"ok": False, "error": actionable_error(e)}


def _op_download(rid, p):
    model_id = p.get("modelId")
    if not model_id:
        return {"ok": False, "error": "modelId required"}

    cancel = threading.Event()
    _ACTIVE_DOWNLOADS[model_id] = cancel
    try:
        info = ENGINE.download(
            model_id,
            on_progress=lambda evt: _progress(rid, {"modelId": model_id, **evt}),
            cancel_event=cancel,
        )
        return {"ok": True, "info": info}
    except DownloadCancelled:
        return {"ok": False, "cancelled": True, "error": "cancelled"}
    except Exception as e:
        return {"ok": False, "error": actionable_error(e)}
    finally:
        if _ACTIVE_DOWNLOADS.get(model_id) is cancel:
            _ACTIVE_DOWNLOADS.pop(model_id, None)


def _op_setup(rid, p):
    """Install the inference stack (torch et al.) for the chosen accelerator.

    pip runs as a subprocess and its output is streamed line by line, because
    this is a multi-gigabyte download and a silent progress bar for ten minutes
    reads as a hang.
    """
    import subprocess

    if _SETUP_RUNNING.is_set():
        return {"ok": False, "error": "A setup is already running."}
    _SETUP_RUNNING.set()

    accel = (p or {}).get("accelerator") or "cpu"
    try:
        for step_label, cmd in _pip_phases(accel):
            _progress(rid, {"kind": "step", "text": step_label})
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    _progress(rid, {"kind": "log", "text": line})
            proc.wait()
            if proc.returncode != 0:
                return {"ok": False, "error": f"pip exited {proc.returncode} during: {step_label}"}
        _progress(rid, {"kind": "step", "text": "Runtime ready"})
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        _SETUP_RUNNING.clear()


def _op_stop(rid, p):
    _STOP.set()
    return {"ok": True}


def _op_cancel_download(rid, p):
    ev = _ACTIVE_DOWNLOADS.get(p.get("modelId"))
    if ev is None:
        return {"ok": True, "cancelled": False, "reason": "no active download"}
    ev.set()
    return {"ok": True, "cancelled": True}


def _op_uninstall(rid, p):
    mid = p.get("id")
    if not hf.is_valid_model_id(mid):
        return {"ok": False, "error": "Invalid model id"}
    try:
        ENGINE.unload(mid)
    except Exception:
        pass
    store.uninstall(mid)
    cache = hf.delete_model_cache(mid)
    return {"ok": True, "removed": cache.get("removed", []), "errors": cache.get("errors", [])}


def _op_mark_installed(rid, p):
    mid = p.get("id")
    if not hf.is_valid_model_id(mid):
        return {"ok": False, "error": "Invalid model id"}
    return {"ok": store.mark_installed(mid, p.get("meta"))}


# --- the local HTTP API ------------------------------------------------------
#
# Optional, and off unless the user turns it on. `api/server.py` explains the
# posture; the persisted decision lives in settings.json so it survives restarts.

def _api():
    from api.server import SERVER, DEFAULT_PORT
    return SERVER, DEFAULT_PORT


def _api_port() -> int:
    return int((store.get_settings() or {}).get("apiPort") or 11500)


def _op_api_start(rid, p):
    SERVER, default_port = _api()
    port = int(p.get("port") or _api_port() or default_port)
    res = SERVER.start(port)
    if res.get("running"):
        store.save_settings({"apiEnabled": True, "apiPort": port})
    return res


def _op_api_stop(rid, p):
    SERVER, _ = _api()
    res = SERVER.stop()
    store.save_settings({"apiEnabled": False})
    return res


def _op_api_status(rid, p):
    SERVER, _ = _api()
    st = SERVER.status()
    st["enabled"] = bool((store.get_settings() or {}).get("apiEnabled"))
    return st


def _autostart_api() -> None:
    """Bring the API up at boot if the user left it on.

    Deliberately not fatal: a port already taken (a second InferML, or anything
    else on 11500) must not stop the app from starting. The failure is reported
    through api.status instead, where Settings can show it.
    """
    settings = store.get_settings() or {}
    if not settings.get("apiEnabled"):
        return
    try:
        SERVER, default_port = _api()
        SERVER.start(int(settings.get("apiPort") or default_port))
    except Exception as e:
        print(f"[runner] API autostart failed: {e}")


# --- dispatch ----------------------------------------------------------------

# Every operation the UI can invoke. Handlers take (request_id, payload) and
# return the result; raising is fine - `_dispatch` turns it into an error frame.
#
# This table and src/main/ipc.js's ALLOWED set are two halves of one contract:
# an op that isn't in both is unreachable.
OPS = {
    "ping":               lambda rid, p: {"ok": True},

    "tasks.run":          _op_run,
    "tasks.stop":         _op_stop,
    "tasks.status":       lambda rid, p: runtime.probe_status(),
    "tasks.setup":        _op_setup,
    "tasks.download":     _op_download,
    "tasks.cancelDownload": _op_cancel_download,

    "api.status":         _op_api_status,
    "api.start":          _op_api_start,
    "api.stop":           _op_api_stop,

    "hf.search":          lambda rid, p: hf.search(p.get("q"), p.get("task")),
    "hf.installed":       lambda rid, p: store.list_installed(),
    "hf.markInstalled":   _op_mark_installed,
    "hf.uninstall":       _op_uninstall,
    "hf.modelInfo":       lambda rid, p: hf.model_info(p.get("id")),
    "hf.getToken":        lambda rid, p: {"token": hf.get_masked_token()},
    "hf.setToken":        lambda rid, p: hf.set_token(p.get("token") or ""),
    "hf.clearToken":      lambda rid, p: hf.clear_token(),
    "hf.verifyToken":     lambda rid, p: hf.verify_token(p.get("token") or ""),

    "chats.list":         lambda rid, p: store.list_chats(),
    "chats.get":          lambda rid, p: store.get_chat(p.get("id")),
    "chats.save":         lambda rid, p: {"ok": store.save_chat(p.get("chat") or {})},
    "chats.patch":        lambda rid, p: {"ok": store.patch_chat(p.get("id"), p.get("patch") or {})},
    "chats.delete":       lambda rid, p: {"ok": store.delete_chat(p.get("id"))},

    "settings.get":       lambda rid, p: store.get_settings(),
    "settings.save":      lambda rid, p: store.save_settings(p.get("patch") or {}),

    "hw.get":             lambda rid, p: hw_service.sample_hw(),

    "storage.size":       lambda rid, p: _storage_size(p.get("key") or ""),
    "storage.clear":      lambda rid, p: (
        _clear_hf_cache() if (p.get("key") in ("hfCache", "hf"))
        else {"ok": False, "error": "Only the models cache can be cleared here."}
    ),
}


def _dispatch(msg: dict) -> None:
    rid = msg.get("id")
    op = OPS.get(msg.get("type"))
    if op is None:
        _err(rid, f"unknown op {msg.get('type')!r}")
        return
    try:
        _ok(rid, op(rid, msg))
    except Exception as e:
        print(f"[runner] {msg.get('type')} failed:\n{traceback.format_exc()}")
        _err(rid, actionable_error(e))


def _hw_poller(interval: float = 2.5) -> None:
    while True:
        try:
            data = hw_service.sample_hw()
            if not data.get("error"):
                HUB.publish("hw:update", data)
        except Exception:
            pass
        time.sleep(interval)


def main() -> None:
    threading.Thread(target=_hw_poller, name="hw-poller", daemon=True).start()
    _autostart_api()

    # Downloads and setup can each occupy a worker for minutes at a time while
    # the UI keeps polling status, so the pool has room to spare.
    pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="op")

    _send({"event": "ready", "data": {"pid": os.getpid()}})

    # EOF on stdin means Electron closed the pipe - it has quit, or crashed.
    # Either way this process has no reason to exist, and lingering would hold
    # GPU memory with no UI left to free it.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        pool.submit(_dispatch, msg)


if __name__ == "__main__":
    main()
