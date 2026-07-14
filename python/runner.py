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


def _installer() -> list[str]:
    """`uv pip install`, falling back to pip if uv somehow isn't there.

    uv is not just faster here - it is the difference between a progress display
    and a frozen one. pip renders its download bar with carriage returns and, on
    a pipe, only flushes it once the file has fully arrived: a 2.4GB CUDA wheel
    means several minutes of *zero* output, which reads as a hang. uv announces
    each package and its size up front ("Downloading torch (2.4GiB)") and reports
    each one as it lands.

    Note this invokes uv's *binary* directly, not `python -m uv`. That module is
    a shim which re-spawns uv.exe through subprocess - and since the venv's
    python.exe is itself a launcher stub that re-execs the real interpreter, the
    chain became runner -> stub -> python -> shim -> uv.exe. Output never made it
    back down four levels of inherited handles: the install ran, and the UI saw
    complete silence. uv tells us where its binary is; use it.
    """
    try:
        from uv import find_uv_bin
        return [str(find_uv_bin()), "pip", "install", "--python", sys.executable]
    except Exception:
        return [sys.executable, "-m", "pip", "install"]


def _install_phases(accelerator: str):
    torch_pkgs = ["torch>=2.6", "torchvision", "torchaudio>=2.6"]
    install = _installer()
    label, index = _torch_index(accelerator)
    torch_cmd = install + (["--index-url", index] if index else []) + torch_pkgs
    return [
        (f"Installing PyTorch ({label})", torch_cmd),
        ("Installing transformers, diffusers and supporting libraries", install + _INFERENCE_PKGS),
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
        # The venv *and* the wheel cache that fills it. Reporting only the venv
        # understated the runtime's real footprint enormously - 140MB of venv can
        # sit on top of 10GB of cached wheels - and a Storage screen that hides ten
        # gigabytes is worse than no Storage screen.
        roots = [sys.prefix]
        cache = os.environ.get("UV_CACHE_DIR")
        if cache and os.path.isdir(cache):
            roots.append(cache)
        b, f = _dir_size(roots)
        out = {"ok": True, "bytes": b, "files": f, "paths": roots}
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


def _stream_process(cmd: list[str], emit) -> int:
    """Run `cmd`, calling emit(text) for every line it writes. Returns the exit code.

    Reads a character at a time and treats BOTH \\r and \\n as end-of-line.
    Iterating the pipe (`for line in proc.stdout`) instead - the obvious way, and
    what this used to do - blocks until a *newline* arrives, and installers draw
    their progress bars by rewriting one line with carriage returns. The bar only
    terminates in a newline once the download has finished, so a 2.4GB wheel
    produced several minutes of complete silence followed by one line saying it
    was done. The window looked frozen because, as far as it could tell, it was.
    """
    import subprocess

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=0, encoding="utf-8", errors="replace",
    )

    last = [time.time()]

    def say(text: str) -> None:
        """Report a line, but never at the cost of the drain.

        This is the whole ballgame. If emit() raises, the pump thread dies, nobody
        reads uv's stdout, uv blocks writing into a full pipe buffer, and it never
        exits - so the poll loop below waits forever on a process that is waiting on
        us. The install appears to freeze mid-log and stays frozen, permanently.
        Draining the pipe is the reader's real job; reporting is a courtesy, and a
        courtesy must never be able to deadlock the thing it is reporting on.
        """
        try:
            last[0] = time.time()
            emit(text)
        except Exception:
            pass

    def pump():
        buf = ""
        try:
            while True:
                ch = proc.stdout.read(1)
                if not ch:
                    break
                if ch in "\r\n":
                    text = buf.strip()
                    buf = ""
                    if text:
                        say(text)
                else:
                    buf += ch
            if buf.strip():
                say(buf.strip())
        except Exception:
            pass    # the pipe broke; the wait below will collect the exit code

    reader = threading.Thread(target=pump, name="setup-reader", daemon=True)
    reader.start()

    # Even uv goes quiet for the body of a multi-GB download - it says
    # "Downloading torch (2.4GiB)" and then nothing until the file lands. Without
    # a sign of life, a correct install is indistinguishable from a hung one, so
    # say something on the way.
    started = time.time()
    while True:
        try:
            proc.wait(timeout=1.0)
            break
        except subprocess.TimeoutExpired:
            pass

        idle = time.time() - last[0]
        if idle >= 15:
            mins, secs = divmod(int(time.time() - started), 60)
            say(f"… still working ({mins}m {secs:02d}s elapsed) - large downloads take a while")

        # The reader reaching EOF means the process closed its output: it is done
        # talking and should be about to exit. If it is somehow still alive a minute
        # later it is wedged, and waiting on it forever - which is what this loop
        # used to do - only turns a stuck install into a stuck application. Kill it
        # and let the caller report a real failure.
        if not reader.is_alive() and time.time() - last[0] > 60:
            say("The installer stopped responding. Giving up on it.")
            try:
                proc.kill()
            except Exception:
                pass
            proc.wait(timeout=10)
            break

    reader.join(timeout=5)
    try:
        proc.stdout.close()
    except Exception:
        pass
    return proc.returncode


def _op_setup(rid, p):
    """Install the inference stack (torch et al.) for the chosen accelerator."""
    if _SETUP_RUNNING.is_set():
        return {"ok": False, "error": "A setup is already running."}
    _SETUP_RUNNING.set()

    accel = (p or {}).get("accelerator") or "cpu"
    try:
        for step_label, cmd in _install_phases(accel):
            _progress(rid, {"kind": "step", "text": step_label})
            echo = " ".join([os.path.basename(cmd[0])] + cmd[1:])
            _progress(rid, {"kind": "log", "text": "$ " + echo})
            code = _stream_process(cmd, lambda text: _progress(rid, {"kind": "log", "text": text}))
            if code != 0:
                return {"ok": False, "error": f"Install failed (exit {code}) during: {step_label}"}
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
    """Route one request frame: {"id", "type", "payload"}.

    The payload is a nested object, not spread across the frame, and it has to
    stay that way. Flattened, any argument named `id` collides with the frame's
    own id - and it wins, because it is merged in last. The engine then answers
    the request with the *argument* as its id, Electron looks that up among the
    requests it is waiting on, finds nothing, and drops the reply. The caller's
    promise never settles: no error, no timeout, just a spinner forever. That is
    what "fetching…" was on every model card, and it silently broke six ops.
    """
    rid = msg.get("id")
    op = OPS.get(msg.get("type"))
    if op is None:
        _err(rid, f"unknown op {msg.get('type')!r}")
        return
    payload = msg.get("payload")
    if not isinstance(payload, dict):
        payload = {}
    try:
        _ok(rid, op(rid, payload))
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


# The OpenBLAS-backed extensions. Both numpy and scipy ship their own copy, and
# both deadlock if they are first imported while a read is pending on stdin.
_WARM = ("numpy", "scipy.linalg")


def _warm_native_libs() -> None:
    """Import numpy and scipy now, on this thread, before anything reads stdin.

    This is not an optimisation. Skip it and the first inference deadlocks, every
    single time, on Windows.

    Once the loop below is running there is always a blocking read pending on the
    stdin pipe - that read *is* the loop. Importing numpy from any other thread
    while it is outstanding never returns: it wedges inside the LoadLibrary of
    numpy's `_multiarray_umath` and stays there. The GIL is not the problem, since
    every other thread keeps running normally; it is the DLL load itself that
    hangs, and nothing recovers it. The request is never answered, so the window
    sits on "Loading…" and an MCP call for the same inference hangs with it - both
    arrive as ops, and ops run on pool threads.

    What the two culprits have in common is OpenBLAS, which each of them bundles
    and which spawns its worker threads from inside DLL initialisation. Extensions
    without it - pillow, psutil, tokenizers, safetensors - are entirely unaffected,
    and once these two are in sys.modules, torch, transformers, diffusers, timm,
    numba and librosa all import on worker threads with no trouble at all. torch is
    the reason this reaches everything: it imports numpy, so every model load
    inherited the deadlock.

    Warming them costs about half a second and a few tens of MB - far less than
    pre-importing torch, which is the alternative and which every user would pay
    for whether or not they ever ran a model.

    Guarded, because a fresh install has neither until the runtime is set up. That
    is fine: the setup op installs them, and the engine restarts afterwards.
    """
    for mod in _WARM:
        try:
            __import__(mod)
        except Exception:
            pass    # not installed yet, or genuinely broken - either way, not fatal here


def main() -> None:
    _warm_native_libs()

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
