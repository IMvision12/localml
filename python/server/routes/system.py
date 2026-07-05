"""System routes: hardware, logs, app info, and the SSE broadcast stream.

  GET /api/hw       ← hw:get
  GET /api/events   ← replaces Electron webContents broadcasts
                      (hw:update, chats:updated, hf:installsChanged)
  GET /api/logs     ← logs:list
  GET /api/app      ← app:version / app:paths
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

from fastapi import APIRouter, Body, Query, Request
from fastapi.responses import StreamingResponse

from server import __version__, deps
from server.events import HUB, EventHub
from server import hw_service

router = APIRouter(prefix="/api")

@router.get("/hw")
async def hw():
    return await deps.run_blocking(hw_service.sample_hw)

@router.get("/logs")
async def logs():
    return deps.logs()

@router.get("/app")
async def app_info():
    from server.appdata import data_dir
    return {"name": "inferml", "version": __version__, "dataDir": str(data_dir())}

_SIZE_CACHE: dict[str, tuple] = {}
_SIZE_TTL = 20.0

def _hf_cache_root() -> Path:
    from server.hf_service import hf_cache_dir
    return hf_cache_dir()

def _dir_size(paths) -> tuple[int, int]:
    total = 0
    files = 0
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

def _compute_size(key: str) -> dict:
    cached = _SIZE_CACHE.get(key)
    if cached and cached[1] > time.time():
        return cached[0]
    if key in ("hfCache", "hf"):
        root = _hf_cache_root()
        b, f = _dir_size([root])
        out = {"ok": True, "bytes": b, "files": f, "paths": [str(root)]}
    elif key in ("pyRuntime", "py"):
        b, f = _dir_size([sys.prefix])
        out = {"ok": True, "bytes": b, "files": f, "paths": [sys.prefix]}
    else:
        return {"ok": False, "error": f"unknown storage key {key!r}"}
    _SIZE_CACHE[key] = (out, time.time() + _SIZE_TTL)
    return out

@router.get("/storage/size")
async def storage_size(key: str = Query(...)):
    return await deps.run_blocking(_compute_size, key)

@router.post("/storage/clear")
async def storage_clear(payload: dict = Body(default={})):
    key = (payload or {}).get("key")
    if key not in ("hfCache", "hf"):
        return {"ok": False, "error": "Only the models cache can be cleared here."}
    return await deps.run_blocking(_clear_hf_cache)

def _clear_hf_cache() -> dict:
    from server.hf_service import _cache_roots
    from server.appdata import write_json, installs_file
    removed, freed, errors = 0, 0, []
    for root in _cache_roots():
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

@router.get("/events")
async def events(request: Request):
    """Server-sent events: hardware ticks + store-change notifications. The
    bridge opens one of these and routes by event name."""
    q = HUB.subscribe()

    async def stream():
        try:
            yield ": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield EventHub.format_sse(payload)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            HUB.unsubscribe(q)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

async def hw_poller(interval: float = 2.5):
    """Single global poller: sample hardware and broadcast `hw:update` to every
    connected client. Started from the app lifespan."""
    while True:
        try:
            data = await deps.run_blocking(hw_service.sample_hw)
            if not data.get("error"):
                HUB.publish("hw:update", data)
        except Exception:
            pass
        await asyncio.sleep(interval)
