"""Chats + settings + installs persistence.

Python port of Electron's `ipc/chats.js`, `ipc/settings.js`, and the installs
registry parts of `services/huggingface.js`. Same validation and merge rules.
"""
from __future__ import annotations

import time

from services.appdata import (
    chats_dir, chat_file, settings_file, installs_file,
    read_json, write_json,
)
from services.events import HUB

_PROTECTED_PATCH_FIELDS = {"id", "createdAt"}
_METADATA_FIELDS = ["pinned"]

def is_valid_chat_id(cid) -> bool:
    if not isinstance(cid, str):
        return False
    if len(cid) == 0 or len(cid) > 200:
        return False
    if "/" in cid or "\\" in cid:
        return False
    if ".." in cid or cid.startswith("."):
        return False
    return True

def _now_ms() -> int:
    return int(time.time() * 1000)

def list_chats() -> list[dict]:
    out = []
    try:
        for f in chats_dir().iterdir():
            if not f.name.endswith(".json"):
                continue
            c = read_json(f, None)
            if not c:
                continue
            out.append({
                "id": c.get("id"),
                "title": c.get("title"),
                "sub": c.get("sub"),
                "tag": c.get("tag"),
                "kind": c.get("kind") or "chat",
                "modelId": c.get("modelId") or c.get("model"),
                "model": c.get("model") or c.get("modelId"),
                "task": c.get("task"),
                "workspace": c.get("workspace"),
                "turns": len(c.get("messages") or []),
                "runs": len(c.get("runs") or []),
                "updatedAt": c.get("updatedAt") or 0,
                "createdAt": c.get("createdAt") or 0,
                "running": bool(c.get("running")),
                "pinned": bool(c.get("pinned")),
            })
    except Exception:
        return []
    out.sort(key=lambda a: (0 if a.get("pinned") else 1, -(a.get("updatedAt") or 0)))
    return out

def get_chat(cid: str):
    if not is_valid_chat_id(cid):
        return None
    return read_json(chat_file(cid), None)

def save_chat(chat: dict) -> bool:
    if not chat or not is_valid_chat_id(chat.get("id")):
        raise ValueError("chat.id required")
    existing = read_json(chat_file(chat["id"]), None) or {}
    merged = {**existing, **chat}
    for k in _METADATA_FIELDS:
        if k not in chat and k in existing:
            merged[k] = existing[k]
    merged["updatedAt"] = _now_ms()
    if not merged.get("createdAt"):
        merged["createdAt"] = merged["updatedAt"]
    write_json(chat_file(chat["id"]), merged)
    HUB.publish("chats:updated")
    return True

def patch_chat(cid: str, patch: dict) -> bool:
    if not is_valid_chat_id(cid):
        raise ValueError("invalid chat id")
    if not isinstance(patch, dict):
        raise ValueError("patch must be an object")
    chat = read_json(chat_file(cid), None)
    if not chat:
        return False
    for k, v in patch.items():
        if k in _PROTECTED_PATCH_FIELDS:
            continue
        chat[k] = v
    write_json(chat_file(cid), chat)
    HUB.publish("chats:updated")
    return True

def delete_chat(cid: str) -> bool:
    if not is_valid_chat_id(cid):
        return False
    try:
        chat_file(cid).unlink()
    except Exception:
        pass
    HUB.publish("chats:updated")
    return True

def get_settings() -> dict:
    return read_json(settings_file(), {}) or {}

def save_settings(patch: dict) -> dict:
    cur = get_settings()
    nxt = {**cur, **(patch or {})}
    write_json(settings_file(), nxt)
    return nxt

def list_installed() -> dict:
    return read_json(installs_file(), {}) or {}

def mark_installed(model_id: str, meta: dict | None = None) -> bool:
    cur = list_installed()
    cur[model_id] = {**(meta or {}), "installedAt": _now_ms()}
    write_json(installs_file(), cur)
    HUB.publish("hf:installsChanged")
    return True

def uninstall(model_id: str) -> dict:
    cur = list_installed()
    cur.pop(model_id, None)
    write_json(installs_file(), cur)
    HUB.publish("hf:installsChanged")
    return {"ok": True}
