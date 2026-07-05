"""Application data directory + JSON persistence.

Replaces Electron's `app.getPath('userData')` + `services/storage.js`. Uses
platformdirs so chats/settings/installs land in the OS-conventional per-user
location (e.g. %APPDATA%/InferML on Windows, ~/.local/share/InferML on Linux,
~/Library/Application Support/InferML on macOS).
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

try:
    from platformdirs import user_data_dir
except Exception:  # pragma: no cover - platformdirs is a declared dependency
    user_data_dir = None

def data_dir() -> Path:
    override = os.environ.get("INFERML_DATA_DIR")
    if override:
        base = Path(override)
    elif user_data_dir is not None:
        base = Path(user_data_dir("InferML", "InferML"))
    else:
        base = Path.home() / ".inferml"
    base.mkdir(parents=True, exist_ok=True)
    return base

def chats_dir() -> Path:
    d = data_dir() / "chats"
    d.mkdir(parents=True, exist_ok=True)
    return d

def settings_file() -> Path:
    return data_dir() / "settings.json"

def installs_file() -> Path:
    return data_dir() / "installs.json"

def hf_token_file() -> Path:
    return data_dir() / "hf-token.json"

def chat_file(chat_id: str) -> Path:
    return chats_dir() / f"{chat_id}.json"

def read_json(path: Path, fallback):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return fallback

def write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
