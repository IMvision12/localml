"""Filesystem locations the server needs.

The compiled React build ("webui") is resolved from, in priority order:
  1. $INFERML_WEBUI_DIR                      explicit override
  2. python/server/webui/                    bundled in the package (Phase 5)
  3. <repo>/src/renderer/dist/               dev: `npm run build:renderer` output
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve()
PYTHON_DIR = _HERE.parents[1]
REPO_ROOT = _HERE.parents[2]

def webui_dir() -> Path:
    env = os.environ.get("INFERML_WEBUI_DIR")
    if env:
        return Path(env)
    bundled = PYTHON_DIR / "server" / "webui"
    if (bundled / "index.html").exists():
        return bundled
    return REPO_ROOT / "src" / "renderer" / "dist"
