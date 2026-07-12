"""Hugging Face search / model info / token / cache - Python port of Electron's
`services/huggingface.js`, `services/hf-auth.js`, and `services/hf-cache.js`.

Uses the same `python/supported_architectures.json` whitelist so search results
match exactly what the inference backend can actually run.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

from services.appdata import hf_token_file, read_json, write_json

_PY_DIR = Path(__file__).resolve().parents[1]
_SERVER_DIR = Path(__file__).resolve().parent

def _matrix_path() -> Path:
    for cand in (_PY_DIR / "supported_architectures.json",
                 _SERVER_DIR / "_data" / "supported_architectures.json"):
        if cand.exists():
            return cand
    return _PY_DIR / "supported_architectures.json"

_TYPES_BY_TASK: dict[str, set] = {}
_LIBRARY_PASSTHROUGH: set = set()

try:
    _raw = json.loads(_matrix_path().read_text(encoding="utf-8"))
    for _k, _v in _raw.items():
        if _k.startswith("_") or not isinstance(_v, list):
            continue
        _TYPES_BY_TASK[_k] = {str(x).lower() for x in _v}
    _LIBRARY_PASSTHROUGH = {str(x).lower() for x in (_raw.get("_library_passthrough", {}).get("libraries") or [])}
except Exception as e:  # pragma: no cover
    print(f"[hf_service] failed to load supported_architectures.json: {e}")

# Text-embedding pipeline tags. These aren't in supported_architectures.json
# (embeddings run through the generic sentence-transformers / mean-pooling
# backend, which handles any text encoder, so there's no fixed architecture
# whitelist). They're handled explicitly in _is_embedding_model below.
_EMBED_TASKS = {"feature-extraction", "sentence-similarity"}

# Decoder-LM families our engine routes to a text-generation family adapter, not
# the embedding task. Some decoder-based embedders (gte-Qwen, e5-mistral,
# Qwen3-Embedding) carry embedding tags but can't run through our vector path, so
# they're filtered out of the Embeddings tab even when their arch tag is missing.
_DECODER_EMBED_FAMILIES = ("qwen", "llama", "mistral", "mixtral", "gemma", "falcon", "olmo", "bloom", "stablelm")

_UNSUPPORTED_FORMAT_TAGS = {"gguf", "ggml", "llama.cpp", "exl2", "exllama", "exllamav2"}
_TRUST_REMOTE_CODE_LIBRARIES = {"ml-fastvlm", "mistral-common"}

_WEIGHT_FORMAT_ORDER = ["safetensors", "bin", "pt", "ckpt", "msgpack", "h5", "onnx", "ot"]
_WEIGHT_EXT_RX = re.compile(r"\.(safetensors|bin|pt|ckpt|msgpack|h5|onnx|ot)$", re.IGNORECASE)

_MODEL_ID_RX = re.compile(r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$")

def is_valid_model_id(mid) -> bool:
    return isinstance(mid, str) and 0 < len(mid) <= 200 and bool(_MODEL_ID_RX.match(mid))

def hf_cache_dir() -> Path:
    """The Hugging Face hub cache directory, resolved exactly the way
    huggingface_hub resolves it - the single cross-platform source of truth
    (macOS, Windows, Linux). Respects HF_HUB_CACHE / HF_HOME / XDG_CACHE_HOME."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except Exception:
        if os.environ.get("HF_HUB_CACHE"):
            return Path(os.environ["HF_HUB_CACHE"])
        hf_home = os.environ.get("HF_HOME")
        if not hf_home:
            xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
            hf_home = os.path.join(xdg, "huggingface")
        return Path(hf_home) / "hub"

def _lower(x) -> str:
    return str(x or "").lower()

def _resolve_task(m: dict):
    tags = [_lower(t) for t in (m.get("tags") or [])]
    for task, types in _TYPES_BY_TASK.items():
        if any(t in types for t in tags):
            return task
    if m.get("pipeline_tag") and m["pipeline_tag"] in _TYPES_BY_TASK:
        return m["pipeline_tag"]
    for t in tags:
        if t in _TYPES_BY_TASK:
            return t
    # Embedding tags aren't in the arch whitelist; recognize them last so a
    # model with a "real" task (ASR / text-gen) that *also* carries a
    # feature-extraction tag resolves to its real task above, not to embeddings.
    if _lower(m.get("pipeline_tag")) in _EMBED_TASKS:
        return _lower(m.get("pipeline_tag"))
    return None

def _is_embedding_model(m: dict) -> bool:
    """A text-embedding model our generic embedding backend can actually run:
    tagged as a sentence-transformers / feature-extraction / sentence-similarity
    model, loadable via transformers, and whose architecture doesn't belong to
    another task. Audio feature extractors (wav2vec2/wavlm) and decoder-based
    embedders (Qwen3-Embedding) carry embedding tags but resolve to ASR /
    text-generation by architecture - our engine routes those elsewhere, so we
    exclude them rather than open a workspace that can't produce a vector."""
    library = _lower(m.get("library_name"))
    tags = [_lower(t) for t in (m.get("tags") or [])]
    is_st = library == "sentence-transformers" or "sentence-transformers" in tags
    pipeline = _lower(m.get("pipeline_tag"))
    if not (is_st or pipeline in _EMBED_TASKS):
        return False
    if library not in ("transformers", "sentence-transformers", ""):
        return False
    if _resolve_task(m) not in _EMBED_TASKS:
        return False
    hay = _lower(m.get("id") or m.get("modelId")) + " " + " ".join(tags)
    if any(fam in hay for fam in _DECODER_EMBED_FAMILIES):
        return False
    return True

def _is_natively_supported(m: dict) -> bool:
    library = _lower(m.get("library_name"))
    tags = [_lower(t) for t in (m.get("tags") or [])]
    if any(t in _UNSUPPORTED_FORMAT_TAGS for t in tags):
        return False
    if _is_embedding_model(m):
        return True
    if library in _LIBRARY_PASSTHROUGH:
        return True
    if library in ("transformers", "") or library in _TRUST_REMOTE_CODE_LIBRARIES:
        pass
    else:
        return False
    task = _resolve_task(m)
    if not task:
        return False
    allowed = _TYPES_BY_TASK.get(task)
    if not allowed:
        return False
    return any(t in allowed for t in tags)

def _hf_headers() -> dict:
    headers = {"User-Agent": "inferml"}
    token = get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def _http_get_json(url: str, timeout: float = 20.0):
    req = urllib.request.Request(url, headers=_hf_headers())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def _fetch_model_list(q=None, task=None, library=None, limit=100) -> list:
    params = {"limit": str(limit), "full": "true", "sort": "downloads", "direction": "-1"}
    if q:
        params["search"] = q
    if task:
        params["pipeline_tag"] = task
    if library:
        params["library"] = library
    url = "https://huggingface.co/api/models?" + urllib.parse.urlencode(params)
    return _http_get_json(url)

def search(q=None, task=None) -> dict:
    # The Embeddings tab (feature-extraction / sentence-similarity) fans out to
    # both tags - the best encoders are split across them on the Hub - and then
    # keeps only genuine embedding models.
    embed_search = _lower(task) in _EMBED_TASKS
    search_tasks = ["sentence-similarity", "feature-extraction"] if embed_search else [task]

    queries = []
    for st in search_tasks:
        queries.append({"q": q, "task": st, "library": "transformers", "limit": 100})
        if st or q:
            queries.append({"q": q, "task": st, "limit": 100})

    lists = []
    errors = []
    for opts in queries:
        try:
            lists.append(_fetch_model_list(**opts))
        except Exception as e:
            errors.append(e)
    if not lists and errors:
        raise errors[0]

    by_id: dict[str, dict] = {}
    for lst in lists:
        for m in lst:
            mid = m.get("id") or m.get("modelId")
            if not mid or mid in by_id:
                continue
            if not _is_natively_supported(m):
                continue
            if embed_search and not _is_embedding_model(m):
                continue
            by_id[mid] = m

    merged = sorted(by_id.values(), key=lambda m: -(m.get("downloads") or 0))

    from services.store_service import list_installed
    installs = list_installed()

    items = []
    for m in merged:
        mid = m.get("id") or m.get("modelId")
        size = _estimate_size(m)
        items.append({
            "id": mid,
            "path": mid,
            "nm": mid.split("/")[-1],
            "author": mid.split("/")[0],
            "task": _resolve_task(m) or task or "",
            "library": m.get("library_name"),
            "tags": m.get("tags") if isinstance(m.get("tags"), list) else [],
            "params": _estimate_params(m),
            "size": _human_bytes(size) if size else "-",
            "dl": _human_compact(m.get("downloads") or 0),
            "likes": m.get("likes") or 0,
            "desc": (_card(m, "summary") or _card(m, "description") or _short_lib(m) or ""),
            "license": _card(m, "license") or "",
            "hw": _size_to_hw(size),
            "installed": mid in installs,
        })
    return {"items": items}

def model_info(mid: str) -> dict:
    if not is_valid_model_id(mid):
        return {"id": mid, "size": None, "bytes": 0, "error": "Invalid model id"}
    safe = "/".join(urllib.parse.quote(seg) for seg in str(mid).split("/"))
    try:
        m = _http_get_json(f"https://huggingface.co/api/models/{safe}?blobs=true")
        bytes_ = _estimate_size(m) or (0 if isinstance(m.get("siblings"), list) else (m.get("usedStorage") or 0))
        return {"id": mid, "bytes": bytes_, "size": _human_bytes(bytes_) if bytes_ else None, "hw": _size_to_hw(bytes_)}
    except Exception:
        return {"id": mid, "size": None, "bytes": 0}

def get_token():
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env
    return (read_json(hf_token_file(), {}) or {}).get("token")

def get_masked_token():
    t = get_token()
    if not t:
        return None
    return t[:3] + "…" + t[-4:] if len(t) > 8 else "…"

def set_token(token: str) -> dict:
    write_json(hf_token_file(), {"token": token})
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return {"ok": True}

def clear_token() -> dict:
    try:
        hf_token_file().unlink()
    except Exception:
        pass
    os.environ.pop("HF_TOKEN", None)
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    return {"ok": True}

def verify_token(token: str) -> dict:
    try:
        req = urllib.request.Request(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}", "User-Agent": "inferml"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            who = json.loads(r.read().decode("utf-8"))
        return {"ok": True, "name": who.get("name"), "type": who.get("type")}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _cache_roots() -> list[Path]:
    roots = [hf_cache_dir()]
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    if os.environ.get("XDG_CACHE_HOME"):
        roots.append(Path(os.environ["XDG_CACHE_HOME"]) / "huggingface" / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    seen, out = set(), []
    for r in roots:
        try:
            rp = r.resolve()
        except Exception:
            rp = r
        if rp not in seen:
            seen.add(rp)
            out.append(r)
    return out

def delete_model_cache(mid: str) -> dict:
    dir_name = "models--" + str(mid).replace("/", "--")
    removed, errors = [], []
    for root in _cache_roots():
        candidate = root / dir_name
        if candidate.is_dir():
            try:
                shutil.rmtree(candidate)
                removed.append(str(candidate))
            except Exception as e:
                errors.append({"path": str(candidate), "error": str(e)})
    return {"removed": removed, "errors": errors}

def _card(m, key):
    cd = m.get("cardData") or {}
    return cd.get(key)

def _estimate_params(m: dict) -> str:
    for t in (m.get("tags") or []):
        mt = re.search(r"(\d+(?:\.\d+)?[bmk])", str(t), re.IGNORECASE)
        if mt:
            return mt.group(1).upper()
    m2 = re.search(r"(\d+(?:\.\d+)?)\s*b\b", _lower(m.get("id")))
    return (m2.group(1) + "B") if m2 else ""

def _estimate_size(m: dict) -> int:
    sibs = m.get("siblings") if isinstance(m.get("siblings"), list) else []
    if not sibs:
        return m.get("usedStorage") or 0
    groups: dict[str, list] = {}
    for s in sibs:
        mt = _WEIGHT_EXT_RX.search(_lower(s.get("rfilename")))
        if not mt or not s.get("size"):
            continue
        groups.setdefault(mt.group(1).lower(), []).append(s)
    for ext in _WEIGHT_FORMAT_ORDER:
        if groups.get(ext):
            return sum((s.get("size") or 0) for s in groups[ext])
    return m.get("usedStorage") or 0

def _size_to_hw(b) -> str:
    if not b:
        return "ok"
    return "warn" if (b / (1024 ** 3)) > 20 else "ok"

def _short_lib(m: dict) -> str:
    tags = m.get("tags") or []
    for lib in ("transformers", "diffusers", "peft"):
        if lib in tags:
            return f"{lib} · {m.get('pipeline_tag') or ''}"
    return m.get("pipeline_tag") or ""

def _human_bytes(b) -> str:
    if not b:
        return ""
    units = ["B", "KB", "MB", "GB", "TB"]
    i, n = 0, float(b)
    while n >= 1024 and i < len(units) - 1:
        n /= 1024
        i += 1
    return f"{n:.1f} {units[i]}" if (n < 10 and i > 0) else f"{n:.0f} {units[i]}"

def _human_compact(n) -> str:
    if not n:
        return "0"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}k"
    return str(n)
