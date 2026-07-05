"""Route an incoming request to the right adapter.

Strategy (tiered):
    1. `model_overrides.json` pin            (explicit escape hatch)
    2. Plugin from python/plugins/*.py       (community-contributed)
    3. models/<family>/ model_type registry  (transformers families)
    4a. models/<family>/ library registry    (e.g. diffusion: per-family)
    4b. DiffusersAdapter                     (generic diffusers fallback)
    5. StandardPipelineAdapter               (pipeline_tag in supported set)

If nothing matches we raise. There's no terminal AutoModel fallback because
its only output kind (raw embeddings) isn't rendered by the UI; surfacing a
useful error is more helpful than producing junk JSON.

Tiers 3 and 4a cover per-family inference code. Tier 3 is keyed on the
transformers `model_type` tag (DETR, SAM, Janus, ...). Tier 4a is keyed on
`library_name + repo pattern`, used for runtimes like diffusers that don't
expose a transformers model_type (Stable Diffusion, FLUX, SDXL, ...).
A broken folder won't break siblings.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from adapters import (
    Adapter,
    StandardPipelineAdapter,
    DiffusersAdapter,
    NAMED_ADAPTERS,
)
from models import (
    REGISTRY as MODEL_REGISTRY,
    LOAD_ERRORS as MODEL_LOAD_ERRORS,
    adapter_for_library,
)

_OVERRIDES_CACHE = None
_PLUGIN_CACHE = None

PLUGIN_DIR = Path(__file__).parent / "plugins"
OVERRIDES_PATH = Path(__file__).parent / "model_overrides.json"

def _log(msg: str) -> None:
    print(f"[routing] {msg}", file=sys.stderr, flush=True)

for _family, _err in MODEL_LOAD_ERRORS.items():
    _log(f"models/{_family} failed to load: {type(_err).__name__}: {_err}")

def load_overrides() -> dict:
    global _OVERRIDES_CACHE
    if _OVERRIDES_CACHE is not None:
        return _OVERRIDES_CACHE
    try:
        data = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
        _OVERRIDES_CACHE = (data or {}).get("overrides", {}) or {}
    except Exception as e:
        _log(f"could not load model_overrides.json: {e}")
        _OVERRIDES_CACHE = {}
    return _OVERRIDES_CACHE

def override_for(model_id: str) -> dict:
    return load_overrides().get(model_id, {}) or {}

def _load_plugin_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"inferml_plugin_{path.stem}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def plugin_adapters() -> list[type]:
    """Scan python/plugins/*.py and return a list of Adapter subclasses."""
    global _PLUGIN_CACHE
    if _PLUGIN_CACHE is not None:
        return _PLUGIN_CACHE
    out: list[type] = []
    if not PLUGIN_DIR.is_dir():
        _PLUGIN_CACHE = out
        return out
    for f in sorted(PLUGIN_DIR.iterdir()):
        if not f.is_file() or f.suffix != ".py" or f.name.startswith("_"):
            continue
        try:
            mod = _load_plugin_module(f)
            for name in dir(mod):
                obj = getattr(mod, name)
                if isinstance(obj, type) and issubclass(obj, Adapter) and obj is not Adapter:
                    out.append(obj)
                    _log(f"registered plugin adapter {obj.__name__} from {f.name}")
        except Exception as e:
            _log(f"plugin load failed for {f.name}: {e}")
    _PLUGIN_CACHE = out
    return out

def inspect_model(model_id: str) -> dict:
    """Gather enough metadata to route the model without downloading weights.

    Network calls are best-effort - if we're offline or the model is gated, we
    return what we can and let the adapter raise a clearer error during load.
    """
    info = {
        "model_id": model_id,
        "pipeline_tag": None,
        "library": None,
        "tags": [],
        "architectures": [],
        "model_type": None,
        "config": {},
    }
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        try:
            card = api.model_info(model_id)
            info["pipeline_tag"] = getattr(card, "pipeline_tag", None)
            info["library"] = getattr(card, "library_name", None)
            info["tags"] = list(getattr(card, "tags", None) or [])
        except Exception as e:
            _log(f"model_info failed for {model_id}: {e}")

        try:
            cfg_path = hf_hub_download(model_id, "config.json")
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            info["config"] = cfg
            info["architectures"] = cfg.get("architectures") or []
            info["model_type"] = cfg.get("model_type")
        except Exception as e:
            _log(f"config.json fetch failed for {model_id}: {e}")

    except Exception as e:
        _log(f"huggingface_hub not available: {e}")

    return info

def pick_adapter(info: dict) -> Adapter:
    ovr = override_for(info["model_id"])

    named = ovr.get("adapter")
    if named:
        cls = NAMED_ADAPTERS.get(named)
        if cls:
            inst = cls()
            inst.override = ovr
            _log(f"routing {info['model_id']} → {cls.__name__} (override)")
            return inst

    for cls in plugin_adapters():
        try:
            if cls.can_handle(info):
                inst = cls()
                inst.override = ovr
                _log(f"routing {info['model_id']} → {cls.__name__} (plugin)")
                return inst
        except Exception as e:
            _log(f"plugin {cls.__name__}.can_handle raised: {e}")

    mt = (info.get("model_type") or "").lower()
    if mt and mt in MODEL_REGISTRY:
        entry = MODEL_REGISTRY[mt]
        cls = entry["adapter"]
        try:
            inst = cls()
            inst.override = ovr
            _log(f"routing {info['model_id']} → {cls.__name__} (models/{entry['family']})")
            return inst
        except Exception as e:
            _log(f"models/{entry['family']} adapter init raised: {e}")

    library = (info.get("library") or "").lower()
    if library == "diffusers":
        match = adapter_for_library(library, info["model_id"])
        if match is not None:
            cls, family = match
            try:
                inst = cls()
                inst.override = ovr
                _log(f"routing {info['model_id']} → {cls.__name__} (models/{family})")
                return inst
            except Exception as e:
                _log(f"models/{family} adapter init raised: {e}")

    if DiffusersAdapter.can_handle(info):
        inst = DiffusersAdapter()
        inst.override = ovr
        _log(f"routing {info['model_id']} → DiffusersAdapter")
        return inst

    if StandardPipelineAdapter.can_handle(info):
        inst = StandardPipelineAdapter()
        inst.override = ovr
        _log(f"routing {info['model_id']} → StandardPipelineAdapter")
        return inst

    raise ValueError(
        f"No adapter matched {info['model_id']!r}. "
        f"model_type={info.get('model_type')!r} pipeline_tag={info.get('pipeline_tag')!r}. "
        "Add a folder under python/models/ for this family, or pin it in "
        "python/model_overrides.json with an explicit `adapter` name."
    )
