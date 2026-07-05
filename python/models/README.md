# models/

One folder per supported model family. Each folder owns its inference code so
that breaking one family does not break the rest of the codebase.

## Layout

```
models/
  <family>/
    __init__.py             # registration metadata + ADAPTER export
    adapter.py              # optional. custom load + run logic for the family
  __init__.py               # auto-discovery + registry. don't edit
  _pipeline_helper.py       # helper for transformers pipeline-based families
  _diffusion_helper.py      # helper for diffusion families
  README.md                 # you are here
```

## Registration shapes

Each `<family>/__init__.py` MUST export an `ADAPTER` class. For dispatch, declare ONE of:

**Shape A: model_type-keyed** (transformers families with a `model_type` in their config.json)

```python
MODEL_TYPES = ["detr", "rt_detr"]   # transformers `model_type` tags this family handles
TASK = "object-detection"            # optional. InferML task name hint
ADAPTER = DetrAdapter                # subclass of adapters.base.Adapter
```

**Shape B: library-keyed** (diffusion families. they don't expose a transformers `model_type`)

```python
LIBRARY = "diffusers"
REPO_PATTERNS = ["black-forest-labs/flux*", "*/flux*"]   # fnmatch on `owner/name`, case-insensitive
TASK = "text-to-image"
ADAPTER = FluxAdapter                # typically a DiffusionFamilyAdapter subclass
```

A folder must declare exactly one of those. The registry rejects folders missing both.

## Three kinds of family folders

### Custom-code family (Janus, Florence-2, Moondream, FastVLM, ...)

Full custom `adapter.py` with load + run logic.

```python
# models/janus/__init__.py
from .adapter import JanusAdapter
MODEL_TYPES = ["janus"]
TASK = "image-text-to-text"
ADAPTER = JanusAdapter
```

```python
# models/janus/adapter.py
from adapters.base import Adapter
import output_kinds as ok
from io_utils import decode_image, resolve_device

class JanusAdapter(Adapter):
    @classmethod
    def can_handle(cls, info): ...
    def load(self, info, device): ...
    def run(self, inputs, params): ...
```

### Pipeline-based family (DETR, SAM, Whisper, Llama, ...)

Delegates to the shared `tasks/<task>.py` handler. Three lines:

```python
# models/detr/__init__.py
from models._pipeline_helper import make_pipeline_adapter
MODEL_TYPES = ["detr"]
TASK = "object-detection"
ADAPTER = make_pipeline_adapter("object-detection", name="DetrAdapter")
```

If the family later needs a quirk, promote to a custom `adapter.py` without touching siblings.

### Diffusion family (Stable Diffusion, FLUX, SDXL, ...)

Subclasses `DiffusionFamilyAdapter` to set per-family inference defaults. Library-keyed dispatch.

```python
# models/flux/__init__.py
from models._diffusion_helper import DiffusionFamilyAdapter

class FluxAdapter(DiffusionFamilyAdapter):
    DEFAULT_STEPS = 4         # FLUX-schnell is CFG-distilled
    DEFAULT_GUIDANCE = 0.0

LIBRARY = "diffusers"
TASK = "text-to-image"
REPO_PATTERNS = ["black-forest-labs/flux*", "*/flux*"]
ADAPTER = FluxAdapter
```

User-supplied params still win on conflict - these are just defaults.

## Adding a new family

1. Pick a folder name. lowercase, underscores not hyphens.
2. Create `models/<family>/__init__.py` with the required exports for whichever shape applies.
3. If custom code is needed, add `adapter.py` next to it.
4. For Shape A families, add the `model_type` tags to `python/supported_architectures.json` so the Hub filter surfaces matching repos.
5. Restart the sidecar. The registry rediscovers on import.

## How routing works

1. `models/__init__.py` auto-discovers every subfolder at import time.
2. Shape A folders register in `REGISTRY` (model_type → adapter). First-registered wins on collision.
3. Shape B folders register in `LIBRARY_REGISTRY` (ordered list, library + patterns + adapter).
4. `routing.py` dispatches in order:
   - Tier 3: `REGISTRY[model_type]` if the request has a known model_type
   - Tier 4a: `LIBRARY_REGISTRY` matched against `library_name + model_id`
   - Tier 4b: generic `DiffusersAdapter` for any unmatched diffusers repo
   - Tier 5: `StandardPipelineAdapter` for any unmatched repo with a known pipeline_tag
5. If nothing matches, routing raises a clear "no adapter matched" error pointing the caller to `python/models/`.

Folders that fail to import are recorded in `LOAD_ERRORS` and logged at startup. They don't break the rest of the registry.

## Why this layout

- **Isolation**: a syntax error in `models/janus/adapter.py` doesn't break DETR or SAM or any other family.
- **Discoverability**: each family is a folder you can find in one cd.
- **Quirks have a home**: when a model needs custom behavior, the place to put it is its own folder. no scrolling through a monolith.
- **Per-family tracking**: each folder is a unit of "is this model supported", "does it work today", "who fixed it last".
