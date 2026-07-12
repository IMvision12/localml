"""Shared text-embedding core.

Loads an embedding model and turns text into vectors, behind the engine task
`tasks/feature_extraction.py` (which renders a `vector` output in the UI).

It prefers sentence-transformers when installed - it reads each model's own
pooling config, so bge / e5 / gte / MiniLM all come out right - and falls back
to a plain transformers AutoModel with attention-masked mean pooling otherwise.
Loaded models are cached process-wide by id; callers serialize inference
themselves (the server holds one lock; torch isn't thread-safe against itself).
"""
from __future__ import annotations

import threading

_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


class _STEmbedder:
    """sentence-transformers backend - correct pooling per the model's config."""
    kind = "sentence-transformers"

    def __init__(self, model_id: str):
        from sentence_transformers import SentenceTransformer
        from io_utils import resolve_device

        dev = resolve_device()
        device = getattr(dev, "type", None) if dev else None
        self.model = SentenceTransformer(model_id, device=device or "cpu")
        self.tokenizer = self.model.tokenizer

    def raw_encode(self, texts: list[str]):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)


class _HFEmbedder:
    """transformers fallback: AutoModel + attention-masked mean pooling."""
    kind = "transformers"

    def __init__(self, model_id: str):
        import torch  # noqa: F401
        from transformers import AutoModel, AutoTokenizer
        from io_utils import resolve_device

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        dev = resolve_device()
        self.device = dev if dev else "cpu"
        try:
            self.model.to(self.device)
        except Exception:
            self.device = "cpu"

    def raw_encode(self, texts: list[str]):
        import torch

        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        last_hidden = out.last_hidden_state  # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean = summed / counts
        return mean.cpu().numpy()


def _build(model_id: str):
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        return _HFEmbedder(model_id)
    try:
        return _STEmbedder(model_id)
    except Exception:
        # sentence-transformers is present but couldn't build this model; a
        # plain AutoModel may still work (and gives a clearer error if not).
        return _HFEmbedder(model_id)


def load_embedder(model_id: str):
    """Return a cached embedder for `model_id`, loading it on first use. The
    returned object exposes `.raw_encode(list[str]) -> ndarray`, `.tokenizer`,
    and `.kind`."""
    with _CACHE_LOCK:
        emb = _CACHE.get(model_id)
    if emb is not None:
        return emb
    emb = _build(model_id)
    with _CACHE_LOCK:
        _CACHE[model_id] = emb
    return emb


def evict(model_id: str) -> bool:
    """Drop a cached embedder so its weights can be freed. Returns True if one
    was present."""
    with _CACHE_LOCK:
        return _CACHE.pop(model_id, None) is not None


def finalize(vecs, normalize: bool = True, dimensions=None):
    """Shape raw encoder output into a 2-D float32 array, optionally truncating
    to `dimensions` (Matryoshka) and L2-normalizing each row."""
    import numpy as np

    arr = np.asarray(vecs, dtype="float32")
    if arr.ndim == 1:
        arr = arr[None, :]
    if dimensions and int(dimensions) > 0 and int(dimensions) < arr.shape[1]:
        arr = arr[:, : int(dimensions)]
    if normalize:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
    return arr
