"""OpenAI request/response shaping for /v1/embeddings.

The actual model loading + encoding lives in the shared, layer-neutral
`embedding_backend` module (also used by the engine's feature-extraction task,
so the UI and the API share one loaded copy). This file only does the
OpenAI-specific parts: model-name resolution, coercing the many `input` shapes,
and token accounting.
"""
from __future__ import annotations

from embedding_backend import load_embedder, finalize


class EmbeddingError(Exception):
    """A bad request the caller should turn into a 400."""


# Generic OpenAI embedding model names (and InferML placeholders) map onto a
# small, fast default so drop-in clients that hard-code `text-embedding-3-small`
# still work. Any real Hugging Face id is used verbatim.
_GENERIC_EMBED_NAMES = {
    "", "inferml", "default", "current",
    "text-embedding-3-small", "text-embedding-3-large",
    "text-embedding-ada-002", "text-embedding-002", "text-embedding-3",
    "ada", "ada-002",
}
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def resolve_model_id(requested: str | None) -> str:
    name = (requested or "").strip()
    if not name or name.lower() in _GENERIC_EMBED_NAMES:
        return DEFAULT_EMBED_MODEL
    return name


def _decode(emb, ids) -> str:
    tok = getattr(emb, "tokenizer", None)
    if tok is None:
        raise EmbeddingError("This embedding model can't decode token-id input.")
    return tok.decode([int(i) for i in ids], skip_special_tokens=True)


def _coerce_inputs(inputs, emb) -> list[str]:
    """OpenAI accepts a string, an array of strings, an array of token ids, or an
    array of token-id arrays. Normalize all of them to a list of strings."""
    if inputs is None:
        return []
    if isinstance(inputs, str):
        return [inputs]
    if isinstance(inputs, list):
        if not inputs:
            return []
        if all(isinstance(x, str) for x in inputs):
            return list(inputs)
        if all(isinstance(x, bool) for x in inputs):
            pass  # bool is an int subclass; fall through to the error
        elif all(isinstance(x, int) for x in inputs):
            return [_decode(emb, inputs)]
        elif all(isinstance(x, list) for x in inputs):
            return [_decode(emb, seq) for seq in inputs]
    raise EmbeddingError(
        "`input` must be a string, an array of strings, or an array of token-id arrays."
    )


def embed(requested_model, inputs, normalize: bool = True, dimensions=None) -> dict:
    """Return {model_id, vectors (numpy 2-D), prompt_tokens}. Raises
    EmbeddingError for bad requests; lets load/inference errors propagate so the
    route can map them via engine.actionable_error."""
    model_id = resolve_model_id(requested_model)
    emb = load_embedder(model_id)
    texts = _coerce_inputs(inputs, emb)
    if not texts:
        raise EmbeddingError("`input` must be a non-empty string or array.")

    vecs = finalize(emb.raw_encode(texts), normalize, dimensions)

    tok = getattr(emb, "tokenizer", None)
    if tok is not None:
        prompt_tokens = int(sum(len(tok(t).get("input_ids", [])) for t in texts))
    else:
        prompt_tokens = int(sum(len(t.split()) for t in texts))

    return {"model_id": model_id, "vectors": vecs, "prompt_tokens": prompt_tokens}
