"""Feature extraction / text embeddings.

Embedding models (sentence-transformers, bge, e5, gte, MiniLM, nomic, …) don't
fit the standard `transformers.pipeline()` shape well and their output - a raw
vector - isn't one of the visual output kinds, so this task bypasses the default
loader and uses the shared `embedding_backend`. The UI renders the `vector`
output kind: dimensionality plus a short preview of the values.
"""
from __future__ import annotations

from .base import TaskHandler, TaskVariant, LoadedPipeline
import output_kinds as ok


class TextEmbeddingVariant(TaskVariant):
    name = "standard"

    def can_handle(self, info, inputs):
        return bool((inputs.get("text") or "").strip())

    def run(self, state, inputs, params):
        from embedding_backend import finalize

        text = inputs["text"].strip()
        embedder = state.model  # the load below stashes the embedder here
        normalize = bool(params.get("normalize", True))
        dimensions = params.get("dimensions")
        vecs = finalize(embedder.raw_encode([text]), normalize, dimensions)
        return ok.vector(vecs[0])


class FeatureExtractionTask(TaskHandler):
    name = "feature-extraction"
    output_kind = "vector"
    default_params = {"normalize": True}
    variants = [TextEmbeddingVariant()]

    def load_pipeline(self, info, device, extra_kwargs=None) -> LoadedPipeline:
        from embedding_backend import load_embedder

        embedder = load_embedder(info["model_id"])
        return LoadedPipeline(
            info=info,
            device=device,
            pipe=None,
            model=embedder,  # variants read the embedder off state.model
            processor=getattr(embedder, "tokenizer", None),
        )


class SentenceSimilarityTask(FeatureExtractionTask):
    # Many embedding repos carry pipeline_tag `sentence-similarity` rather than
    # `feature-extraction`; route both to the same embedding path.
    name = "sentence-similarity"
