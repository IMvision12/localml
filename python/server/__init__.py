"""InferML local web server.

Replaces the Electron shell: a FastAPI + uvicorn app that serves the compiled
React build as static files and (from Phase 2 onward) exposes the inference
backend over HTTP, including an OpenAI-compatible API.

The inference engine is imported in-process (see `python/engine.py`, added in
Phase 2) so native GPU/MPS access is preserved and the OpenAI endpoint can hold
a live handle to the currently-loaded LLM.
"""
from __future__ import annotations

__version__ = "1.1.0"
