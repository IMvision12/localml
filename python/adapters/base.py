"""Adapter base class.

One instance of an adapter = one loaded model. The router picks the adapter,
calls `load(info, device)` once, then `run(inputs, params)` per request.
Instances are cached by (adapter_class, model_id) in the engine.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Adapter(ABC):
    # Override options merged with request params. Safe to read even before load().
    override: dict = {}

    @classmethod
    def can_handle(cls, info: dict) -> bool:
        """Return True if this adapter can run the described model.

        `info` is the dict from routing.inspect_model. Implementations should
        inspect `model_id`, `model_type`, `architectures`, `tags`, etc. -
        *not* download any weights."""
        return False

    @abstractmethod
    def load(self, info: dict, device) -> None:
        """Instantiate the underlying model + any helpers (processor, tokenizer)."""

    @abstractmethod
    def run(self, inputs: dict, params: dict) -> dict:
        """Execute inference. Must return a dict matching one of the kinds in
        `output_kinds.py` (`boxes`, `masks`, `labels`, `text`, `image`,
        `audio`, `vector`)."""

    def unload(self) -> None:
        """Hook for freeing GPU memory - default: drop references."""
        for attr in list(self.__dict__.keys()):
            if attr not in ("override",):
                setattr(self, attr, None)
