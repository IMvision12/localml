"""Llama 3.1 / 3.2 tool-call format.

Llama emits either a `<|python_tag|>`-prefixed JSON object or bare JSON, using
the key `parameters` (not `arguments`). Multiple calls may be `;`-separated.
"""
from __future__ import annotations

from .base import make_tool_call, iter_json_values

_STRIP = ["<|python_tag|>", "<|eom_id|>", "<|eot_id|>"]


class LlamaParser:
    name = "llama"

    def handles(self, model_type: str, model_id: str) -> bool:
        return model_type in ("llama", "mllama") or "llama" in model_id

    def parse(self, text: str) -> list:
        t = text or ""
        for tok in _STRIP:
            t = t.replace(tok, "")
        t = t.strip()
        calls = []
        for obj in iter_json_values(t):
            if isinstance(obj, dict) and obj.get("name"):
                args = obj.get("parameters", obj.get("arguments", {}))
                calls.append(make_tool_call(obj["name"], args))
        return calls
