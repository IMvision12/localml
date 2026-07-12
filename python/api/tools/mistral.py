"""Mistral / Mixtral tool-call format.

Mistral's v3 tokenizer emits `[TOOL_CALLS] [{"name": ..., "arguments": {...}}]`
- a JSON array following the [TOOL_CALLS] control token.
"""
from __future__ import annotations

from .base import make_tool_call, safe_loads, iter_json_values

_MARKER = "[TOOL_CALLS]"

class MistralParser:
    name = "mistral"

    def handles(self, model_type: str, model_id: str) -> bool:
        return model_type in ("mistral", "mixtral") or "mistral" in model_id or "mixtral" in model_id

    def parse(self, text: str) -> list:
        t = text or ""
        idx = t.find(_MARKER)
        payload = t[idx + len(_MARKER):] if idx >= 0 else t
        calls = []
        arr = safe_loads(payload)
        items = arr if isinstance(arr, list) else ([arr] if isinstance(arr, dict) else None)
        if items is None:
            items = [o for o in iter_json_values(payload) if isinstance(o, dict)]
        for obj in items or []:
            if isinstance(obj, dict) and obj.get("name"):
                args = obj.get("arguments", obj.get("parameters", {}))
                calls.append(make_tool_call(obj["name"], args))
        return calls
