"""Hermes / Qwen tool-call format.

Qwen2.5/Qwen3 and NousResearch Hermes models emit tool calls wrapped in
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` tags (one per call).
"""
from __future__ import annotations

import re

from .base import make_tool_call, safe_loads

_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


class HermesQwenParser:
    name = "hermes_qwen"

    def handles(self, model_type: str, model_id: str) -> bool:
        return model_type.startswith("qwen") or "qwen" in model_id or "hermes" in model_id

    def parse(self, text: str) -> list:
        calls = []
        for m in _BLOCK.finditer(text or ""):
            obj = safe_loads(m.group(1))
            if isinstance(obj, dict) and obj.get("name"):
                args = obj.get("arguments", obj.get("parameters", {}))
                calls.append(make_tool_call(obj["name"], args))
        return calls
