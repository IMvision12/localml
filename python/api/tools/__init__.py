"""Tool-call parsing, keyed on model family.

Parsing generated output back into structured `tool_calls` is model-specific:
Qwen/Hermes, Llama, and Mistral each emit a different format. This layer detects
the family (from the loaded adapter's `model_type`, falling back to the model id)
and dispatches to the matching parser. An unknown family raises ToolFormatUnknown
so the endpoint returns a clear error rather than silently mis-parsing.
"""
from __future__ import annotations

from engine import ENGINE
from .hermes_qwen import HermesQwenParser
from .llama import LlamaParser
from .mistral import MistralParser

_PARSERS = [HermesQwenParser(), LlamaParser(), MistralParser()]


class ToolFormatUnknown(Exception):
    """Raised when a loaded model has no known tool-call output format."""


def _model_type_for(model_id: str) -> str:
    adapter = ENGINE.get_cached_adapter(model_id)
    info = getattr(adapter, "info", None) or {}
    return (info.get("model_type") or "").lower()


def get_parser(model_id: str):
    mt = _model_type_for(model_id)
    mid = (model_id or "").lower()
    for p in _PARSERS:
        try:
            if p.handles(mt, mid):
                return p
        except Exception:
            continue
    return None


def supported_families() -> list[str]:
    return ["Qwen/Hermes", "Llama", "Mistral"]


def parse_tool_calls(model_id: str, text: str) -> list:
    """Return OpenAI `tool_calls[]` parsed from `text`. Empty list if the model
    (from a known family) simply answered in plain text. Raises
    ToolFormatUnknown if the family has no registered parser."""
    parser = get_parser(model_id)
    if parser is None:
        raise ToolFormatUnknown(
            f"Model {model_id!r} has no known tool-call format. InferML currently "
            f"parses tool calls for: {', '.join(supported_families())}."
        )
    return parser.parse(text or "")
