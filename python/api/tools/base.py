"""Shared helpers for the per-family tool-call parsers."""
from __future__ import annotations

import json
import uuid

def make_tool_call(name: str, arguments) -> dict:
    """Build an OpenAI `tool_calls[]` entry. `arguments` is emitted as a JSON
    *string* per the spec (dicts are serialized; strings pass through)."""
    if isinstance(arguments, (dict, list)):
        args_str = json.dumps(arguments, ensure_ascii=False)
    elif isinstance(arguments, str):
        args_str = arguments
    elif arguments is None:
        args_str = "{}"
    else:
        args_str = json.dumps(arguments)
    return {
        "id": "call_" + uuid.uuid4().hex[:24],
        "type": "function",
        "function": {"name": name, "arguments": args_str},
    }

def safe_loads(s: str):
    """Parse JSON, tolerating single quotes and trailing junk after the object."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    for obj in iter_json_values(s):
        return obj
    return None

def iter_json_values(text: str):
    """Yield successive top-level JSON objects/arrays found in `text` by scanning
    balanced brackets (string-aware). Handles multiple concatenated or
    `;`-separated objects that some templates emit."""
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "{[":
            end = _match_bracket(text, i)
            if end is not None:
                chunk = text[i:end + 1]
                try:
                    yield json.loads(chunk)
                except Exception:
                    pass
                i = end + 1
                continue
        i += 1

def _match_bracket(text: str, start: int):
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return None
