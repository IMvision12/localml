"""Bridge between the OpenAI request shape and a loaded transformers LLM.

Digs the (model, tokenizer) out of whichever adapter the engine has cached,
builds a prompt via the tokenizer's chat template, and generates - full or
streamed. Tool handling (Phase 4) plugs in via the optional `tools` argument to
`build_inputs` and the parser layer in `openai_api.tools`.
"""
from __future__ import annotations

import threading

class LLMNotLoaded(Exception):
    """No usable text-generation model is available for the request."""

def _extract_model_tokenizer(adapter):
    """Both LLM adapter shapes expose a LoadedPipeline; pull model+tokenizer out.

      - PipelineAdapter (models/qwen, llama, …):  adapter._state
      - StandardPipelineAdapter (fallback):        adapter.state
    Either has `.pipe` (transformers pipeline with .model/.tokenizer) or, for
    the seq2seq-direct path, `.model` + `.processor` (tokenizer).
    """
    state = getattr(adapter, "_state", None) or getattr(adapter, "state", None)
    if state is None:
        return None, None
    model = tokenizer = None
    pipe = getattr(state, "pipe", None)
    if pipe is not None:
        model = getattr(pipe, "model", None)
        tokenizer = getattr(pipe, "tokenizer", None)
    if model is None:
        model = getattr(state, "model", None)
    if tokenizer is None:
        tokenizer = getattr(state, "processor", None)
    if tokenizer is not None and not hasattr(tokenizer, "encode"):
        tokenizer = None
    return model, tokenizer

_GENERIC_MODEL_NAMES = {"", "inferml", "default", "current", "gpt-3.5-turbo", "gpt-4", "gpt-4o"}

def resolve_llm(engine, requested_model: str | None):
    """Return (model, tokenizer, model_id). Prefers an explicitly named,
    loadable model; otherwise the currently-loaded LLM. Raises LLMNotLoaded."""
    adapter = None
    model_id = None

    name = (requested_model or "").strip()
    if name and name.lower() not in _GENERIC_MODEL_NAMES:
        adapter = engine.get_cached_adapter(name)
        if adapter is not None:
            model_id = name
        else:
            try:
                adapter = engine.ensure_loaded(name, "text-generation")
                model_id = name
            except Exception:
                adapter = None

    if adapter is None:
        cur = engine.current_llm_id()
        if cur:
            adapter = engine.get_cached_adapter(cur)
            model_id = cur

    if adapter is None:
        raise LLMNotLoaded(
            "No LLM is loaded. Open a text-generation model in InferML first, "
            "or pass a valid installed model id as `model`."
        )

    model, tokenizer = _extract_model_tokenizer(adapter)
    if model is None or tokenizer is None:
        raise LLMNotLoaded(
            f"The loaded model {model_id!r} isn't a text-generation model - "
            "the OpenAI endpoint only serves LLMs."
        )
    return model, tokenizer, model_id

def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return str(content)

def normalize_messages(messages: list) -> list:
    out = []
    for m in messages or []:
        msg = {"role": m.get("role", "user"), "content": _content_to_text(m.get("content"))}
        for k in ("tool_calls", "tool_call_id", "name"):
            if m.get(k) is not None:
                msg[k] = m[k]
        out.append(msg)
    return out

def build_inputs(tokenizer, messages: list, tools=None):
    """Apply the tokenizer's chat template (with tools if provided) and return a
    dict-like of model inputs (`input_ids` [+ `attention_mask`]). Falls back to a
    plain role-tagged prompt for base models with no chat template.

    Returns a BatchEncoding / dict so generation can splat it with `**inputs`,
    which is required in transformers 5.x where apply_chat_template yields a
    BatchEncoding, not a bare tensor.
    """
    norm = normalize_messages(messages)
    try:
        kwargs = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        if tools:
            kwargs["tools"] = tools
        enc = tokenizer.apply_chat_template(norm, **kwargs)
        if hasattr(enc, "shape") and not hasattr(enc, "keys"):
            return {"input_ids": enc}
        return enc
    except Exception:
        try:
            kwargs = dict(add_generation_prompt=True, return_tensors="pt")
            if tools:
                kwargs["tools"] = tools
            ids = tokenizer.apply_chat_template(norm, **kwargs)
            return {"input_ids": ids}
        except Exception:
            text = ""
            for m in norm:
                text += f"{m['role']}: {m['content']}\n"
            text += "assistant:"
            return tokenizer(text, return_tensors="pt")

def _gen_kwargs(tokenizer, params: dict) -> dict:
    kwargs: dict = {"max_new_tokens": int(params.get("max_tokens") or 512)}
    temp = params.get("temperature")
    if temp is not None and float(temp) > 0:
        kwargs["do_sample"] = True
        kwargs["temperature"] = float(temp)
    else:
        kwargs["do_sample"] = False
    if params.get("top_p") is not None:
        kwargs["top_p"] = float(params["top_p"])
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        kwargs["pad_token_id"] = tokenizer.eos_token_id
    return kwargs

def _to_device(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

def _prompt_len(inputs) -> int:
    ids = inputs["input_ids"]
    return int(ids.shape[-1])

def generate_full(model, tokenizer, inputs, params: dict):
    """Non-streaming generation. Returns (text, prompt_tokens, completion_tokens)."""
    import torch
    device = next(model.parameters()).device
    inputs = _to_device(inputs, device)
    kwargs = _gen_kwargs(tokenizer, params)
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    prompt_len = _prompt_len(inputs)
    new_tokens = out[0][prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    text = _apply_stop(text, params.get("stop"))
    return text, prompt_len, int(new_tokens.shape[-1])

def stream_generate(model, tokenizer, inputs, params: dict, stop_flag=lambda: False):
    """Yield decoded text deltas as the model generates. Runs generate() on a
    worker thread and drains a TextIteratorStreamer. Honors a cooperative
    stop_flag between tokens."""
    from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
    import torch

    device = next(model.parameters()).device
    inputs = _to_device(inputs, device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    class _StopCriteria(StoppingCriteria):
        def __call__(self, ids, scores, **kw):
            return bool(stop_flag())

    kwargs = _gen_kwargs(tokenizer, params)
    kwargs.update(dict(inputs))
    kwargs.update(
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([_StopCriteria()]),
    )

    error = {}

    def _worker():
        try:
            with torch.no_grad():
                model.generate(**kwargs)
        except Exception as e:
            error["e"] = e

    thread = threading.Thread(target=_worker, name="oai-generate", daemon=True)
    thread.start()
    stops = _stop_list(params.get("stop"))
    acc = ""
    for piece in streamer:
        acc += piece
        cut = _first_stop_index(acc, stops)
        if cut is not None:
            remaining = acc[:cut]
            already = len(acc) - len(piece)
            if remaining and remaining[already:]:
                yield remaining[already:]
            break
        yield piece
    thread.join()
    if "e" in error:
        raise error["e"]

def _stop_list(stop):
    if not stop:
        return []
    return [stop] if isinstance(stop, str) else [s for s in stop if isinstance(s, str)]

def _first_stop_index(text: str, stops: list):
    idxs = [text.find(s) for s in stops if s and text.find(s) >= 0]
    return min(idxs) if idxs else None

def _apply_stop(text: str, stop):
    for s in _stop_list(stop):
        i = text.find(s)
        if i >= 0:
            text = text[:i]
    return text
