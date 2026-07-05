"""OpenAI-compatible API surface (`/v1/*`).

Routes the standard OpenAI chat-completions request to whichever LLM is
currently loaded in InferML (or lazy-loads one named in the request), so agent
frameworks - LangChain, LangGraph, the OpenAI SDK - can point `base_url` at
InferML the way they point at Ollama.

Named `openai_api` (not `openai`) to avoid shadowing the real `openai` SDK.
"""
