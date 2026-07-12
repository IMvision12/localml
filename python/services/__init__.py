"""Stateful services behind the stdio runner: app data, the HF hub client,
hardware sampling, and the chats/settings/installs store.

These were `server/*.py` when InferML ran as a FastAPI web app. The HTTP layer
is gone - `runner.py` drives them directly over stdin/stdout - but the logic is
unchanged, so this package is deliberately transport-agnostic: nothing in here
knows how it is being called.
"""
