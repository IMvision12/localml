"""Unsolicited notifications from the engine to the UI.

Three things happen without anyone asking: hardware ticks (`hw:update`), the
chat list changing (`chats:updated`), and models appearing or disappearing
(`hf:installsChanged`). Everything else is request/response.

The old web build fanned these out to EventSource subscribers over SSE, so this
was an asyncio hub with a queue per connected client. There is exactly one
consumer now - the Electron main process on the other end of our stdout - so it
collapses to a single sink.

The sink is injected by `runner.py` rather than imported, which keeps this
module (and `store_service`, which publishes through it) unaware of the
transport. Publishing before a sink is installed is a no-op, not an error: it is
normal for services to be imported and used before the loop is wired up.
"""
from __future__ import annotations

from typing import Any, Callable, Optional


class EventHub:
    def __init__(self) -> None:
        self._sink: Optional[Callable[[str, Any], None]] = None

    def set_sink(self, sink: Callable[[str, Any], None]) -> None:
        self._sink = sink

    def publish(self, event: str, data: Any = None) -> None:
        sink = self._sink
        if sink is None:
            return
        # A broken sink must never take down the caller: `publish` is invoked
        # from inside store mutations and the hardware poll thread, and neither
        # has any business failing because the UI went away.
        try:
            sink(event, data)
        except Exception:
            pass


HUB = EventHub()
