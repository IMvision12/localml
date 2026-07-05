"""`inferml` console entry point.

Starts uvicorn on a configurable port (default 11500), prints the URL, and
opens the browser. This is the single command a pipx-installed user runs.
"""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

_PY_DIR = Path(__file__).resolve().parents[1]
if str(_PY_DIR) not in sys.path:
    sys.path.insert(0, str(_PY_DIR))

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11500

def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="inferml",
        description="Run the InferML local web server (serves the UI + inference API).",
    )
    p.add_argument("--host", default=DEFAULT_HOST,
                   help=f"Bind address (default {DEFAULT_HOST}; use 0.0.0.0 to expose on the LAN).")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"Port (default {DEFAULT_PORT}).")
    p.add_argument("--no-browser", action="store_true",
                   help="Do not open a browser window on start.")
    p.add_argument("--reload", action="store_true",
                   help="Auto-reload on source changes (development).")
    return p.parse_args(argv)

def _force_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

def main(argv=None) -> None:
    _force_utf8_console()
    args = _parse_args(argv)

    import uvicorn

    display_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    url = f"http://{display_host}:{args.port}"
    print(
        "\n  InferML - local, on-device model runner\n"
        f"  ->  {url}\n"
        "  Open that URL in your browser. Press Ctrl+C to stop.\n",
        flush=True,
    )

    if not args.no_browser and not args.reload:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    if args.reload:
        uvicorn.run("server.app:app", host=args.host, port=args.port, reload=True)
    else:
        from server.app import app
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
