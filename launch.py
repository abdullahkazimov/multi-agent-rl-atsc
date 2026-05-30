#!/usr/bin/env python3
"""
launch.py
=========
One command to view the MARL-ATSC results website locally.

Serves the bundled static site (``website/``) over HTTP and opens it in your
browser. Serving over HTTP is required: opening ``index.html`` as a ``file://``
URL makes the browser block the data fetches, so the dashboards stay empty.

Usage
-----
    python launch.py                 # serve website/ at http://localhost:8000
    python launch.py --port 9000     # use a specific port
    python launch.py --no-browser    # don't auto-open a browser (e.g. headless)
    python launch.py --host 0.0.0.0  # expose on the network (not just localhost)
"""
from __future__ import annotations

import argparse
import contextlib
import http.server
import os
import socket
import sys
import threading
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
WEBDIR = os.path.join(ROOT, "website")


class Handler(http.server.SimpleHTTPRequestHandler):
    """Serve from website/, quietly (suppress per-request access logs)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEBDIR, **kwargs)

    def log_message(self, *args):  # noqa: D401 - silence default logging
        pass


def _free_port(start: int, host: str) -> int:
    """Return `start` if free, otherwise the next open port within +50."""
    for port in range(start, start + 50):
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            if s.connect_ex((host, port)) != 0:  # nothing is listening here
                return port
    raise SystemExit(f"No free port found in {start}..{start + 49}.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Serve and open the MARL-ATSC website.")
    ap.add_argument("--port", type=int, default=8000, help="preferred port (default 8000)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1; use 0.0.0.0 to expose)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser")
    args = ap.parse_args(argv)

    if not os.path.exists(os.path.join(WEBDIR, "index.html")):
        sys.exit(f"website/index.html not found next to launch.py (looked in {WEBDIR}).")

    probe_host = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    port = _free_port(args.port, probe_host)
    view_host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0", "") else args.host
    url = f"http://{view_host}:{port}"

    httpd = http.server.ThreadingHTTPServer((args.host, port), Handler)

    print(f"\n  MARL-ATSC website  →  {url}")
    if port != args.port:
        print(f"  (port {args.port} was busy, using {port})")
    print(f"  serving {WEBDIR}")
    print("  press Ctrl+C to stop\n", flush=True)

    if not args.no_browser:
        # Open after a short delay so the server is ready; harmless if headless.
        def _open():
            with contextlib.suppress(Exception):
                webbrowser.open(url)
        threading.Timer(0.6, _open).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
