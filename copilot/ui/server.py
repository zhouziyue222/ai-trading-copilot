# -*- coding: utf-8 -*-
"""Run the local FastAPI dashboard."""

from __future__ import annotations

import argparse
import os
import sys

import uvicorn
from ai_trading_copilot.copilot.services.run_lifecycle import discard_active_runs
from ai_trading_copilot.copilot.services.windows_job import ensure_windows_job


class LifecycleServer(uvicorn.Server):
    def run(self, sockets=None):
        ensure_windows_job()
        return super().run(sockets=sockets)

    def handle_exit(self, sig, frame):
        discard_active_runs()
        super().handle_exit(sig, frame)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    os.environ.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    parser = argparse.ArgumentParser(description="Run the AI trading copilot web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--reports-dir", help="Override the report directory (also isolates test runs).")
    args = parser.parse_args()
    if args.reports_dir:
        os.environ["COPILOT_UI_REPORTS_DIR"] = os.path.abspath(args.reports_dir)
    ensure_windows_job()
    config = uvicorn.Config(
        "ai_trading_copilot.copilot.ui.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        use_colors=bool(sys.stderr.isatty()),
    )
    server = LifecycleServer(config)
    try:
        if config.should_reload:
            from uvicorn.supervisors import ChangeReload
            sock = config.bind_socket()
            try:
                ChangeReload(config, target=server.run, sockets=[sock]).run()
            finally:
                sock.close()
        else:
            # Bind before lifespan migration: an occupied port must not cause
            # cleanup of an older service's files before failing to start.
            sock = config.bind_socket()
            try:
                server.run(sockets=[sock])
            finally:
                sock.close()
    except KeyboardInterrupt:
        pass
    if not server.started and not config.should_reload:
        raise SystemExit(3)


if __name__ == "__main__":  # pragma: no cover
    main()
