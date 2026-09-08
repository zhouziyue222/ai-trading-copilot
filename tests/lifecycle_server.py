# -*- coding: utf-8 -*-
"""Offline test server; test controls are never registered in production."""

import signal
import subprocess
import sys
from pathlib import Path
from threading import Timer

import uvicorn

from ai_trading_copilot.copilot.ui.app import UISettings, create_app
from ai_trading_copilot.copilot.ui.server import LifecycleServer
from tests.lifecycle_fakes import BoundaryGraph


def main():
    root, port = Path(sys.argv[1]), int(sys.argv[2])
    graph_cls = BoundaryGraph
    if len(sys.argv) > 3 and sys.argv[3] == "parallel":
        from tests.parallel_fakes import OfflineParallelGraph
        graph_cls = OfflineParallelGraph
        graph_cls.reset()
    BoundaryGraph.finish_second.clear()
    app = create_app(UISettings(reports_dir=root / "reports", subscriptions_file=root / "subscriptions.json",
                                 memory_database=root / "memory.sqlite3", graph_cls=graph_cls,
                                 long_term_memory_enabled=False))
    children = []

    @app.post("/test/finish")
    def finish():
        BoundaryGraph.finish_second.set()
        for event in getattr(graph_cls, "releases", {}).values():
            event.set()
        return {"ok": True}

    @app.post("/test/release/{name}")
    def release(name: str):
        graph_cls.releases[name].set()
        return {"ok": True}

    @app.post("/test/sigint")
    def sigint():
        Timer(0.1, lambda: signal.raise_signal(signal.SIGINT)).start()
        return {"ok": True}

    @app.post("/test/child")
    def child():
        process = subprocess.Popen([sys.executable, "-X", "utf8", "-c", "import time; time.sleep(300)"],
                                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        children.append(process)
        return {"pid": process.pid}

    @app.get("/test/calls")
    def calls():
        return {"calls": graph_cls.calls}

    @app.get("/test/failure")
    def failure():
        raise RuntimeError("test exception")

    try:
        LifecycleServer(uvicorn.Config(app, host="127.0.0.1", port=port, use_colors=False)).run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
