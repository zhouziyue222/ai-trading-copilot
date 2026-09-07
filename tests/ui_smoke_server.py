# -*- coding: utf-8 -*-
"""Explicit offline UI smoke server using the main branch checkpoint engine."""
import argparse
import os
from pathlib import Path

import uvicorn
from ai_trading_copilot.copilot.ui.app import UISettings, create_app
from ai_trading_copilot.copilot.ui.server import LifecycleServer
from tests.lifecycle_fakes import BoundaryGraph


class SmokeGraph(BoundaryGraph):
    NODE_ORDER = ["Load Persona Markdown", "Trader"]

    def _node_map(self):
        return {self.NODE_ORDER[0]: self._tracked_node(self.NODE_ORDER[0], self.first),
                self.NODE_ORDER[1]: self._tracked_node(self.NODE_ORDER[1], self.second)}

    def run(self, **params):
        self.finish_second.clear()
        return super().run(**params)

    def resume(self, snapshot):
        self.finish_second.set()
        return super().resume(snapshot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8017)
    args = parser.parse_args()
    os.environ["COPILOT_INTERNAL_UI"] = "1"
    app = create_app(UISettings(reports_dir=args.root / "reports", subscriptions_file=args.root / "subscriptions.json",
                                memory_database=args.root / "memory.sqlite3", graph_cls=SmokeGraph,
                                long_term_memory_enabled=False))
    LifecycleServer(uvicorn.Config(app, host="127.0.0.1", port=args.port, use_colors=False)).run()


if __name__ == "__main__":
    main()
