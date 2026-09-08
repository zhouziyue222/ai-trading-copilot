"""Run with python -m tests.benchmark_parallel; no network or production data."""
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from ai_trading_copilot.copilot.services.run_tracker import RunTracker
from tests.parallel_fakes import OfflineParallelGraph as Graph


def main():
    measurements = {}
    with TemporaryDirectory(prefix="copilot-parallel-") as temporary:
        for parallel in (False, True):
            elapsed, stages = [], []
            for iteration in range(3):
                Graph.reset()
                Graph.delay = 0.25
                for event in Graph.releases.values():
                    event.set()
                root = Path(temporary) / f"{parallel}-{iteration}"
                tracker = RunTracker(output_dir=root, run_id=root.name, symbols=["AAPL"],
                                     params={}, defaults={}, nodes=Graph.NODE_ORDER)
                graph = Graph(run_tracker=tracker, force_sequential=True, parallel_analysts=parallel)
                started = perf_counter()
                graph.run(subscription_symbols=["AAPL"], selected_analysts=[name.lower().replace(" ", "_") for name in Graph.names],
                          report_output_dir=str(root), run_id=root.name)
                elapsed.append(perf_counter() - started)
                from datetime import datetime
                nodes = [tracker.status["nodes"][name] for name in Graph.names]
                stages.append((max(datetime.fromisoformat(n["finished_at"]) for n in nodes)
                               - min(datetime.fromisoformat(n["started_at"]) for n in nodes)).total_seconds())
            measurements["parallel" if parallel else "sequential"] = {
                "total_seconds": round(median(elapsed), 3), "analyst_seconds": round(median(stages), 3)}
    print(json.dumps(measurements, indent=2))


if __name__ == "__main__":
    main()
