from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from ai_trading_copilot.copilot.services.tracing import TraceRecorder


def test_parallel_spans_preserve_context_parent_and_callback_can_snapshot(tmp_path):
    barrier = Barrier(3)
    callback_snapshots = []
    holder = {}

    def on_activity(event, span, recorder):
        # This intentionally re-enters serialization while the worker is
        # notifying. It would deadlock if callbacks ran under the recorder lock.
        holder["last"] = recorder.to_dict()
        callback_snapshots.append((event, span))

    recorder = TraceRecorder(
        output_dir=tmp_path,
        run_id="parallel",
        activity_callback=on_activity,
    )
    with recorder.activate():
        with recorder.start_span("root") as root:
            contexts = [contextvars.copy_context() for _ in range(2)]

            def worker(index):
                def run():
                    barrier.wait(timeout=5)
                    with recorder.start_span(f"analyst-{index}") as child:
                        child.set_attribute("worker", index)
                        child.add_event("ready", {"index": index})
                        barrier.wait(timeout=5)
                contexts[index].run(run)

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(worker, index) for index in range(2)]
                barrier.wait(timeout=5)
                barrier.wait(timeout=5)
                for future in futures:
                    future.result()

    payload = recorder.to_dict()
    children = {span["name"]: span for span in payload["spans"] if span["name"].startswith("analyst-")}
    assert set(children) == {"analyst-0", "analyst-1"}
    assert all(span["parent_span_id"] == root.span.span_id for span in children.values())
    assert all(span["status"] == "ok" for span in children.values())
    assert all(any(event["name"] == "ready" for event in span["events"]) for span in children.values())
    assert holder["last"]["spans"]

    # Callbacks receive independent shallow snapshots, not mutable registry
    # records that can change underneath a consumer.
    end_snapshots = [span for event, span in callback_snapshots if event == "end"]
    assert end_snapshots
    end_snapshots[0].attributes["changed"] = True
    assert "changed" not in recorder.to_dict()["spans"][0]["attributes"]


def test_concurrent_snapshots_are_consistent(tmp_path):
    recorder = TraceRecorder(output_dir=tmp_path, run_id="snapshots")
    with recorder.start_span("root") as root:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = []
            for index in range(20):
                futures.append(
                    pool.submit(
                        lambda i=index: (
                            root.set_attribute(f"a{i}", i),
                            root.add_event("tick", {"i": i}),
                            recorder.to_dict(),
                        )[-1]
                    )
                )
            payloads = [future.result() for future in futures]

    assert all(payload["spans"] for payload in payloads)
    final = recorder.to_dict()["spans"][0]
    assert len(final["events"]) == 20
    assert len([key for key in final["attributes"] if key.startswith("a")]) == 20

