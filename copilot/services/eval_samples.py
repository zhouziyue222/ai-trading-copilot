"""JSONL evaluation sample recording for RAGAS-style offline evals."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Iterator, Mapping

from ai_trading_copilot.copilot.domain.models import RagDocument
from ai_trading_copilot.copilot.services.tracing import sanitize_value


DEFAULT_EVAL_SAMPLES_FILE = "ragas_eval_samples.jsonl"

_current_recorder: ContextVar["EvalSampleRecorder | None"] = ContextVar(
    "copilot_eval_sample_recorder",
    default=None,
)


class EvalSampleRecorder:
    """Append-only recorder for samples that can be converted to RAGAS rows."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        run_id: str,
        file_name: str = DEFAULT_EVAL_SAMPLES_FILE,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / file_name
        self.run_id = run_id
        self.samples: list[dict[str, Any]] = []
        self._lock = RLock()
        self.path.write_text("", encoding="utf-8")

    @contextmanager
    def activate(self) -> Iterator["EvalSampleRecorder"]:
        token = _current_recorder.set(self)
        try:
            yield self
        finally:
            _current_recorder.reset(token)

    def record(self, *, stage: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        sample = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "stage": stage,
            **dict(payload),
        }
        sample = sanitize_value(sample, text_limit=4000)
        with self._lock:
            self.samples.append(sample)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
        return sample


def get_current_eval_sample_recorder() -> EvalSampleRecorder | None:
    return _current_recorder.get()


def record_eval_sample(*, stage: str, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    recorder = get_current_eval_sample_recorder()
    if recorder is None:
        return None
    return recorder.record(stage=stage, payload=payload)


def rag_documents_eval_payload(docs: Iterable[RagDocument]) -> dict[str, Any]:
    documents = list(docs)
    return {
        "retrieved_contexts": [doc.text for doc in documents],
        "retrieved_context_ids": [doc.id for doc in documents],
        "retrieval_results": [
            {
                "rank": index + 1,
                "id": doc.id,
                "title": doc.title,
                "source": doc.source,
                "source_type": doc.source_type,
                "score": doc.score,
                "symbols": list(doc.symbols),
                "tags": list(doc.tags),
                "channels": doc.metadata.get("retrieval_channels")
                or doc.metadata.get("retrieval_channel", ""),
                "query": doc.metadata.get("retrieval_query", ""),
            }
            for index, doc in enumerate(documents)
        ],
    }
