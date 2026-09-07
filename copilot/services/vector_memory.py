"""Local vector index for distilled trading memories."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Protocol, Sequence

from ai_trading_copilot.copilot.domain.models import DistilledMemory


VECTOR_INDEX_SCHEMA_VERSION = 1
DEFAULT_VECTOR_DIMENSIONS = 128


class TextEmbedder(Protocol):
    """Embeds retrieval text into a fixed-size vector."""

    dimensions: int
    name: str

    def embed(self, text: str) -> List[float]:
        ...


@dataclass(frozen=True)
class VectorSearchResult:
    """Vector search hit mapped back to the memory list ordinal."""

    ordinal: int
    memory_id: str
    score: float


class HashingTextEmbedder:
    """Deterministic dependency-free embedder for local development and tests."""

    name = "local_hashing_text_embedder"

    def __init__(self, dimensions: int = DEFAULT_VECTOR_DIMENSIONS):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = _embedding_tokens(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], byteorder="big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        return _normalize(vector)


class LocalVectorMemoryIndex:
    """Small persisted vector index for distilled memories.

    SQLite remains the source of truth. This rebuildable sidecar stores local
    similarity vectors and can be replaced without changing the store API.
    """

    def __init__(self, path: str | Path, *, embedder: TextEmbedder | None = None):
        self.path = Path(path)
        self.embedder = embedder or HashingTextEmbedder()

    def sync(self, memories: Iterable[DistilledMemory]) -> None:
        memory_list = list(memories)
        if not memory_list and not self.path.exists():
            return

        records = [
            {
                "ordinal": index,
                "memory_id": memory_fingerprint(memory),
                "text": memory_search_text(memory),
                "vector": self.embedder.embed(memory_search_text(memory)),
                "metadata": {
                    "memory_type": memory.memory_type.value,
                    "symbols": memory.symbols,
                    "tags": memory.tags,
                    "source_run_id": memory.source_run_id,
                    "source_path": memory.source_path,
                },
            }
            for index, memory in enumerate(memory_list)
        ]
        payload = {
            "schema_version": VECTOR_INDEX_SCHEMA_VERSION,
            "embedding_model": self.embedder.name,
            "dimensions": self.embedder.dimensions,
            "records": records,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)

    def search(
        self,
        query: str,
        *,
        memories: Iterable[DistilledMemory],
        limit: int,
    ) -> List[VectorSearchResult]:
        memory_list = list(memories)
        if limit <= 0 or not memory_list or not query.strip():
            return []

        self.sync(memory_list)
        query_vector = self.embedder.embed(query)
        if not any(query_vector):
            return []

        hits = []
        for record in self._load_records():
            score = _dot(query_vector, record["vector"])
            hits.append(
                VectorSearchResult(
                    ordinal=record["ordinal"],
                    memory_id=record["memory_id"],
                    score=max(score, 0.0),
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return [hit for hit in hits if hit.score > 0][:limit]

    def _load_records(self) -> List[dict]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != VECTOR_INDEX_SCHEMA_VERSION:
            return []
        if payload.get("dimensions") != self.embedder.dimensions:
            return []

        records = []
        for raw in payload.get("records", []):
            vector = raw.get("vector")
            ordinal = raw.get("ordinal")
            memory_id = raw.get("memory_id")
            if (
                isinstance(vector, list)
                and len(vector) == self.embedder.dimensions
                and isinstance(ordinal, int)
                and isinstance(memory_id, str)
            ):
                records.append(
                    {
                        "ordinal": ordinal,
                        "memory_id": memory_id,
                        "vector": [float(value) for value in vector],
                    }
                )
        return records


def default_vector_index_path(memory_path: str | Path) -> Path:
    """Return the rebuildable sidecar path used for a SQLite memory database."""

    return Path(memory_path).with_suffix(".vectors.json")


def memory_search_text(memory: DistilledMemory) -> str:
    """Build the text embedded into the vector index."""

    return " ".join(
        part
        for part in [
            memory.memory_type.value,
            memory.memory_kind.value,
            memory.lesson,
            memory.trigger,
            memory.rationale,
            " ".join(memory.symbols),
            " ".join(memory.tags),
            " ".join(memory.market_regimes),
            " ".join(memory.timeframes),
        ]
        if part
    )


def memory_fingerprint(memory: DistilledMemory) -> str:
    payload = json.dumps(memory.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def retrieval_query_text(
    *,
    symbol: str | None = None,
    tags: Iterable[str] = (),
    query: str | None = None,
) -> str:
    return " ".join(
        part
        for part in [
            query or "",
            symbol or "",
            " ".join(tag for tag in tags if tag),
        ]
        if part
    )


def _embedding_tokens(value: str) -> List[str]:
    tokens = [
        match.group(0).lower()
        for match in re.finditer(r"[A-Za-z0-9_]+", value)
        if len(match.group(0)) >= 2
    ]
    for match in re.finditer(r"[\u3400-\u9fff]+", value):
        run = match.group(0)
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def _normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(left_value * right_value for left_value, right_value in zip(left, right))
