"""Versioned memory repository backed by SQLite."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Protocol, Sequence

from ai_trading_copilot.copilot.domain import (
    DistilledMemory,
    MemoryEvaluation,
    MemoryStatus,
    RunOutcome,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryRepository(Protocol):
    """Persistence contract used by agents and evaluation services."""

    def upsert(
        self,
        memory: DistilledMemory,
        *,
        actor: str = "system",
        reason: str = "upsert",
    ) -> DistilledMemory: ...

    def get(self, memory_id: str) -> DistilledMemory | None: ...

    def list(
        self,
        *,
        statuses: Iterable[MemoryStatus | str] | None = None,
        symbol: str | None = None,
        include_expired: bool = False,
    ) -> list[DistilledMemory]: ...

    def transition(
        self,
        memory_id: str,
        status: MemoryStatus | str,
        *,
        actor: str,
        reason: str,
    ) -> DistilledMemory: ...

    def record_usage(
        self,
        *,
        run_id: str,
        symbol: str,
        memories: Sequence[DistilledMemory],
        mode: str,
    ) -> None: ...

    def record_outcome(self, outcome: RunOutcome) -> RunOutcome: ...


_ALLOWED_TRANSITIONS: dict[MemoryStatus, set[MemoryStatus]] = {
    MemoryStatus.CANDIDATE: {
        MemoryStatus.SHADOW,
        MemoryStatus.REJECTED,
        MemoryStatus.DEPRECATED,
    },
    MemoryStatus.SHADOW: {
        MemoryStatus.CANDIDATE,
        MemoryStatus.APPROVED,
        MemoryStatus.REJECTED,
        MemoryStatus.DEPRECATED,
    },
    MemoryStatus.APPROVED: {MemoryStatus.DEPRECATED},
    MemoryStatus.DEPRECATED: {
        MemoryStatus.CANDIDATE,
        MemoryStatus.APPROVED,
    },
    MemoryStatus.REJECTED: {MemoryStatus.CANDIDATE},
}


class SQLiteMemoryRepository:
    """Thread-safe-by-connection SQLite repository with append-only versions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    memory_kind TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    lesson TEXT NOT NULL,
                    symbols_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    valid_from TEXT,
                    valid_until TEXT,
                    version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_status
                    ON memories(status);
                CREATE INDEX IF NOT EXISTS idx_memories_updated
                    ON memories(updated_at DESC);

                CREATE TABLE IF NOT EXISTS memory_versions (
                    memory_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(memory_id, version)
                );

                CREATE TABLE IF NOT EXISTS memory_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    from_status TEXT,
                    to_status TEXT,
                    actor TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_usage (
                    run_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    memory_version INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(run_id, symbol, memory_id, memory_version, mode)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_usage_memory
                    ON memory_usage(memory_id, mode);

                CREATE TABLE IF NOT EXISTS run_outcomes (
                    run_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    horizon_days INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    evaluated_at TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY(run_id, symbol, horizon_days)
                );

                CREATE TABLE IF NOT EXISTS memory_evaluations (
                    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    memory_version INTEGER NOT NULL,
                    eligible INTEGER NOT NULL,
                    passed_safety_gate INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    evaluated_at TEXT NOT NULL
                );
                """
            )

    def upsert(
        self,
        memory: DistilledMemory,
        *,
        actor: str = "system",
        reason: str = "upsert",
    ) -> DistilledMemory:
        now = utc_now()
        memory_id = memory.memory_id.strip() or memory_fingerprint(memory)
        with self._connection() as connection:
            existing_row = connection.execute(
                "SELECT payload_json FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            existing = (
                DistilledMemory.model_validate_json(existing_row["payload_json"])
                if existing_row
                else None
            )
            if existing and _semantic_payload(existing) == _semantic_payload(memory):
                return existing

            version = existing.version + 1 if existing else max(memory.version, 1)
            created_at = existing.created_at if existing else (memory.created_at or now)
            evidence = list(memory.evidence_run_ids)
            if memory.source_run_id and memory.source_run_id.lower() not in evidence:
                evidence.append(memory.source_run_id.lower())
            normalized = memory.model_copy(
                update={
                    "memory_id": memory_id,
                    "version": version,
                    "created_at": created_at,
                    "updated_at": now,
                    "evidence_run_ids": sorted(set(evidence)),
                }
            )
            payload = normalized.model_dump_json()
            connection.execute(
                """
                INSERT INTO memories(
                    memory_id, status, memory_kind, scope, lesson,
                    symbols_json, tags_json, valid_from, valid_until, version,
                    payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    status=excluded.status,
                    memory_kind=excluded.memory_kind,
                    scope=excluded.scope,
                    lesson=excluded.lesson,
                    symbols_json=excluded.symbols_json,
                    tags_json=excluded.tags_json,
                    valid_from=excluded.valid_from,
                    valid_until=excluded.valid_until,
                    version=excluded.version,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    memory_id,
                    normalized.status.value,
                    normalized.memory_kind.value,
                    normalized.scope.value,
                    normalized.lesson,
                    json.dumps(normalized.symbols),
                    json.dumps(normalized.tags),
                    normalized.valid_from,
                    normalized.valid_until,
                    version,
                    payload,
                    created_at,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO memory_versions(
                    memory_id, version, payload_json, actor, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (memory_id, version, payload, actor, reason, now),
            )
            connection.execute(
                """
                INSERT INTO memory_events(
                    memory_id, from_status, to_status, actor, reason,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    existing.status.value if existing else None,
                    normalized.status.value,
                    actor,
                    reason,
                    payload,
                    now,
                ),
            )
        return normalized

    def get(self, memory_id: str) -> DistilledMemory | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload_json FROM memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        return DistilledMemory.model_validate_json(row["payload_json"]) if row else None

    def list(
        self,
        *,
        statuses: Iterable[MemoryStatus | str] | None = None,
        symbol: str | None = None,
        include_expired: bool = False,
    ) -> list[DistilledMemory]:
        clauses: list[str] = []
        parameters: list[str] = []
        if statuses is not None:
            status_values = [MemoryStatus(value).value for value in statuses]
            if not status_values:
                return []
            placeholders = ", ".join("?" for _ in status_values)
            clauses.append(f"status IN ({placeholders})")
            parameters.extend(status_values)
        query = "SELECT payload_json FROM memories"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at, memory_id"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        now = utc_now()
        requested_symbol = symbol.strip().upper() if symbol else None
        result: list[DistilledMemory] = []
        for row in rows:
            memory = DistilledMemory.model_validate_json(row["payload_json"])
            if requested_symbol and requested_symbol not in memory.symbols:
                continue
            if not include_expired:
                if memory.valid_from and memory.valid_from > now:
                    continue
                if memory.valid_until and memory.valid_until <= now:
                    continue
            result.append(memory)
        return result

    def transition(
        self,
        memory_id: str,
        status: MemoryStatus | str,
        *,
        actor: str,
        reason: str,
    ) -> DistilledMemory:
        target = MemoryStatus(status)
        memory = self.get(memory_id)
        if memory is None:
            raise KeyError(f"memory not found: {memory_id}")
        if target == memory.status:
            return memory
        if target not in _ALLOWED_TRANSITIONS[memory.status]:
            raise ValueError(
                f"invalid memory transition: {memory.status.value} -> {target.value}"
            )
        updates: dict[str, object] = {"status": target}
        if target == MemoryStatus.APPROVED:
            updates["approved_by"] = actor
            updates["last_validated_at"] = utc_now()
        return self.upsert(
            memory.model_copy(update=updates),
            actor=actor,
            reason=reason,
        )

    def versions(self, memory_id: str) -> list[DistilledMemory]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM memory_versions
                WHERE memory_id = ? ORDER BY version DESC
                """,
                (memory_id,),
            ).fetchall()
        return [DistilledMemory.model_validate_json(row["payload_json"]) for row in rows]

    def rollback(
        self,
        memory_id: str,
        version: int,
        *,
        actor: str,
        reason: str,
    ) -> DistilledMemory:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM memory_versions
                WHERE memory_id = ? AND version = ?
                """,
                (memory_id, version),
            ).fetchone()
        if row is None:
            raise KeyError(f"memory version not found: {memory_id}@{version}")
        historical = DistilledMemory.model_validate_json(row["payload_json"])
        return self.upsert(
            historical.model_copy(update={"version": 1}),
            actor=actor,
            reason=reason,
        )

    def record_usage(
        self,
        *,
        run_id: str,
        symbol: str,
        memories: Sequence[DistilledMemory],
        mode: str,
    ) -> None:
        if not run_id or not memories:
            return
        now = utc_now()
        with self._connection() as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO memory_usage(
                    run_id, symbol, memory_id, memory_version, mode, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        symbol.strip().upper(),
                        memory.memory_id,
                        memory.version,
                        mode,
                        now,
                    )
                    for memory in memories
                    if memory.memory_id
                ],
            )

    def usage_for_memory(self, memory_id: str, *, mode: str | None = None) -> list[dict]:
        query = "SELECT * FROM memory_usage WHERE memory_id = ?"
        parameters: list[str] = [memory_id]
        if mode:
            query += " AND mode = ?"
            parameters.append(mode)
        query += " ORDER BY created_at DESC"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def memory_ids_for_usage(self, *, run_id: str, symbol: str, mode: str) -> list[str]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT memory_id FROM memory_usage
                WHERE run_id = ? AND symbol = ? AND mode = ?
                ORDER BY memory_id
                """,
                (run_id, symbol.strip().upper(), mode),
            ).fetchall()
        return [str(row["memory_id"]) for row in rows]

    def record_outcome(self, outcome: RunOutcome) -> RunOutcome:
        payload = outcome.model_dump_json()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO run_outcomes(
                    run_id, symbol, horizon_days, payload_json,
                    evaluated_at, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, symbol, horizon_days) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    evaluated_at=excluded.evaluated_at,
                    recorded_at=excluded.recorded_at
                """,
                (
                    outcome.run_id,
                    outcome.symbol,
                    outcome.horizon_days,
                    payload,
                    outcome.evaluated_at,
                    outcome.recorded_at,
                ),
            )
        return outcome

    def outcomes_for_memory(self, memory_id: str, *, mode: str = "shadow") -> list[RunOutcome]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT o.payload_json
                FROM run_outcomes o
                JOIN memory_usage u
                  ON u.run_id = o.run_id AND u.symbol = o.symbol
                WHERE u.memory_id = ? AND u.mode = ?
                ORDER BY o.evaluated_at DESC
                """,
                (memory_id, mode),
            ).fetchall()
        return [RunOutcome.model_validate_json(row["payload_json"]) for row in rows]

    def save_evaluation(self, evaluation: MemoryEvaluation) -> MemoryEvaluation:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO memory_evaluations(
                    memory_id, memory_version, eligible, passed_safety_gate,
                    payload_json, evaluated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation.memory_id,
                    evaluation.memory_version,
                    int(evaluation.eligible),
                    int(evaluation.passed_safety_gate),
                    evaluation.model_dump_json(),
                    evaluation.evaluated_at,
                ),
            )
        return evaluation

    def latest_evaluation(self, memory_id: str) -> MemoryEvaluation | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM memory_evaluations
                WHERE memory_id = ? ORDER BY evaluation_id DESC LIMIT 1
                """,
                (memory_id,),
            ).fetchone()
        return MemoryEvaluation.model_validate_json(row["payload_json"]) if row else None

    def counts_by_status(self) -> dict[str, int]:
        counts = {status.value: 0 for status in MemoryStatus}
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM memories GROUP BY status"
            ).fetchall()
        for row in rows:
            counts[row["status"]] = row["count"]
        return counts


def memory_fingerprint(memory: DistilledMemory) -> str:
    """Stable ID used for exact candidate deduplication."""

    identity = {
        "memory_type": memory.memory_type.value,
        "memory_kind": memory.memory_kind.value,
        "scope": memory.scope.value,
        "validation_target": memory.validation_target.value,
        "lesson": " ".join(memory.lesson.lower().split()),
        "symbols": sorted(memory.symbols),
        "trigger": " ".join(memory.trigger.lower().split()),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"mem_{digest[:24]}"


def _semantic_payload(memory: DistilledMemory) -> dict:
    payload = memory.model_dump(mode="json")
    for key in ("memory_id", "version", "created_at", "updated_at"):
        payload.pop(key, None)
    return payload


__all__ = [
    "MemoryRepository",
    "SQLiteMemoryRepository",
    "memory_fingerprint",
    "utc_now",
]
