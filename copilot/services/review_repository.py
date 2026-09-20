"""Durable delayed-review records in the memory database (no report files)."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from ai_trading_copilot.copilot.domain.models import normalize_symbol
from ai_trading_copilot.copilot.adapters.stock_info import normalize_futu_code_to_symbol, normalize_futu_symbol


def canonical_symbol(value):
    return normalize_futu_code_to_symbol(normalize_futu_symbol(value))
from ai_trading_copilot.copilot.services.memory_repository import SQLiteMemoryRepository


def timestamp(value=None):
    value = value or datetime.now(timezone.utc)
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return value.astimezone(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class ReviewFill(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    environment: Literal["SIMULATE", "REAL"]
    account_id: str = Field(min_length=1)
    deal_id: str = Field(min_length=1)
    order_id: str = Field(min_length=1)
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    executed_at: str
    fees: float | None = Field(default=None, ge=0)

    @field_validator("account_id", mode="before")
    @classmethod
    def account_string(cls, value):
        return str(value)

    @field_validator("side", mode="before")
    @classmethod
    def side_alias(cls, value):
        return {"SELL_SHORT": "SELL", "BUY_BACK": "BUY"}.get(value, value)

    @field_validator("executed_at")
    @classmethod
    def valid_time(cls, value):
        return timestamp(value)

    @field_validator("symbol")
    @classmethod
    def valid_symbol(cls, value):
        value = normalize_symbol(normalize_futu_code_to_symbol(value))
        if not value:
            raise ValueError("symbol required")
        return value


class ReviewObservation(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="allow")
    category: Literal["hypothetical", "SIMULATE", "REAL"]
    account_id: str | None = None
    eligible: bool = False
    realized_return: float | None = None
    benchmark_return: float | None = None
    max_drawdown: float | None = None


class ReviewOrder(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    environment: Literal["SIMULATE", "REAL"]
    account_id: str
    order_id: str = Field(min_length=1)
    symbol: str
    side: str
    quantity: float = Field(gt=0)
    price: float = Field(ge=0)
    filled_quantity: float = Field(ge=0)
    status: str
    created_at: str
    updated_at: str | None = Field(default=None, validation_alias=AliasChoices("updated_at", "order_updated_at"))

    @field_validator("account_id", mode="before")
    @classmethod
    def account_string(cls, value):
        return str(value)

    @field_validator("symbol")
    @classmethod
    def symbol_string(cls, value):
        return normalize_symbol(normalize_futu_code_to_symbol(value))

    @field_validator("created_at", "updated_at")
    @classmethod
    def order_time(cls, value):
        return timestamp(value) if value else None


class ReviewResult(BaseModel):
    model_config = ConfigDict(extra="allow")
    run_id: str
    symbol: str
    horizon_days: int = Field(gt=0)
    cutoff_at: str
    observations: list[ReviewObservation]


class ReviewRepository:
    def __init__(self, memory: SQLiteMemoryRepository):
        self.memory = memory
        with memory._connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS review_snapshots (
                    run_id TEXT NOT NULL, symbol TEXT NOT NULL,
                    generated_at TEXT NOT NULL, payload_json TEXT NOT NULL,
                    PRIMARY KEY(run_id,symbol));
                CREATE TABLE IF NOT EXISTS review_tasks (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, symbol TEXT NOT NULL,
                    horizon_days INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                    reason TEXT NOT NULL DEFAULT '', next_check_at TEXT NOT NULL,
                    lease_token TEXT, lease_until TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(run_id,symbol,horizon_days));
                CREATE INDEX IF NOT EXISTS review_due ON review_tasks(status,next_check_at);
                CREATE TABLE IF NOT EXISTS review_results (
                    review_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS review_result_history (
                    id INTEGER PRIMARY KEY, review_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL, recorded_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS review_fills (
                    environment TEXT NOT NULL, account_id TEXT NOT NULL, deal_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL, run_id TEXT, symbol TEXT,
                    linked_at TEXT, PRIMARY KEY(environment,account_id,deal_id));
                CREATE TABLE IF NOT EXISTS review_orders (
                    environment TEXT NOT NULL, account_id TEXT NOT NULL, order_id TEXT NOT NULL,
                    run_id TEXT NOT NULL, symbol TEXT NOT NULL,
                    PRIMARY KEY(environment,account_id,order_id,run_id,symbol));
                CREATE TABLE IF NOT EXISTS review_broker_orders (
                    id INTEGER PRIMARY KEY, environment TEXT NOT NULL, account_id TEXT NOT NULL,
                    order_id TEXT NOT NULL, payload_json TEXT NOT NULL, observed_at TEXT NOT NULL,
                    UNIQUE(environment,account_id,order_id,payload_json));
                CREATE TABLE IF NOT EXISTS review_fill_events (
                    id INTEGER PRIMARY KEY, environment TEXT, account_id TEXT, deal_id TEXT,
                    run_id TEXT, symbol TEXT, recorded_at TEXT NOT NULL);
            """)

    def register(self, snapshot, horizons):
        run_id, symbol = snapshot["run_id"], snapshot["symbol"]
        count = 0
        with self.memory.transaction() as db:
            old = db.execute("SELECT payload_json FROM review_snapshots WHERE run_id=? AND symbol=?",
                             (run_id, symbol)).fetchone()
            if old:
                # The first completed snapshot is immutable, including its timestamp.
                return 0
            db.execute("INSERT INTO review_snapshots VALUES (?,?,?,?)",
                       (run_id, symbol, snapshot["generated_at"], encode(snapshot)))
            for horizon in horizons:
                key = hashlib.sha256(encode([run_id, symbol, horizon]).encode()).hexdigest()[:32]
                db.execute("INSERT INTO review_tasks(id,run_id,symbol,horizon_days,next_check_at) VALUES (?,?,?,?,?)",
                           (key, run_id, symbol, horizon, snapshot["generated_at"]))
                count += 1
        return count

    def list_reviews(self):
        with self.memory._connection() as db:
            return [dict(row) for row in db.execute(
                "SELECT id,run_id,symbol,horizon_days,status,reason,next_check_at,attempts FROM review_tasks ORDER BY next_check_at,id")]

    def detail(self, review_id):
        with self.memory._connection() as db:
            task = db.execute("SELECT * FROM review_tasks WHERE id=?", (review_id,)).fetchone()
            if not task:
                return None
            snapshot = db.execute("SELECT payload_json FROM review_snapshots WHERE run_id=? AND symbol=?",
                                  (task["run_id"], task["symbol"])).fetchone()
            result = db.execute("SELECT payload_json FROM review_results WHERE review_id=?", (review_id,)).fetchone()
            history = [json.loads(row[0]) for row in db.execute(
                "SELECT payload_json FROM review_result_history WHERE review_id=? ORDER BY id", (review_id,))]
            return {"task": dict(task), "snapshot": json.loads(snapshot[0]),
                    "result": json.loads(result[0]) if result else None, "previous_results": history}

    def claim(self, review_id, now, lease_seconds=900):
        now = timestamp(now)
        token = uuid.uuid4().hex
        until = timestamp(datetime.fromisoformat(now) + timedelta(seconds=lease_seconds))
        with self.memory.transaction() as db:
            changed = db.execute("""UPDATE review_tasks SET lease_token=?,lease_until=?,attempts=attempts+1
                WHERE id=? AND status!='completed' AND next_check_at<=?
                AND (lease_until IS NULL OR lease_until<=?)""", (token, until, review_id, now, now)).rowcount
        return token if changed else None

    def check_lease(self, db, review_id, token):
        row = db.execute("SELECT lease_token FROM review_tasks WHERE id=?", (review_id,)).fetchone()
        if not row or row[0] != token:
            raise RuntimeError("review lease lost")

    def save_result(self, review_id, token, result):
        payload = ReviewResult.model_validate(result).model_dump(mode="json")
        with self.memory.transaction() as db:
            self.check_lease(db, review_id, token)
            db.execute("INSERT INTO review_results VALUES (?,?) ON CONFLICT(review_id) DO UPDATE SET payload_json=excluded.payload_json",
                       (review_id, encode(payload)))

    def release(self, review_id, token, *, status, reason="", next_check_at=None):
        with self.memory.transaction() as db:
            self.check_lease(db, review_id, token)
            db.execute("UPDATE review_tasks SET status=?,reason=?,next_check_at=?,lease_token=NULL,lease_until=NULL WHERE id=?",
                       (status, reason[:2000], timestamp(next_check_at), review_id))

    def retry(self, review_id):
        with self.memory.transaction() as db:
            task = db.execute("SELECT * FROM review_tasks WHERE id=?", (review_id,)).fetchone()
            if not task or (task["lease_until"] and task["lease_until"] > timestamp()):
                return False
            if task["status"] == "completed":
                db.execute("INSERT INTO review_result_history(review_id,payload_json,recorded_at) SELECT review_id,payload_json,? FROM review_results WHERE review_id=?",
                           (timestamp(), review_id))
                row = db.execute("SELECT payload_json FROM review_results WHERE review_id=?", (review_id,)).fetchone()
                if row:
                    result = json.loads(row[0])
                    for key in ("reflection", "memories", "reflection_skipped", "evaluations"):
                        result.pop(key, None)
                    result["execution_complete"] = False
                    db.execute("UPDATE review_results SET payload_json=? WHERE review_id=?", (encode(result), review_id))
            db.execute("UPDATE review_tasks SET status='pending',next_check_at=?,reason='manual retry',lease_token=NULL,lease_until=NULL WHERE id=?",
                       (timestamp(), review_id))
            return True

    def record_order(self, environment, account_id, order_id, run_id, symbol):
        if environment not in {"SIMULATE", "REAL"} or not account_id or not order_id:
            raise ValueError("explicit account environment, ID and order ID required")
        with self.memory.transaction() as db:
            if not db.execute("SELECT 1 FROM review_snapshots WHERE run_id=? AND symbol=?", (run_id, symbol)).fetchone():
                raise ValueError("plan snapshot not found")
            db.execute("INSERT OR IGNORE INTO review_orders VALUES (?,?,?,?,?)",
                       (environment, str(account_id), str(order_id), run_id, symbol))

    def save_fills(self, fills):
        with self.memory.transaction() as db:
            for raw in fills:
                fill = ReviewFill.model_validate(raw)
                key = (fill.environment, fill.account_id, fill.deal_id)
                prior = db.execute("SELECT payload_json FROM review_fills WHERE environment=? AND account_id=? AND deal_id=?", key).fetchone()
                if prior and json.loads(prior[0]) != fill.model_dump(mode="json"):
                    raise ValueError("conflicting broker fill; original retained")
                db.execute("INSERT OR IGNORE INTO review_fills(environment,account_id,deal_id,payload_json) VALUES (?,?,?,?)",
                           (*key, fill.model_dump_json()))
                linked = db.execute("SELECT run_id FROM review_fills WHERE environment=? AND account_id=? AND deal_id=?", key).fetchone()
                matches = db.execute("SELECT run_id,symbol FROM review_orders WHERE environment=? AND account_id=? AND order_id=?",
                                     (fill.environment, fill.account_id, fill.order_id)).fetchall()
                matches = [m for m in matches if canonical_symbol(m["symbol"]) == canonical_symbol(fill.symbol)]
                if linked[0] is None and len(matches) == 1:
                    self.associate_fill(*key, matches[0][0], matches[0][1])

    def save_orders(self, orders):
        with self.memory.transaction() as db:
            for raw in orders:
                order = ReviewOrder.model_validate(raw)
                if order.filled_quantity > order.quantity:
                    raise ValueError("order filled quantity exceeds quantity")
                db.execute("INSERT OR IGNORE INTO review_broker_orders(environment,account_id,order_id,payload_json,observed_at) VALUES (?,?,?,?,?)",
                           (order.environment, order.account_id, order.order_id, order.model_dump_json(), timestamp()))

    def orders_for_review(self, environment, account_id, run_id, symbol, cutoff):
        linked_ids = {fill["order_id"] for fill in self.list_fills() if fill["environment"] == environment
                      and fill["account_id"] == str(account_id) and fill["run_id"] == run_id and fill["linked_symbol"] == symbol}
        with self.memory._connection() as db:
            mappings = db.execute("SELECT order_id,run_id,symbol FROM review_orders WHERE environment=? AND account_id=?", (environment, str(account_id))).fetchall()
            for order_id in {row["order_id"] for row in mappings}:
                targets = {(row["run_id"], row["symbol"]) for row in mappings if row["order_id"] == order_id}
                if targets == {(run_id, symbol)}:
                    linked_ids.add(order_id)
            rows = db.execute("SELECT payload_json FROM review_broker_orders WHERE environment=? AND account_id=? ORDER BY id DESC", (environment, str(account_id))).fetchall()
        known, unknown = {}, {}
        for row in rows:
            order = json.loads(row[0])
            identifier = order["order_id"]
            if identifier not in linked_ids or canonical_symbol(order["symbol"]) != canonical_symbol(symbol) or order["created_at"] > cutoff:
                continue
            if order.get("updated_at") and order["updated_at"] <= cutoff:
                known.setdefault(identifier, order)
            else:
                unknown[identifier] = {key: order[key] for key in ("environment", "account_id", "order_id", "symbol", "created_at")}
                unknown[identifier]["status"] = "unknown_at_cutoff"
        return [known.get(key, unknown.get(key)) for key in sorted(set(known) | set(unknown))]

    def list_fills(self):
        with self.memory._connection() as db:
            return [{**json.loads(row["payload_json"]), "run_id": row["run_id"],
                     "linked_symbol": row["symbol"], "linked_at": row["linked_at"]}
                    for row in db.execute("SELECT * FROM review_fills ORDER BY environment,account_id,deal_id")]

    def associate_fill(self, environment, account_id, deal_id, run_id, symbol):
        key = (environment, str(account_id), str(deal_id))
        with self.memory.transaction() as db:
            row = db.execute("SELECT * FROM review_fills WHERE environment=? AND account_id=? AND deal_id=?", key).fetchone()
            if not row:
                raise ValueError("fill not found")
            fill = json.loads(row["payload_json"])
            snap = db.execute("SELECT generated_at FROM review_snapshots WHERE run_id=? AND symbol=?", (run_id, symbol)).fetchone()
            if not snap or canonical_symbol(fill["symbol"]) != canonical_symbol(symbol) or fill["executed_at"] < snap[0]:
                raise ValueError("fill must match symbol and follow the original plan")
            if row["run_id"] is not None and (row["run_id"], row["symbol"]) != (run_id, symbol):
                raise ValueError("fill is already associated; cannot silently reassign")
            if row["run_id"] is None:
                db.execute("UPDATE review_fills SET run_id=?,symbol=?,linked_at=? WHERE environment=? AND account_id=? AND deal_id=?",
                           (run_id, symbol, timestamp(), *key))
                db.execute("INSERT INTO review_fill_events(environment,account_id,deal_id,run_id,symbol,recorded_at) VALUES (?,?,?,?,?,?)",
                           (*key, run_id, symbol, timestamp()))
