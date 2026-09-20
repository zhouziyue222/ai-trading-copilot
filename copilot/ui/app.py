# -*- coding: utf-8 -*-
"""FastAPI application for the local AI trading copilot dashboard."""

from __future__ import annotations

import json
import os
import re
import socket
import secrets
import time
from contextvars import copy_context
import subprocess
import sys
from dataclasses import dataclass
from contextlib import asynccontextmanager
from copy import deepcopy
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Callable, Dict, Iterable, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from ai_trading_copilot.copilot.adapters.futu_execution import FutuSimulatedExecutionAdapter
from ai_trading_copilot.copilot.adapters.portfolio import get_futu_portfolio_snapshot
from ai_trading_copilot.copilot.adapters.stock_info import get_futu_stock_info
from ai_trading_copilot.copilot.config import DEFAULT_REPORT_OUTPUT_DIR
from ai_trading_copilot.copilot.config.defaults import DEFAULT_PERSONA_CONFIG
from ai_trading_copilot.copilot.config.llm import (
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    load_copilot_env,
)
from ai_trading_copilot.copilot.domain.enums import (
    AnalystType,
    ExecutionMode,
    MarketType,
    MemoryStatus,
    SubscriptionStatus,
)
from ai_trading_copilot.copilot.domain.models import (
    BrokerExecutionRequest,
    BrokerExecutionResult,
    RunOutcome,
    Subscription,
    SubscriptionBook,
    UserPersonaConfig,
    normalize_symbol,
)
from ai_trading_copilot.copilot.graph import CopilotLangGraph, RunCancelled
from ai_trading_copilot.copilot.run import (
    DEFAULT_ANALYSTS,
    DEFAULT_MEMORY_DATABASE,
    DEFAULT_RISK_POSITION_FILE,
    DEFAULT_SUBSCRIPTIONS_FILE,
    accepts_keyword,
    create_default_fundamental_research_retriever,
    create_default_memory_agent,
    resolve_long_term_memory_enabled,
)
from ai_trading_copilot.copilot.services.run_tracker import (
    NODE_CANCELLED,
    NODE_DEGRADED,
    NODE_FAILED,
    NODE_SUCCEEDED,
    RunTracker,
    write_json_atomic,
)
from ai_trading_copilot.copilot.services.run_lifecycle import (
    RunControl, RunLifecycle, checked_path, remove_owned,
    safe_id,
)
from ai_trading_copilot.copilot.services.rag_store import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
)
from ai_trading_copilot.copilot.services.tracing import get_observability_status, configure_otel, start_http_span
from ai_trading_copilot.copilot.services.diagnostics import CONTEXT, DiagnosticStore, record, exception_data
from ai_trading_copilot.copilot.services.subscription_service import SubscriptionStore
from ai_trading_copilot.copilot.services.risk_position_store import RiskPositionStore
from ai_trading_copilot.copilot.review import create_review_service


SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class UISettings:
    subscriptions_file: Path = DEFAULT_SUBSCRIPTIONS_FILE
    memory_database: Path = DEFAULT_MEMORY_DATABASE
    risk_position_file: Path = DEFAULT_RISK_POSITION_FILE
    rag_chroma_dir: Optional[Path] = None
    reports_dir: Path = DEFAULT_REPORT_OUTPUT_DIR
    graph_cls: type = CopilotLangGraph
    tracker_cls: type = RunTracker
    run_in_background: bool = True
    fundamental_retriever_factory: Optional[Callable[["UISettings", bool], object]] = None
    simulated_broker_factory: Optional[Callable[[], object]] = None
    latest_price_getter: Optional[Callable[[str], float]] = None
    portfolio_snapshot_getter: Optional[Callable[[ExecutionMode], object]] = None
    portfolio_snapshot_timeout_seconds: float = 5.0
    long_term_memory_enabled: Optional[bool] = None


class ReviewAssociationRequest(BaseModel):
    environment: str = Field(min_length=1, max_length=32)
    account_id: str = Field(min_length=1, max_length=128)
    deal_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    symbol: str = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def nonblank(self):
        for name in type(self).model_fields:
            value = getattr(self, name).strip()
            if not value:
                raise ValueError(f"{name} must not be blank")
            setattr(self, name, value)
        return self


class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


class SubscriptionCreateRequest(BaseModel):
    symbol: str
    market_type: MarketType = MarketType.US_STOCK
    reason: str = ""
    target_action: str = ""
    status: SubscriptionStatus = SubscriptionStatus.OBSERVING


class SubscriptionUpdateRequest(BaseModel):
    status: Optional[SubscriptionStatus] = None
    reason: Optional[str] = None
    target_action: Optional[str] = None


class RunCreateRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    manual_symbols: str = ""
    selected_analysts: List[AnalystType] = Field(
        default_factory=lambda: list(DEFAULT_ANALYSTS)
    )
    trade_date: Optional[str] = None
    look_back_days: int = Field(default=90, ge=1, le=1000)
    portfolio_mode: ExecutionMode = ExecutionMode.SIMULATION
    long_term_memory_enabled: Optional[bool] = None


class RagTextIngestRequest(BaseModel):
    title: str = "Manual fundamental note"
    text: str
    source_type: str = "fundamental_research_note"
    symbols: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=lambda: ["manual", "fundamentals", "fundamental_research_note"])


class RagOnlineIngestRequest(BaseModel):
    symbols: List[str] = Field(default_factory=list)
    manual_symbols: str = ""
    trade_date: Optional[str] = None
    look_back_days: int = Field(default=7, ge=1, le=60)


class MemoryTransitionRequest(BaseModel):
    status: MemoryStatus
    actor: str = Field(default="local_user", min_length=1, max_length=100)
    reason: str = Field(default="manual_review", min_length=1, max_length=500)


class MemoryRollbackRequest(BaseModel):
    version: int = Field(ge=1)
    actor: str = Field(default="local_user", min_length=1, max_length=100)
    reason: str = Field(default="manual_rollback", min_length=1, max_length=500)


class RiskPositionUpdateRequest(BaseModel):
    persona: UserPersonaConfig
    target_weights: Dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_target_weights(self) -> "RiskPositionUpdateRequest":
        normalized: Dict[str, float] = {}
        for raw_symbol, raw_weight in self.target_weights.items():
            symbol = normalize_symbol(raw_symbol)
            if not symbol:
                raise ValueError("target weight symbol is required")
            weight = float(raw_weight)
            if weight != weight or weight < -1 or weight > 1:
                raise ValueError(f"target weight for {symbol} must be in [-1, 1]")
            normalized[symbol] = round(weight, 6)
        return self.model_copy(update={"target_weights": normalized})


def create_app(settings: UISettings | None = None) -> FastAPI:
    resolved = settings or UISettings(reports_dir=Path(os.getenv("COPILOT_UI_REPORTS_DIR") or DEFAULT_REPORT_OUTPUT_DIR))
    static_dir = Path(__file__).resolve().parent / "static"
    runtime = RunLifecycle(resolved.reports_dir)
    diagnostics = DiagnosticStore(resolved.reports_dir / ".diagnostics")

    @asynccontextmanager
    async def lifespan(app):
        runtime.start()
        configure_otel()
        review_stop = Event()
        def review_worker():
            while not review_stop.is_set():
                try:
                    create_review_service(resolved.memory_database).run_due(stop_event=review_stop)
                except Exception as exc:
                    record("review.scheduler.failed", level="ERROR", **exception_data(exc))
                if review_stop.wait(3600):
                    break
        review_thread = Thread(target=review_worker, name="delayed-reviews", daemon=True)
        review_thread.start()
        try:
            yield
        finally:
            # In-flight provider calls finish under their leases; never block shutdown.
            review_stop.set()
            runtime.shutdown()
            diagnostics.close()

    app = FastAPI(title="AI Trading Copilot UI", lifespan=lifespan,
                  default_response_class=UTF8JSONResponse)
    app.state.ui_settings = resolved
    app.state.runtime = runtime
    app.state.run_controls = runtime.controls
    app.state.run_controls_lock = runtime.lock
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/reviews")
    def reviews_page():
        return FileResponse(static_dir / "reviews.html")

    @app.get("/api/reviews")
    def reviews_list():
        return {"items": create_review_service(resolved.memory_database).repository.list_reviews()}

    @app.post("/api/reviews/run-due")
    def reviews_run_due():
        return create_review_service(resolved.memory_database).run_due()

    @app.get("/api/reviews/{review_id}")
    def review_detail(review_id: str):
        result = create_review_service(resolved.memory_database).repository.detail(review_id)
        if result is None:
            raise HTTPException(404, "Review not found")
        return result

    @app.post("/api/reviews/{review_id}/retry")
    def review_retry(review_id: str):
        repository = create_review_service(resolved.memory_database).repository
        if repository.detail(review_id) is None:
            raise HTTPException(404, "Review not found")
        try:
            if not repository.retry(review_id):
                raise HTTPException(409, "Review cannot be retried in its current state")
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"retried": True}

    @app.get("/api/review-fills")
    def review_fills():
        return {"items": create_review_service(resolved.memory_database).repository.list_fills()}

    @app.post("/api/review-fills/associate")
    def review_associate(body: ReviewAssociationRequest):
        try:
            create_review_service(resolved.memory_database).repository.associate_fill(
                body.environment, body.account_id, body.deal_id, body.run_id, body.symbol)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"associated": True}

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = secrets.token_hex(16)
        incoming = request.headers.get("traceparent", "")
        parts = incoming.split("-")
        valid = bool(re.fullmatch(r"00-[0-9a-f]{32}-[0-9a-f]{16}-0[01]", incoming)) and int(parts[1], 16) != 0 and int(parts[2], 16) != 0
        trace_id = parts[1] if valid else secrets.token_hex(16)
        http_span = start_http_span(incoming if valid else "") if request.method != "GET" else None
        span_id = secrets.token_hex(8)
        if http_span is not None:
            sdk_context = http_span.get_span_context()
            trace_id, span_id = format(sdk_context.trace_id, "032x"), format(sdk_context.span_id, "016x")
        token = CONTEXT.set({"request_id": request_id, "trace_id": trace_id, "span_id": span_id, "trace_flags": parts[3] if valid else "01",
                             "session_id": runtime.session_id, "store": diagnostics})
        started = time.perf_counter()
        if request.method != "GET":
            record("span.start", name="http.server", parent_span_id=parts[2] if valid else None)
        try:
            try:
                response = await call_next(request)
            except Exception as exc:
                record("request.failed", level="ERROR", **exception_data(exc))
                response = UTF8JSONResponse({"detail": "请求失败，请复制故障编号联系开发人员。", "request_id": request_id}, status_code=500)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Session-ID"] = runtime.session_id
            response.headers["Cache-Control"] = "no-store"
            if request.method != "GET" or response.status_code >= 400:
                route = getattr(request.scope.get("route"), "path", "unmatched")
                if request.method != "GET":
                    record("span.end", name="http.server", outcome="error" if response.status_code >= 500 else "ok",
                           duration_ms=round((time.perf_counter()-started)*1000, 3))
                if http_span is not None:
                    http_span.set_attribute("http.route", route)
                    http_span.set_attribute("http.response.status_code", response.status_code)
                    if response.status_code >= 500:
                        from opentelemetry.trace import Status, StatusCode
                        http_span.set_status(Status(StatusCode.ERROR))
                record("request.end", method=request.method, route=route, status_code=response.status_code,
                       duration_ms=round((time.perf_counter()-started)*1000, 3))
            return response
        finally:
            if http_span is not None:
                http_span.end()
            CONTEXT.reset(token)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        return UTF8JSONResponse({"detail": exc.detail, "request_id": CONTEXT.get().get("request_id")}, status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return UTF8JSONResponse({"detail": jsonable_encoder(exc.errors()), "request_id": CONTEXT.get().get("request_id")}, status_code=422)

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        return UTF8JSONResponse({"detail": "Internal Server Error"}, status_code=500)

    def run_dir_for(run_id: str) -> Path:
        try:
            return runtime.locate(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def locked_run_io(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            with runtime.lock:
                return function(*args, **kwargs)
        return wrapper

    def run_response(run_id: str) -> dict:
        with runtime.lock:
            directory = run_dir_for(run_id)
            control = runtime.controls.get(run_id)
            if control is not None:
                with control.tracker._lock:
                    status = deepcopy(control.tracker.status)
                payload = {"run_id": run_id, "output_dir": str(directory), "status": status,
                           "reports": status.get("reports", {}), "errors": status.get("errors", [])}
            else:
                payload = _run_response(run_id, directory)
            if not payload["status"]:
                raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
            if payload["status"].get("status") == "running" and run_id not in runtime.controls:
                payload["status"]["status"] = "interrupted"
                payload["status"]["cancel_requested"] = False
            if payload["status"].get("status") == "cancelled":
                checkpoint_id = payload["status"].get("checkpoint_id") or ""
                payload["status"]["resumable"] = bool(
                    safe_id(checkpoint_id) and
                    checked_path(runtime.root, runtime.checkpoints_dir / f"{checkpoint_id}.json").is_file())
            payload["session_id"] = runtime.session_id
            return payload

    @app.get("/api/runtime")
    def get_runtime() -> dict:
        with runtime.lock:
            status = runtime.runtime_status()
            for run in status["active_runs"]:
                run.update(worker_alive=True, heartbeat_at=datetime.now().astimezone().isoformat(),
                           last_progress_at=runtime.controls[run["run_id"]].tracker.status["updated_at"])
            return status

    def require_internal(request):
        if os.getenv("COPILOT_INTERNAL_UI", "").lower() not in {"1", "true"} or request.client.host not in {"127.0.0.1", "::1", "testclient"}:
            raise HTTPException(404, "Not found")

    @app.get("/internal/observability")
    def internal_page(request: Request):
        require_internal(request)
        return FileResponse(static_dir / "internal.html")

    @app.get("/api/internal/observability")
    def internal_data(request: Request, query: str = ""):
        require_internal(request)
        return {**diagnostics.query(query), "runtime": get_runtime(), "export": get_observability_status()}

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/subscriptions")
    def list_subscriptions() -> dict:
        book = _store(resolved).load()
        return {
            "path": str(resolved.subscriptions_file),
            "items": book.model_dump(mode="json")["items"],
        }

    @app.post("/api/subscriptions")
    def add_subscription(request: SubscriptionCreateRequest) -> dict:
        try:
            book = _store(resolved).add(
                request.symbol,
                request.market_type,
                reason=request.reason,
                target_action=request.target_action,
                status=request.status,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"items": book.model_dump(mode="json")["items"]}

    @app.patch("/api/subscriptions/{symbol}")
    def update_subscription(symbol: str, request: SubscriptionUpdateRequest) -> dict:
        store = _store(resolved)
        book = store.load()
        normalized = normalize_symbol(symbol)
        updated_items = []
        found = False
        for item in book.items:
            if item.symbol != normalized:
                updated_items.append(item)
                continue
            found = True
            updated_items.append(
                item.model_copy(
                    update={
                        "status": request.status or item.status,
                        "reason": item.reason
                        if request.reason is None
                        else request.reason,
                        "target_action": item.target_action
                        if request.target_action is None
                        else request.target_action,
                    }
                )
            )
        if not found:
            raise HTTPException(status_code=404, detail=f"Symbol not found: {normalized}")
        next_book = SubscriptionBook(items=updated_items)
        store.save(next_book)
        return {"items": next_book.model_dump(mode="json")["items"]}

    @app.delete("/api/subscriptions/{symbol}")
    def delete_subscription(symbol: str) -> dict:
        book = _store(resolved).remove(symbol)
        return {"items": book.model_dump(mode="json")["items"]}

    @app.get("/api/rag/status")
    def rag_status(probe: bool = True) -> dict:
        if probe:
            missing_key_status = _missing_rag_embedding_key_status(resolved)
            if missing_key_status is not None:
                return missing_key_status
            if resolved.fundamental_retriever_factory is None:
                return _probe_rag_status_subprocess(resolved)
        return _fundamental_retriever(resolved, auto_ingest_seed=False).rag_status(probe=probe)

    @app.get("/api/observability/status")
    def observability_status() -> dict:
        return get_observability_status()

    @app.get("/api/runtime-config")
    def runtime_config() -> dict:
        return {
            "long_term_memory_enabled": resolve_long_term_memory_enabled(
                resolved.long_term_memory_enabled
            )
        }

    @app.get("/api/risk-position")
    def get_risk_position() -> dict:
        return _risk_position_store(resolved).load()

    @app.put("/api/risk-position")
    def update_risk_position(request: RiskPositionUpdateRequest) -> dict:
        try:
            return _risk_position_store(resolved).save(
                request.persona,
                request.target_weights,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/risk-position/reset")
    def reset_risk_position() -> dict:
        return _risk_position_store(resolved).reset()

    @app.get("/api/memories")
    def list_memories(status: Optional[MemoryStatus] = None, symbol: str | None = None) -> dict:
        agent = _memory_agent(resolved, auto_ingest_seed=False)
        statuses = [status] if status is not None else None
        items = agent.list_memories(statuses=statuses, symbol=symbol)
        return {
            "database_path": str(resolved.memory_database),
            "counts": agent.store.repository.counts_by_status(),
            "items": [item.model_dump(mode="json") for item in items],
        }

    @app.get("/api/memories/{memory_id}")
    def get_memory(memory_id: str) -> dict:
        agent = _memory_agent(resolved, auto_ingest_seed=False)
        memory = agent.store.repository.get(memory_id)
        if memory is None:
            raise HTTPException(status_code=404, detail="memory not found")
        evaluation = agent.store.repository.latest_evaluation(memory_id)
        return {
            "item": memory.model_dump(mode="json"),
            "versions": [
                item.model_dump(mode="json")
                for item in agent.store.repository.versions(memory_id)
            ],
            "evaluation": evaluation.model_dump(mode="json") if evaluation else None,
        }

    @app.post("/api/memories/{memory_id}/transition")
    def transition_memory(memory_id: str, request: MemoryTransitionRequest) -> dict:
        agent = _memory_agent(resolved, auto_ingest_seed=False)
        try:
            if request.status == MemoryStatus.APPROVED:
                if agent.evaluator is None:
                    raise ValueError("memory evaluator is unavailable")
                memory = agent.evaluator.promote(
                    memory_id,
                    actor=request.actor,
                    reason=request.reason,
                )
            else:
                memory = agent.transition(
                    memory_id,
                    request.status,
                    actor=request.actor,
                    reason=request.reason,
                )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"item": memory.model_dump(mode="json")}

    @app.post("/api/memories/{memory_id}/evaluate")
    def evaluate_memory(memory_id: str) -> dict:
        agent = _memory_agent(resolved, auto_ingest_seed=False)
        try:
            if agent.evaluator is None:
                raise ValueError("memory evaluator is unavailable")
            evaluation = agent.evaluator.evaluate(memory_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"evaluation": evaluation.model_dump(mode="json")}

    @app.post("/api/memories/{memory_id}/rollback")
    def rollback_memory(memory_id: str, request: MemoryRollbackRequest) -> dict:
        agent = _memory_agent(resolved, auto_ingest_seed=False)
        try:
            memory = agent.store.repository.rollback(
                memory_id,
                request.version,
                actor=request.actor,
                reason=request.reason,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"item": memory.model_dump(mode="json")}

    @app.post("/api/memory-outcomes")
    def record_memory_outcome(outcome: RunOutcome) -> dict:
        agent = _memory_agent(resolved, auto_ingest_seed=False)
        saved = agent.record_outcome(outcome)
        return {"outcome": saved.model_dump(mode="json")}

    @app.post("/api/rag/ingest-defaults")
    def rag_ingest_defaults() -> dict:
        return _fundamental_retriever(resolved, auto_ingest_seed=False).ingest_seed_knowledge()

    @app.post("/api/rag/ingest-text")
    def rag_ingest_text(request: RagTextIngestRequest) -> dict:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text is required.")
        return _fundamental_retriever(resolved, auto_ingest_seed=False).ingest_text_knowledge(
            title=request.title,
            text=request.text,
            source_type=request.source_type,
            source="ui:manual",
            symbols=request.symbols,
            tags=request.tags,
        )

    @app.post("/api/rag/ingest-online")
    def rag_ingest_online(request: RagOnlineIngestRequest) -> dict:
        symbols = _unique_symbols([*request.symbols, *_parse_symbol_text(request.manual_symbols)])
        if not symbols:
            raise HTTPException(status_code=400, detail="At least one symbol is required.")
        return _fundamental_retriever(resolved, auto_ingest_seed=False).ingest_online_fundamental_research(
            symbols=symbols,
            trade_date=request.trade_date,
            look_back_days=request.look_back_days,
        )

    @app.get("/api/portfolio/simulated")
    def simulated_portfolio() -> dict:
        getter = resolved.portfolio_snapshot_getter or get_futu_portfolio_snapshot
        try:
            if resolved.portfolio_snapshot_getter is None:
                _ensure_futu_opend_reachable()
            snapshot = _get_portfolio_snapshot_with_timeout(
                getter,
                ExecutionMode.SIMULATION,
                timeout_seconds=resolved.portfolio_snapshot_timeout_seconds,
            )
        except Exception as exc:
            return {
                "available": False,
                "mode": ExecutionMode.SIMULATION.value,
                "total_value": None,
                "cash": None,
                "cash_weight": None,
                "position_weights": {},
                "updated_at": _utc_now_iso(),
                "error": str(exc),
            }
        total_value = snapshot.total_value
        cash = snapshot.cash
        cash_weight = (
            round(float(cash) / float(total_value), 6)
            if cash is not None and total_value is not None and total_value > 0
            else None
        )
        return {
            "available": True,
            "mode": ExecutionMode.SIMULATION.value,
            "total_value": total_value,
            "cash": cash,
            "cash_weight": cash_weight,
            "position_weights": snapshot.position_weights,
            "updated_at": _utc_now_iso(),
            "error": None,
        }

    def launch_run(request: RunCreateRequest, checkpoint: dict | None = None) -> dict:
        if request.portfolio_mode not in {
            ExecutionMode.SIMULATION,
            ExecutionMode.LIVE,
        }:
            raise HTTPException(status_code=400, detail="Invalid portfolio mode.")
        symbols = _unique_symbols([*request.symbols, *_parse_symbol_text(request.manual_symbols)])
        if not symbols:
            raise HTTPException(status_code=400, detail="At least one symbol is required.")

        run_id, output_dir = runtime.new_run_dir()
        long_term_memory_enabled = resolve_long_term_memory_enabled(
            request.long_term_memory_enabled
            if request.long_term_memory_enabled is not None
            else resolved.long_term_memory_enabled
        )
        params = {
            "subscription_symbols": symbols,
            "selected_analysts": request.selected_analysts,
            "price_history_by_symbol": None,
            "fundamental_analysis_by_symbol": None,
            "portfolio": None,
            "report_output_dir": str(output_dir),
            "trade_date": request.trade_date,
            "look_back_days": request.look_back_days,
            "mode": ExecutionMode.SIMULATION,
            "portfolio_mode": request.portfolio_mode,
            "user_confirmed": False,
            "run_id": run_id,
        }
        params["subscription_book"] = _store(resolved).load()
        risk_position = _risk_position_store(resolved).load()
        persona_config = UserPersonaConfig(**risk_position["persona"])
        params["persona_config"] = persona_config
        params["target_weight_overrides"] = risk_position["target_weights"]
        defaults = {
            "persona": persona_config.model_dump(mode="json"),
            "selected_analysts": [analyst.value for analyst in request.selected_analysts],
            "execution_mode_default": ExecutionMode.SIMULATION.value,
            "portfolio_mode_default": ExecutionMode.SIMULATION.value,
            "look_back_days_default": 90,
            "long_term_memory_enabled": long_term_memory_enabled,
        }
        tracker = resolved.tracker_cls(
            output_dir=output_dir,
            run_id=run_id,
            symbols=symbols,
            params=params,
            defaults=defaults,
            nodes=resolved.graph_cls.NODE_ORDER,
        )
        context = CONTEXT.get()
        tracker.status.update(origin_request_id=context.get("request_id", ""), trace_id=context.get("trace_id", ""))
        tracker.flush()
        control = RunControl(run_id=run_id, output_dir=output_dir, tracker=tracker, params=params)
        try:
            if checkpoint is not None:
                runtime.consume(checkpoint["checkpoint_id"], control, checkpoint)
            runtime.controls[run_id] = control
            job_args = (resolved.graph_cls, control, runtime, resolved.memory_database,
                        resolved.rag_chroma_dir, long_term_memory_enabled)
            job_context = copy_context()
            def job():
                token = CONTEXT.set({**CONTEXT.get(), "run_id": run_id,
                                     "origin_request_id": tracker.status["origin_request_id"]})
                try:
                    _run_copilot_job(*job_args)
                finally:
                    CONTEXT.reset(token)
            if resolved.run_in_background:
                thread = Thread(
                target=job_context.run,
                args=(job,),
                name=f"copilot-ui-{run_id}",
                daemon=True,
                )
                control.thread = thread
                thread.start()
            else:
                job_context.run(job)
        except BaseException:
            runtime.controls.pop(run_id, None)
            remove_owned(runtime.root, output_dir)
            raise
        return run_response(run_id)

    @app.post("/api/runs")
    def create_run(request: RunCreateRequest) -> dict:
        with runtime.lock:
            if runtime.shutting_down.is_set():
                raise HTTPException(status_code=503, detail="UI service is shutting down.")
            return launch_run(request)

    @app.get("/api/checkpoints")
    def list_checkpoints() -> dict:
        runtime.start()
        with runtime.lock:
            return {"items": runtime.list_checkpoints(), "session_id": runtime.session_id}

    @app.post("/api/checkpoints/{checkpoint_id}/resume")
    def resume_checkpoint(checkpoint_id: str) -> dict:
        runtime.start()
        with runtime.lock:
            if runtime.shutting_down.is_set():
                raise HTTPException(status_code=503, detail="UI service is shutting down.")
            try:
                allowed = list(resolved.graph_cls.NODE_ORDER)
                checkpoint = runtime.read_checkpoint(checkpoint_id, allowed)
                params = checkpoint["params"]
                request = RunCreateRequest(
                    symbols=params["subscription_symbols"], selected_analysts=params["selected_analysts"],
                    trade_date=params.get("trade_date"), look_back_days=params["look_back_days"],
                    portfolio_mode=params["portfolio_mode"],
                    long_term_memory_enabled=checkpoint.get("defaults", {}).get("long_term_memory_enabled", True),
                )
                if checkpoint["snapshot"]["state"] is not None and not hasattr(resolved.graph_cls, "resume"):
                    raise ValueError("This graph does not support checkpoint recovery.")
                snapshot = checkpoint["snapshot"]
                if snapshot["state"] is not None:
                    expected = resolved.graph_cls.checkpoint_node_order(snapshot["state"].get("selected_analysts", []))
                    if expected != snapshot["node_order"]:
                        raise ValueError("Checkpoint node order does not match this graph.")
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except (ValueError, KeyError, TypeError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return launch_run(request, checkpoint)

    @app.get("/api/runs")
    @locked_run_io
    def list_runs() -> dict:
        runtime.start()
        runs = []
        reports_dir = resolved.reports_dir
        if reports_dir.exists():
            for path in reports_dir.iterdir():
                if not path.is_dir() or not path.name.startswith("run_"):
                    continue
                status = _read_json(path / "run_status.json")
                runs.append(
                    {
                        "run_id": path.name,
                        "status": ("interrupted" if status.get("status") == "running" else status.get("status")) if status else None,
                        "cancel_requested": bool(status.get("cancel_requested")) if status else False,
                        "symbols": status.get("symbols", []) if status else [],
                        "updated_at": status.get("updated_at") if status else None,
                        "path": str(path),
                    }
                )
        with runtime.lock:
            for control in runtime.controls.values():
                status = control.tracker.status
                runs.append({"run_id": control.run_id, "status": status["status"],
                             "cancel_requested": status.get("cancel_requested", False),
                             "symbols": status["symbols"], "updated_at": status["updated_at"],
                             "path": str(control.output_dir)})
        runs = list({run["run_id"]: run for run in runs}.values())
        runs.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return {"items": runs[:50], "session_id": runtime.session_id}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        return run_response(run_id)

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict:
        run_dir_for(run_id)
        if runtime.request_user_cancel(run_id):
            return run_response(run_id)
        payload = run_response(run_id)
        if not _is_terminal_run_status(payload["status"].get("status")):
            raise HTTPException(status_code=409, detail="Run has no active worker.")
        return payload

    @app.get("/api/runs/{run_id}/trace")
    @locked_run_io
    def get_trace(run_id: str, request: Request) -> dict:
        require_internal(request)
        run_dir = run_dir_for(run_id)
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        trace_path = run_dir / "trace.json"
        if not trace_path.exists():
            raise HTTPException(status_code=404, detail=f"Trace not found: {run_id}")
        trace_payload = _read_json(trace_path)
        if not trace_payload:
            raise HTTPException(status_code=404, detail=f"Trace not found: {run_id}")
        return {"run_id": run_id, "trace": trace_payload}

    @app.post("/api/runs/{run_id}/orders/{symbol}/confirm-simulated")
    def confirm_simulated_order(run_id: str, symbol: str) -> dict:
        run_dir = run_dir_for(run_id)
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        status_path = run_dir / "run_status.json"
        status = _read_json(status_path)
        if not status:
            raise HTTPException(status_code=404, detail=f"Run status not found: {run_id}")
        if status.get("status") not in {"succeeded", "degraded"}:
            raise HTTPException(status_code=409, detail="Only completed runs can confirm orders.")
        try:
            order = _confirm_simulated_broker_order(
                settings=resolved,
                run_dir=run_dir,
                status=status,
                symbol=symbol,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        status["updated_at"] = _utc_now_iso()
        write_json_atomic(status_path, status)
        response = _run_response(run_id, run_dir)
        response["order"] = order
        return response

    @app.get("/api/reports/{run_id}/{report_key}", response_class=PlainTextResponse)
    @locked_run_io
    def get_report(run_id: str, report_key: str, request: Request) -> PlainTextResponse:
        if report_key in {"trace_json", "trace_markdown"}:
            require_internal(request)
        run_dir = run_dir_for(run_id)
        if not _is_safe_id(report_key):
            raise HTTPException(status_code=400, detail="Invalid report key.")
        status = run_response(run_id)["status"]
        report = status.get("reports", {}).get(report_key) if status else None
        if not report:
            raise HTTPException(status_code=404, detail=f"Report not found: {report_key}")
        report_path = _safe_report_path(resolved.reports_dir, report.get("path", ""))
        if not report_path.exists():
            raise HTTPException(status_code=404, detail=f"Report file missing: {report_key}")
        return PlainTextResponse(report_path.read_text(encoding="utf-8"))

    return app


BROKER_ACTIONS = {"buy", "sell", "reduce", "short", "cover"}


def _confirm_simulated_broker_order(
    *,
    settings: UISettings,
    run_dir: Path,
    status: dict,
    symbol: str,
) -> dict:
    row = _find_decision_row(status, symbol)
    if row is None:
        raise ValueError(f"Decision not found for symbol: {normalize_symbol(symbol)}")
    if row.get("submitted_to_broker") is True and row.get("broker_idempotency_key"):
        return _order_payload(row)

    _validate_simulated_order(status, row)
    attempts = int(row.get("broker_attempt_count") or 0)
    max_attempts = _max_simulated_order_attempts()
    if attempts >= max_attempts:
        raise ValueError("Simulated order retry limit reached; start a new run.")
    price = _latest_confirm_price(settings, row)
    if price <= 0:
        raise ValueError("Unable to fetch a current price; simulated order not submitted.")
    decision_price = float(row.get("current_price") or 0)
    max_drift = _simulated_order_max_price_drift()
    if decision_price > 0 and abs(price - decision_price) / decision_price > max_drift:
        raise ValueError(
            f"Price moved more than {max_drift:.1%} since the run "
            f"({decision_price:.2f} -> {price:.2f}); rerun before confirming."
        )
    quantity = min(int(row.get("quantity") or 0), _max_simulated_order_quantity())
    if quantity <= 0:
        raise ValueError("Simulated order quantity is zero after max quantity guard.")
    action = str(row.get("action") or "").lower()
    idempotency_key = _broker_idempotency_key(
        status=status,
        symbol=row["symbol"],
        action=action,
        quantity=quantity,
        price=price,
    )
    request = BrokerExecutionRequest(
        idempotency_key=idempotency_key,
        symbol=row["symbol"],
        side=_broker_side(action),
        quantity=quantity,
        price=price,
        order_type="NORMAL",
        trd_env="SIMULATE",
    )
    adapter = (
        settings.simulated_broker_factory()
        if settings.simulated_broker_factory is not None
        else FutuSimulatedExecutionAdapter()
    )
    try:
        result = _broker_result(adapter.place_order(request))
    except Exception as exc:
        result = BrokerExecutionResult(
            idempotency_key=idempotency_key,
            submitted=False,
            status="failed",
            message=f"Futu simulated order failed: {exc}",
        )
    _apply_broker_result(
        row,
        result,
        submitted_quantity=quantity,
        max_attempts=max_attempts,
    )
    _refresh_broker_metrics(status)
    if result.submitted and result.order_id:
        account_id = result.raw.get("acc_id") or result.raw.get("account_id") or getattr(adapter, "acc_id", None)
        if account_id and str(account_id) != "0" and status.get("run_id"):
            try:
                create_review_service(settings.memory_database).repository.record_order(
                    "SIMULATE", str(account_id), str(result.order_id), status["run_id"], row["symbol"])
            except Exception as exc:
                # Submission already succeeded: preserve that fact to prevent duplicate orders.
                status.setdefault("errors", []).append(f"review order linkage failed: {exc}")
    _record_simulated_order_audit(run_dir, status, row)
    return _order_payload(row)


def _find_decision_row(status: dict, symbol: str) -> dict | None:
    normalized = normalize_symbol(symbol)
    rows = status.get("decision_summary", {}).get("symbols", []) or []
    for row in rows:
        if normalize_symbol(str(row.get("symbol") or "")) == normalized:
            return row
    return None


def _validate_simulated_order(status: dict, row: dict) -> None:
    if status.get("status") not in {"succeeded", "degraded"}:
        raise ValueError("Run must finish successfully before simulated order confirmation.")
    if status.get("status") == "degraded":
        raise ValueError("Portfolio fetch failed; simulated order is blocked.")
    params = status.get("params", {}) or {}
    if _value(params.get("mode", ExecutionMode.SIMULATION.value)) != ExecutionMode.SIMULATION.value:
        raise ValueError("Only simulation runs can submit simulated broker orders.")
    if _value(params.get("portfolio_mode", ExecutionMode.SIMULATION.value)) != ExecutionMode.SIMULATION.value:
        raise ValueError("Only simulated portfolio snapshots can submit simulated broker orders.")
    graph_errors = [str(item) for item in status.get("graph_errors", []) or []]
    status_errors = [str(item.get("message", item)) for item in status.get("errors", []) or []]
    if any("portfolio_fetch_failed" in item for item in [*graph_errors, *status_errors]):
        raise ValueError("Portfolio fetch failed for this run; simulated order is blocked.")
    if row.get("approved_by_risk") is not True:
        raise ValueError("Risk Manager did not approve this action.")
    if row.get("pending_broker_order") is not True:
        raise ValueError("No pending simulated broker order for this symbol.")
    if row.get("broker_confirmation_required") is not True:
        raise ValueError("This symbol does not require broker confirmation.")
    action = str(row.get("action") or "").lower()
    if action not in BROKER_ACTIONS:
        raise ValueError(f"Action is not broker-tradable: {action or '-'}")
    if int(row.get("quantity") or 0) <= 0:
        raise ValueError("Portfolio Manager quantity is zero.")
    if float(row.get("current_price") or 0) <= 0:
        raise ValueError("No valid current price is available for simulated order.")
    if float(row.get("portfolio_value") or 0) <= 0:
        raise ValueError("No valid portfolio value is available for simulated order.")


def _broker_side(action: str) -> str:
    if action in {"buy", "cover"}:
        return "BUY"
    if action in {"sell", "reduce", "short"}:
        return "SELL"
    raise ValueError(f"Unsupported broker action: {action}")


def _max_simulated_order_quantity() -> int:
    raw_value = os.getenv("COPILOT_SIMULATED_ORDER_MAX_QUANTITY", "100")
    try:
        value = int(raw_value)
    except ValueError:
        return 100
    return max(value, 0)


def _max_simulated_order_attempts() -> int:
    raw_value = os.getenv("COPILOT_SIMULATED_ORDER_MAX_ATTEMPTS", "3")
    try:
        value = int(raw_value)
    except ValueError:
        return 3
    return max(value, 1)


def _simulated_order_max_price_drift() -> float:
    raw_value = os.getenv("COPILOT_SIMULATED_ORDER_MAX_PRICE_DRIFT", "0.10")
    try:
        value = float(raw_value)
    except ValueError:
        return 0.10
    return max(value, 0.0)


def _latest_confirm_price(settings: UISettings, row: dict) -> float:
    symbol = str(row.get("symbol") or "")
    try:
        if settings.latest_price_getter is not None:
            return max(float(settings.latest_price_getter(symbol)), 0.0)
        info = get_futu_stock_info(symbol)
        for key in ("last_price", "latest_price", "open_price", "open"):
            raw = info.get(key)
            if raw not in (None, ""):
                return max(float(raw), 0.0)
        return 0.0
    except Exception as exc:
        raise ValueError(f"Unable to fetch current price for {symbol}: {exc}") from exc


def _broker_idempotency_key(
    *,
    status: dict,
    symbol: str,
    action: str,
    quantity: int,
    price: float,
) -> str:
    run_id = status.get("run_id") or "run"
    return f"sim:{run_id}:{normalize_symbol(symbol)}:{action}:{quantity}:{price:.4f}"


def _broker_result(value) -> BrokerExecutionResult:
    if isinstance(value, BrokerExecutionResult):
        return value
    if isinstance(value, dict):
        return BrokerExecutionResult.model_validate(value)
    raise ValueError(f"Invalid broker result: {value}")


def _apply_broker_result(
    row: dict,
    result: BrokerExecutionResult,
    *,
    submitted_quantity: int,
    max_attempts: int,
) -> None:
    row["submitted_to_broker"] = result.submitted
    row["submitted_quantity"] = submitted_quantity
    row["broker_order_id"] = result.order_id
    row["broker_status"] = result.status
    row["broker_message"] = result.message
    row["last_broker_attempt_at"] = _utc_now_iso()
    row["broker_attempt_count"] = int(row.get("broker_attempt_count") or 0) + 1
    if result.submitted:
        row["pending_broker_order"] = False
        row["broker_confirmation_required"] = False
        row["broker_idempotency_key"] = result.idempotency_key
        row["broker_retry_available"] = False
    else:
        attempts = int(row.get("broker_attempt_count") or 0)
        retry_available = attempts < max_attempts
        row["broker_retry_available"] = retry_available
        if not retry_available:
            row["pending_broker_order"] = False
            row["broker_confirmation_required"] = False
    base_message = row.get("execution_message") or ""
    broker_message = result.message or result.status or ""
    if broker_message:
        row["execution_message"] = f"{base_message} Broker: {broker_message}".strip()


def _refresh_broker_metrics(status: dict) -> None:
    rows = status.get("decision_summary", {}).get("symbols", []) or []
    metrics = status.setdefault("decision_summary", {}).setdefault("metrics", {})
    metrics["pending_broker_order_count"] = sum(
        1 for row in rows if row.get("pending_broker_order") is True
    )
    metrics["submitted_broker_order_count"] = sum(
        1 for row in rows if row.get("submitted_to_broker") is True
    )


def _record_simulated_order_audit(run_dir: Path, status: dict, row: dict) -> None:
    _append_simulated_order_audit(run_dir, row)
    report_path = _write_simulated_order_report(run_dir)
    _record_status_report(status, "simulated_broker_orders", report_path)


def _append_simulated_order_audit(run_dir: Path, row: dict) -> None:
    path = run_dir / "simulated_broker_orders.jsonl"
    event = {
        "timestamp": _utc_now_iso(),
        "symbol": row.get("symbol"),
        "action": row.get("action"),
        "quantity": row.get("quantity"),
        "submitted_quantity": row.get("submitted_quantity"),
        "price": row.get("current_price"),
        "submitted": row.get("submitted_to_broker"),
        "broker_order_id": row.get("broker_order_id"),
        "broker_status": row.get("broker_status"),
        "broker_message": row.get("broker_message"),
        "broker_idempotency_key": row.get("broker_idempotency_key"),
        "broker_attempt_count": int(row.get("broker_attempt_count") or 0),
        "broker_retry_available": bool(row.get("broker_retry_available")),
        "last_broker_attempt_at": row.get("last_broker_attempt_at"),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _write_simulated_order_report(run_dir: Path) -> Path:
    jsonl_path = run_dir / "simulated_broker_orders.jsonl"
    report_path = run_dir / "simulated_broker_orders.md"
    events = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                events.append({"broker_message": f"Invalid JSONL audit line: {line}"})
    lines = [
        "# Simulated Broker Orders",
        "",
        "Orders in this report were submitted only to Futu SIMULATE after result-page confirmation.",
        "",
    ]
    if not events:
        lines.append("- No simulated broker orders recorded.")
    else:
        lines.extend(
            [
                "| Time | Symbol | Action | Submitted Qty | Price | Submitted | Broker Status | Broker Order ID | Message |",
                "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |",
            ]
        )
        for event in events:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _md_cell(event.get("timestamp")),
                        _md_cell(event.get("symbol")),
                        _md_cell(event.get("action")),
                        _md_cell(event.get("submitted_quantity")),
                        _md_cell(event.get("price")),
                        _md_cell(event.get("submitted")),
                        _md_cell(event.get("broker_status")),
                        _md_cell(event.get("broker_order_id")),
                        _md_cell(event.get("broker_message")),
                    ]
                )
                + " |"
            )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _record_status_report(status: dict, key: str, path: Path) -> None:
    exists = path.exists()
    stat = path.stat() if exists else None
    status.setdefault("reports", {})[key] = {
        "path": str(path),
        "exists": exists,
        "bytes": stat.st_size if stat else 0,
        "updated_at": _utc_now_iso(),
    }


def _md_cell(value) -> str:
    text = "-" if value is None or value == "" else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _order_payload(row: dict) -> dict:
    return {
        "symbol": row.get("symbol"),
        "action": row.get("action"),
        "quantity": row.get("quantity"),
        "submitted_quantity": row.get("submitted_quantity"),
        "price": row.get("current_price"),
        "submitted_to_broker": row.get("submitted_to_broker"),
        "broker_order_id": row.get("broker_order_id"),
        "broker_status": row.get("broker_status"),
        "broker_message": row.get("broker_message"),
        "broker_idempotency_key": row.get("broker_idempotency_key"),
        "broker_attempt_count": int(row.get("broker_attempt_count") or 0),
        "broker_retry_available": bool(row.get("broker_retry_available")),
        "last_broker_attempt_at": row.get("last_broker_attempt_at"),
    }


def _value(value):
    return value.value if hasattr(value, "value") else value


def _store(settings: UISettings) -> SubscriptionStore:
    return SubscriptionStore(settings.subscriptions_file)


def _risk_position_store(settings: UISettings) -> RiskPositionStore:
    return RiskPositionStore(settings.risk_position_file)


def _missing_rag_embedding_key_status(settings: UISettings) -> dict | None:
    if settings.fundamental_retriever_factory is not None:
        return None
    load_copilot_env()
    api_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
    )
    if api_key:
        return None
    chroma_dir = settings.rag_chroma_dir or DEFAULT_CHROMA_DIR
    return {
        "backend": "fundamental_chroma",
        "available": False,
        "probe_status": "failed",
        "collection": DEFAULT_COLLECTION_NAME,
        "path": str(chroma_dir),
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL),
        "retrieval": "fundamental_child_bm25_rrf_rerank_parent_context",
        "document_count": 0,
        "scope": "fundamental_only",
        "allowed_source_types": [],
        "chunking": {},
        "error": "OPENAI_API_KEY or DASHSCOPE_API_KEY is not set",
    }


def _probe_rag_status_subprocess(settings: UISettings, timeout_seconds: float = 30.0) -> dict:
    chroma_dir = settings.rag_chroma_dir or DEFAULT_CHROMA_DIR
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    code = (
        "import json, sys\n"
        "from ai_trading_copilot.copilot.run import create_default_fundamental_research_retriever\n"
        "chroma_dir = sys.argv[1]\n"
        "retriever = create_default_fundamental_research_retriever(rag_chroma_dir=chroma_dir, auto_ingest_seed=False)\n"
        "print(json.dumps(retriever.rag_status(probe=True), ensure_ascii=False))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", code, str(chroma_dir)],
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            env=env,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return _rag_probe_error_status(
            settings,
            f"RAG status probe timed out after {timeout_seconds:.0f}s.",
        )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return _rag_probe_error_status(
            settings,
            detail or f"RAG status probe exited with code {result.returncode}.",
        )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return _rag_probe_error_status(settings, "RAG status probe returned no output.")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        return _rag_probe_error_status(settings, f"RAG status probe returned invalid JSON: {exc}")
    return payload if isinstance(payload, dict) else _rag_probe_error_status(settings, "RAG status probe returned a non-object payload.")


def _rag_probe_error_status(settings: UISettings, error: str) -> dict:
    chroma_dir = settings.rag_chroma_dir or DEFAULT_CHROMA_DIR
    return {
        "backend": "fundamental_chroma",
        "available": False,
        "probe_status": "failed",
        "collection": DEFAULT_COLLECTION_NAME,
        "path": str(chroma_dir),
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_OPENAI_EMBEDDING_MODEL),
        "retrieval": "fundamental_child_bm25_rrf_rerank_parent_context",
        "document_count": 0,
        "scope": "fundamental_only",
        "allowed_source_types": [],
        "chunking": {},
        "error": error[:1000],
    }


def _ensure_futu_opend_reachable(timeout_seconds: float = 0.75) -> None:
    host = os.getenv("FUTU_OPEND_HOST", "127.0.0.1")
    port = int(os.getenv("FUTU_OPEND_PORT", "11111"))
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return
    except OSError as exc:
        raise RuntimeError(f"Futu OpenD is not reachable at {host}:{port}: {exc}") from exc


def _get_portfolio_snapshot_with_timeout(
    getter: Callable[[ExecutionMode], object],
    mode: ExecutionMode,
    *,
    timeout_seconds: float,
):
    result_queue: Queue = Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put((True, getter(mode)))
        except Exception as exc:  # pragma: no cover - surfaced through endpoint tests
            result_queue.put((False, exc))

    Thread(target=worker, daemon=True).start()
    try:
        ok, value = result_queue.get(timeout=max(timeout_seconds, 0.001))
    except Empty as exc:
        raise TimeoutError("Futu portfolio snapshot request timed out.") from exc
    if ok:
        return value
    raise value


def _memory_agent(settings: UISettings, *, auto_ingest_seed: bool = True):
    return create_default_memory_agent(settings.memory_database)


def _fundamental_retriever(settings: UISettings, *, auto_ingest_seed: bool = True):
    if settings.fundamental_retriever_factory is not None:
        return settings.fundamental_retriever_factory(settings, auto_ingest_seed)
    return create_default_fundamental_research_retriever(
        rag_chroma_dir=settings.rag_chroma_dir or DEFAULT_CHROMA_DIR,
        auto_ingest_seed=auto_ingest_seed,
    )


def _run_copilot_job(
    graph_cls: type,
    control: RunControl,
    runtime: RunLifecycle,
    memory_database: Path = DEFAULT_MEMORY_DATABASE,
    rag_chroma_dir: Path | None = None,
    long_term_memory_enabled: bool = True,
) -> None:
    tracker = control.tracker
    state = None
    error = None
    try:
        record("run.start")
        if control.cancel_event.is_set():
            raise RunCancelled("Run stopped before initialization.")
        params = dict(control.params)
        fundamental_rag_retriever = None
        if (
            accepts_keyword(graph_cls, "fundamental_rag_retriever")
            and _ui_enable_fundamental_rag()
        ):
            fundamental_rag_retriever = create_default_fundamental_research_retriever(
                rag_chroma_dir=rag_chroma_dir or DEFAULT_CHROMA_DIR,
            )
        memory_agent = (
            create_default_memory_agent(memory_database)
            if long_term_memory_enabled
            else None
        )
        graph = _create_graph(
            graph_cls,
            run_tracker=tracker,
            memory_agent=memory_agent,
            fundamental_rag_retriever=fundamental_rag_retriever,
            cancellation_checker=control.cancel_event.is_set,
            snapshot_callback=lambda state, order, index, **kwargs: runtime.capture(control, state, order, index, **kwargs),
            commit_callback=lambda: runtime.commit_boundary(control),
        )
        if control.cancel_event.is_set():
            raise RunCancelled("Run stopped before graph execution.")
        state = graph.resume(control.resume_snapshot) if control.resume_snapshot is not None else graph.run(**params)
        review_registered = False
        try:
            create_review_service(memory_database).register_run({**state, "memory_learning_enabled": memory_agent is not None})
            review_registered = True
        except Exception as persistence_error:
            tracker.add_error(f"delayed review registration failed: {persistence_error}")
        if review_registered and memory_agent is not None and not control.cancel_event.is_set():
            try:
                candidates = memory_agent.learn_from_run(state)
                if candidates:
                    state["memory_candidates"] = candidates
            except Exception as learning_error:
                tracker.add_error(f"memory reflection skipped: {learning_error}")
    except RunCancelled as exc:
        error = exc
    except Exception as exc:  # pragma: no cover - covered through API behavior
        error = exc
        record("run.failed", level="ERROR", **exception_data(exc))
    except BaseException:
        runtime.request_discard()
        raise
    finally:
        try:
            runtime.finalize(
                control,
                state=state,
                error=error,
                degraded=_degraded_run_state(state),
            )
            record("run.end", outcome=tracker.status["status"], stop_reason=control.reason)
        finally:
            runtime.unregister(control)


def _degraded_run_state(state: dict | None) -> bool:
    if not state:
        return False
    return any(
        str(item).startswith("portfolio_fetch_failed")
        or str(item) == "llm_unavailable"
        for item in (state.get("errors") or [])
    )


def _create_graph(
    graph_cls: type,
    *,
    run_tracker: RunTracker,
    memory_agent,
    fundamental_rag_retriever=None,
    cancellation_checker: Callable[[], bool] | None = None,
    snapshot_callback: Callable | None = None,
    commit_callback: Callable | None = None,
):
    kwargs = {"run_tracker": run_tracker}
    if accepts_keyword(graph_cls, "memory_agent"):
        kwargs["memory_agent"] = memory_agent
    if (
        fundamental_rag_retriever is not None
        and accepts_keyword(graph_cls, "fundamental_rag_retriever")
    ):
        kwargs["fundamental_rag_retriever"] = fundamental_rag_retriever
    if accepts_keyword(graph_cls, "force_sequential"):
        kwargs["force_sequential"] = True
    if accepts_keyword(graph_cls, "parallel_analysts"):
        kwargs["parallel_analysts"] = os.getenv("COPILOT_FORCE_SEQUENTIAL", "").strip().lower() not in {"1", "true", "yes", "on"}
    if commit_callback is not None and accepts_keyword(graph_cls, "commit_callback"):
        kwargs["commit_callback"] = commit_callback
    if cancellation_checker is not None and accepts_keyword(graph_cls, "cancellation_checker"):
        kwargs["cancellation_checker"] = cancellation_checker
    if snapshot_callback is not None and accepts_keyword(graph_cls, "snapshot_callback"):
        kwargs["snapshot_callback"] = snapshot_callback
    return graph_cls(**kwargs)


def _ui_enable_fundamental_rag() -> bool:
    value = os.getenv("COPILOT_UI_ENABLE_FUNDAMENTAL_RAG", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _run_response(run_id: str, output_dir: Path) -> dict:
    status = _read_json(output_dir / "run_status.json")
    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "status": status,
        "reports": status.get("reports", {}) if status else {},
        "errors": status.get("errors", []) if status else [],
    }


def _is_terminal_run_status(status: str | None) -> bool:
    return status in {NODE_SUCCEEDED, NODE_FAILED, NODE_CANCELLED, NODE_DEGRADED}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        content = path.read_text(encoding="utf-8-sig")
        return json.loads(content) if content.strip() else {}
    except (json.JSONDecodeError, UnicodeError):
        return {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_symbol_text(value: str) -> List[str]:
    if not value:
        return []
    return [item for item in re.split(r"[\s,;]+", value) if item]


def _unique_symbols(values: Iterable[str]) -> List[str]:
    normalized = []
    seen = set()
    for value in values:
        symbol = normalize_symbol(value)
        if symbol and symbol not in seen:
            normalized.append(symbol)
            seen.add(symbol)
    return normalized


def _safe_report_path(reports_dir: Path, raw_path: str) -> Path:
    if not raw_path:
        raise HTTPException(status_code=404, detail="Report path missing.")
    try:
        path = checked_path(reports_dir, Path(raw_path))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    reports_root = reports_dir.resolve()
    if reports_root != path and reports_root not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid report path.")
    return path


def _is_safe_id(value: str) -> bool:
    return bool(value and SAFE_ID.fullmatch(value) and ".." not in value)


app = create_app()
