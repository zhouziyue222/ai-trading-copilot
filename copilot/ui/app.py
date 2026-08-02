"""FastAPI application for the local AI trading copilot dashboard."""

from __future__ import annotations

import json
import re
import inspect
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Callable, Iterable, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ai_trading_copilot.copilot.config import DEFAULT_REPORT_OUTPUT_DIR
from ai_trading_copilot.copilot.config.defaults import DEFAULT_PERSONA_CONFIG
from ai_trading_copilot.copilot.domain.enums import (
    AnalystType,
    ExecutionMode,
    MarketType,
    SubscriptionStatus,
)
from ai_trading_copilot.copilot.domain.models import (
    Subscription,
    SubscriptionBook,
    normalize_symbol,
)
from ai_trading_copilot.copilot.graph import CopilotLangGraph
from ai_trading_copilot.copilot.run import (
    DEFAULT_ANALYSTS,
    DEFAULT_MEMORY_FILE,
    DEFAULT_SUBSCRIPTIONS_FILE,
    create_default_fundamental_research_retriever,
    create_default_memory_agent,
)
from ai_trading_copilot.copilot.services.run_tracker import RunTracker
from ai_trading_copilot.copilot.services.subscription_service import SubscriptionStore


SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class UISettings:
    subscriptions_file: Path = DEFAULT_SUBSCRIPTIONS_FILE
    memory_file: Path = DEFAULT_MEMORY_FILE
    rag_chroma_dir: Optional[Path] = None
    reports_dir: Path = DEFAULT_REPORT_OUTPUT_DIR
    graph_cls: type = CopilotLangGraph
    tracker_cls: type = RunTracker
    run_in_background: bool = True
    fundamental_retriever_factory: Optional[Callable[["UISettings", bool], object]] = None


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
    broker_execution_enabled: bool = False


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


def create_app(settings: UISettings | None = None) -> FastAPI:
    resolved = settings or UISettings()
    static_dir = Path(__file__).resolve().parent / "static"
    app = FastAPI(title="AI Trading Copilot UI")
    app.state.ui_settings = resolved
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

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
    def rag_status() -> dict:
        return _fundamental_retriever(resolved, auto_ingest_seed=False).rag_status()

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

    @app.post("/api/runs")
    def create_run(request: RunCreateRequest) -> dict:
        if request.portfolio_mode not in {
            ExecutionMode.SIMULATION,
            ExecutionMode.LIVE,
        }:
            raise HTTPException(status_code=400, detail="Invalid portfolio mode.")
        symbols = _unique_symbols([*request.symbols, *_parse_symbol_text(request.manual_symbols)])
        if not symbols:
            raise HTTPException(status_code=400, detail="At least one symbol is required.")

        run_id = _new_ui_run_id(resolved.reports_dir)
        output_dir = resolved.reports_dir / run_id
        params = {
            "subscription_symbols": symbols,
            "selected_analysts": request.selected_analysts,
            "price_history_by_symbol": None,
            "fundamental_news_by_symbol": None,
            "portfolio": None,
            "report_output_dir": str(output_dir),
            "trade_date": request.trade_date,
            "look_back_days": request.look_back_days,
            "mode": ExecutionMode.SIMULATION,
            "portfolio_mode": request.portfolio_mode,
            "user_confirmed": False,
            "broker_execution_enabled": request.broker_execution_enabled,
            "run_id": run_id,
        }
        defaults = {
            "persona": DEFAULT_PERSONA_CONFIG,
            "selected_analysts": [analyst.value for analyst in request.selected_analysts],
            "execution_mode_default": ExecutionMode.SIMULATION.value,
            "portfolio_mode_default": ExecutionMode.SIMULATION.value,
            "broker_execution_default": False,
            "look_back_days_default": 90,
        }
        tracker = resolved.tracker_cls(
            output_dir=output_dir,
            run_id=run_id,
            symbols=symbols,
            params=params,
            defaults=defaults,
            nodes=resolved.graph_cls.NODE_ORDER,
        )

        if resolved.run_in_background:
            Thread(
                target=_run_copilot_job,
                args=(
                    resolved.graph_cls,
                    tracker,
                    params,
                    resolved.memory_file,
                    resolved.rag_chroma_dir,
                ),
                name=f"copilot-ui-{run_id}",
                daemon=True,
            ).start()
        else:
            _run_copilot_job(
                resolved.graph_cls,
                tracker,
                params,
                resolved.memory_file,
                resolved.rag_chroma_dir,
            )

        return _run_response(run_id, output_dir)

    @app.get("/api/runs")
    def list_runs() -> dict:
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
                        "status": status.get("status") if status else None,
                        "symbols": status.get("symbols", []) if status else [],
                        "updated_at": status.get("updated_at") if status else None,
                        "path": str(path),
                    }
                )
        runs.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return {"items": runs[:50]}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        run_dir = _safe_run_dir(resolved.reports_dir, run_id)
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return _run_response(run_id, run_dir)

    @app.get("/api/reports/{run_id}/{report_key}", response_class=PlainTextResponse)
    def get_report(run_id: str, report_key: str) -> PlainTextResponse:
        run_dir = _safe_run_dir(resolved.reports_dir, run_id)
        if not _is_safe_id(report_key):
            raise HTTPException(status_code=400, detail="Invalid report key.")
        status = _read_json(run_dir / "run_status.json")
        report = status.get("reports", {}).get(report_key) if status else None
        if not report:
            raise HTTPException(status_code=404, detail=f"Report not found: {report_key}")
        report_path = _safe_report_path(resolved.reports_dir, report.get("path", ""))
        if not report_path.exists():
            raise HTTPException(status_code=404, detail=f"Report file missing: {report_key}")
        return PlainTextResponse(report_path.read_text(encoding="utf-8"))

    return app


def _store(settings: UISettings) -> SubscriptionStore:
    return SubscriptionStore(settings.subscriptions_file)


def _memory_agent(settings: UISettings, *, auto_ingest_seed: bool = True):
    return create_default_memory_agent(
        settings.memory_file,
        rag_chroma_dir=settings.rag_chroma_dir or settings.memory_file.parent / "rag_chroma",
        auto_ingest_seed=auto_ingest_seed,
    )


def _fundamental_retriever(settings: UISettings, *, auto_ingest_seed: bool = True):
    if settings.fundamental_retriever_factory is not None:
        return settings.fundamental_retriever_factory(settings, auto_ingest_seed)
    return create_default_fundamental_research_retriever(
        settings.rag_chroma_dir or settings.memory_file.parent / "rag_chroma",
        auto_ingest_seed=auto_ingest_seed,
    )


def _run_copilot_job(
    graph_cls: type,
    tracker: RunTracker,
    params: dict,
    memory_file: Path = DEFAULT_MEMORY_FILE,
    rag_chroma_dir: Path | None = None,
) -> None:
    state = None
    error = None
    try:
        fundamental_rag_retriever = None
        if _accepts_keyword(graph_cls, "fundamental_rag_retriever"):
            fundamental_rag_retriever = create_default_fundamental_research_retriever(
                rag_chroma_dir or memory_file.parent / "rag_chroma",
            )
        graph = _create_graph(
            graph_cls,
            run_tracker=tracker,
            memory_agent=create_default_memory_agent(
                memory_file,
                rag_chroma_dir=rag_chroma_dir or memory_file.parent / "rag_chroma",
            ),
            fundamental_rag_retriever=fundamental_rag_retriever,
        )
        state = graph.run(**params)
    except Exception as exc:  # pragma: no cover - covered through API behavior
        error = exc
        tracker.add_error(str(exc))
    finally:
        tracker.finish(failed=error is not None)
        tracker.write_audit(state=state, error=error)


def _create_graph(
    graph_cls: type,
    *,
    run_tracker: RunTracker,
    memory_agent,
    fundamental_rag_retriever=None,
):
    kwargs = {"run_tracker": run_tracker}
    if _accepts_keyword(graph_cls, "memory_agent"):
        kwargs["memory_agent"] = memory_agent
    if (
        fundamental_rag_retriever is not None
        and _accepts_keyword(graph_cls, "fundamental_rag_retriever")
    ):
        kwargs["fundamental_rag_retriever"] = fundamental_rag_retriever
    return graph_cls(**kwargs)


def _accepts_keyword(callable_obj, name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == name
        for parameter in signature.parameters.values()
    )


def _run_response(run_id: str, output_dir: Path) -> dict:
    status = _read_json(output_dir / "run_status.json")
    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "status": status,
        "reports": status.get("reports", {}) if status else {},
        "errors": status.get("errors", []) if status else [],
    }


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _new_ui_run_id(reports_dir: Path) -> str:
    base = f"run_UI_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_id = base
    suffix = 1
    while (reports_dir / run_id).exists():
        suffix += 1
        run_id = f"{base}_{suffix}"
    return run_id


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


def _safe_run_dir(reports_dir: Path, run_id: str) -> Path:
    if not _is_safe_id(run_id):
        raise HTTPException(status_code=400, detail="Invalid run id.")
    run_dir = (reports_dir / run_id).resolve()
    reports_root = reports_dir.resolve()
    if reports_root != run_dir and reports_root not in run_dir.parents:
        raise HTTPException(status_code=400, detail="Invalid run id.")
    return run_dir


def _safe_report_path(reports_dir: Path, raw_path: str) -> Path:
    if not raw_path:
        raise HTTPException(status_code=404, detail="Report path missing.")
    path = Path(raw_path).resolve()
    reports_root = reports_dir.resolve()
    if reports_root != path and reports_root not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid report path.")
    return path


def _is_safe_id(value: str) -> bool:
    return bool(value and SAFE_ID.fullmatch(value) and ".." not in value)


app = create_app()
