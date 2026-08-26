"""Persistent run tracking for copilot graph executions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

from ai_trading_copilot.copilot.services.tracing import sanitize_value


NODE_PENDING = "pending"
NODE_RUNNING = "running"
NODE_SUCCEEDED = "succeeded"
NODE_FAILED = "failed"
NODE_SKIPPED = "skipped"
NODE_CANCELLED = "cancelled"
MAX_ACTIVITY_HISTORY = 20


class RunTracker:
    """Write machine-readable progress and report status for one run."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        run_id: str,
        symbols: Iterable[str],
        params: dict[str, Any],
        defaults: dict[str, Any],
        nodes: Iterable[str],
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.output_dir / "run_status.json"
        self._lock = RLock()
        now = _utc_now()
        symbol_list = list(symbols)
        self.status: dict[str, Any] = {
            "run_id": run_id,
            "symbols": symbol_list,
            "params": _jsonable(params),
            "defaults": _jsonable(defaults),
            "current_node": None,
            "status": NODE_RUNNING,
            "cancel_requested": False,
            "cancel_requested_at": None,
            "cancelled_at": None,
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "nodes": {
                node: {
                    "status": NODE_PENDING,
                    "started_at": None,
                    "finished_at": None,
                    "error": None,
                    "activity": None,
                    "activity_history": [],
                }
                for node in nodes
            },
            "reports": {},
            "errors": [],
            "graph_errors": [],
        }
        self.status["decision_summary"] = _build_decision_summary(None, self.status)
        self.flush()

    def start_node(self, node_name: str) -> None:
        with self._lock:
            node = self._node(node_name)
            now = _utc_now()
            node["status"] = NODE_RUNNING
            node["started_at"] = node["started_at"] or now
            node["finished_at"] = None
            node["error"] = None
            node["activity"] = None
            node["activity_history"] = []
            self.status["current_node"] = node_name
            self._touch()

    def succeed_node(self, node_name: str) -> None:
        with self._lock:
            node = self._node(node_name)
            node["status"] = NODE_SUCCEEDED
            node["finished_at"] = _utc_now()
            if self.status.get("current_node") == node_name:
                self.status["current_node"] = None
            self._touch()

    def fail_node(self, node_name: str, error: Exception | str) -> None:
        with self._lock:
            node = self._node(node_name)
            node["status"] = NODE_FAILED
            node["finished_at"] = _utc_now()
            node["error"] = str(error)
            self.status["current_node"] = node_name
            self.add_error(f"{node_name}: {error}", flush=False)
            self._touch()

    def skip_node(self, node_name: str, reason: str = "") -> None:
        with self._lock:
            node = self._node(node_name)
            if node["status"] in {NODE_SUCCEEDED, NODE_FAILED, NODE_RUNNING}:
                return
            node["status"] = NODE_SKIPPED
            node["finished_at"] = _utc_now()
            node["error"] = reason or None
            self._touch()

    def record_node_activity(
        self,
        node_name: str,
        activity: dict[str, Any],
        *,
        completed: bool = False,
    ) -> None:
        """Record the current or recently completed work inside one graph node."""
        with self._lock:
            node = self._node(node_name)
            normalized = _normalize_activity(activity)
            node["activity"] = normalized
            if completed:
                history = list(node.get("activity_history") or [])
                history.append(normalized)
                node["activity_history"] = history[-MAX_ACTIVITY_HISTORY:]
            self._touch()

    def record_report(self, key: str, path: str | Path) -> None:
        with self._lock:
            report_path = Path(path)
            exists = report_path.exists()
            stat = report_path.stat() if exists else None
            self.status["reports"][key] = {
                "path": str(report_path),
                "exists": exists,
                "bytes": stat.st_size if stat else 0,
                "updated_at": _utc_now(),
            }
            self._touch()

    def add_error(self, error: str, *, flush: bool = True) -> None:
        with self._lock:
            self.status["errors"].append(
                {
                    "message": str(error),
                    "timestamp": _utc_now(),
                }
            )
            if flush:
                self._touch()

    def request_cancel(self) -> bool:
        """Mark the run for cooperative cancellation.

        Returns True when the run is still active and should be interrupted by
        the worker, False when it is already in a terminal state.
        """
        with self._lock:
            if self.status.get("status") in {NODE_SUCCEEDED, NODE_FAILED, NODE_CANCELLED}:
                return False
            if not self.status.get("cancel_requested"):
                self.status["cancel_requested"] = True
                self.status["cancel_requested_at"] = _utc_now()
            self._touch()
            return True

    def is_cancel_requested(self) -> bool:
        with self._lock:
            return bool(self.status.get("cancel_requested"))

    def cancel(self) -> None:
        with self._lock:
            now = _utc_now()
            self.status["status"] = NODE_CANCELLED
            self.status["cancel_requested"] = True
            self.status["cancel_requested_at"] = self.status.get("cancel_requested_at") or now
            self.status["cancelled_at"] = now
            self.status["finished_at"] = now
            current_node = self.status.get("current_node")
            if current_node:
                node = self._node(current_node)
                if node.get("status") == NODE_RUNNING:
                    node["status"] = NODE_SKIPPED
                    node["finished_at"] = now
                    node["error"] = "cancelled"
            self.status["current_node"] = None
            self._touch()

    def finish(self, *, failed: bool = False) -> None:
        with self._lock:
            if self.status.get("status") == NODE_CANCELLED:
                return
            self.status["status"] = NODE_FAILED if failed else NODE_SUCCEEDED
            self.status["finished_at"] = _utc_now()
            self.status["current_node"] = None
            self._touch()

    def write_audit(
        self,
        *,
        state: dict[str, Any] | None,
        error: Exception | None = None,
    ) -> str:
        path = self.output_dir / "run_audit.md"
        self.status["graph_errors"] = [str(item) for item in state.get("errors", [])] if state else []
        self.status["decision_summary"] = _build_decision_summary(state, self.status)
        lines = [
            "# Trading Copilot Run Audit",
            "",
            f"- Run ID: `{self.status['run_id']}`",
            f"- Symbols: `{', '.join(self.status['symbols'])}`",
            f"- Status: `{_node_status_label(self.status['status'])}`",
            f"- Current node: `{self.status.get('current_node') or '-'}`",
            f"- Status file: `{self.status_path}`",
        ]
        if error is not None:
            lines.append(f"- Exception: `{type(error).__name__}: {error}`")

        lines.extend(["", "## Node Progress", ""])
        for node_name, node in self.status["nodes"].items():
            suffix = f" ({node['error']})" if node.get("error") else ""
            lines.append(f"- `{node_name}`: {_node_status_label(node['status'])}{suffix}")

        lines.extend(["", "## Reports", ""])
        if not self.status["reports"]:
            lines.append("- No reports recorded.")
        for key, report in sorted(self.status["reports"].items()):
            lines.append(
                f"- `{key}`: exists={report['exists']} bytes={report['bytes']} "
                f"path=`{report['path']}`"
            )

        lines.extend(["", "## Errors", ""])
        graph_errors = state.get("errors", []) if state else []
        all_errors = [item["message"] for item in self.status["errors"]]
        all_errors.extend(str(item) for item in graph_errors)
        if not all_errors:
            lines.append("- No errors recorded.")
        else:
            for item in all_errors:
                lines.append(f"- `{item}`")

        if state:
            lines.extend(["", "## Agent Inputs And Outputs", ""])
            for name, input_value, output_value, report_key in _agent_io_sections(state):
                report = self.status["reports"].get(report_key, {})
                lines.extend(
                    [
                        f"### {name}",
                        "",
                        f"- Report: `{report.get('path', 'not generated')}`",
                        "",
                        "Input:",
                        "```json",
                        _audit_json(input_value),
                        "```",
                        "Output:",
                        "```json",
                        _audit_json(output_value),
                        "```",
                    ]
                )

            lines.extend(["", "## Final Outputs", "", "```json"])
            final_outputs = {
                "agent_reports": state.get("agent_reports", {}),
                "analyst_reports": state.get("analyst_reports", {}),
                "radar_items": state.get("radar_items", []),
                "technical_positions": state.get("technical_positions", {}),
                "technical_contexts": state.get("technical_contexts", {}),
                "news_sentiment_by_symbol": state.get("news_sentiment_by_symbol", {}),
                "fundamental_analysis_by_symbol": state.get("fundamental_analysis_by_symbol", {}),
                "trade_plans": state.get("trade_plans", {}),
                "risk_assessments": state.get("risk_assessments", {}),
                "execution_decisions": state.get("execution_decisions", {}),
                "risk_challenges": state.get("explanations", {}).get("risk_challenges", {}),
            }
            lines.append(_audit_json(final_outputs))
            lines.append("```")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.record_report("run_audit", path)
        self.status["decision_summary"] = _build_decision_summary(state, self.status)
        self._touch()
        return str(path)

    def flush(self) -> None:
        with self._lock:
            _write_json_atomic(self.status_path, self.status)

    def _touch(self) -> None:
        self.status["updated_at"] = _utc_now()
        self.flush()

    def _node(self, node_name: str) -> dict[str, Any]:
        return self.status["nodes"].setdefault(
            node_name,
            {
                "status": NODE_PENDING,
                "started_at": None,
                "finished_at": None,
                "error": None,
                "activity": None,
                "activity_history": [],
            },
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _node_status_label(status: str) -> str:
    labels = {
        NODE_PENDING: "pending",
        NODE_RUNNING: "running",
        NODE_SUCCEEDED: "succeeded",
        NODE_FAILED: "failed",
        NODE_SKIPPED: "skipped",
        NODE_CANCELLED: "cancelled",
    }
    return labels.get(status, status)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _normalize_activity(activity: dict[str, Any]) -> dict[str, Any]:
    payload = sanitize_value(dict(activity), text_limit=12000)
    payload.setdefault("updated_at", _utc_now())
    return _jsonable(payload)


def _audit_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=True, indent=2)


def _build_decision_summary(
    state: dict[str, Any] | None,
    status: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact UI-facing decision summary from the final graph state."""

    state = state or {}
    status_symbols = [str(symbol) for symbol in status.get("symbols", [])]
    radar_by_symbol = _index_by_symbol(state.get("radar_items", []))
    technical_by_symbol = state.get("technical_positions", {}) or {}
    plan_by_symbol = state.get("trade_plans", {}) or {}
    risk_by_symbol = state.get("risk_assessments", {}) or {}
    execution_by_symbol = state.get("execution_decisions", {}) or {}
    report = state.get("report")
    explanations = _index_by_symbol(_field(report, "symbols", []) or [])

    symbols = _unique_symbols(
        [
            *status_symbols,
            *radar_by_symbol.keys(),
            *technical_by_symbol.keys(),
            *plan_by_symbol.keys(),
            *risk_by_symbol.keys(),
            *execution_by_symbol.keys(),
        ]
    )

    rows = []
    for symbol in symbols:
        radar = radar_by_symbol.get(symbol)
        technical = technical_by_symbol.get(symbol)
        plan = plan_by_symbol.get(symbol)
        risk = risk_by_symbol.get(symbol)
        execution = execution_by_symbol.get(symbol)
        explanation = explanations.get(symbol)
        approved_by_risk = _first_present(
            _field(risk, "approved"),
            _field(execution, "approved_by_risk"),
        )
        execution_status = _value(_field(execution, "status"))
        rows.append(
            {
                "symbol": symbol,
                "status": _value(
                    _first_present(
                        _field(radar, "status"),
                        _field(plan, "subscription_status"),
                        _field(explanation, "status"),
                    )
                ),
                "status_label": _first_present(_field(radar, "status_label"), ""),
                "current_price": _first_present(
                    _field(execution, "current_price"),
                    _field(radar, "current_price"),
                    _field(technical, "current_price"),
                ),
                "support_level": _first_present(
                    _field(radar, "support_level"),
                    _field(plan, "support_level"),
                    _field(technical, "support_level"),
                ),
                "reward_risk_ratio": _first_present(
                    _field(radar, "reward_risk_ratio"),
                    _field(plan, "reward_risk_ratio"),
                    _field(technical, "reward_risk_ratio"),
                ),
                "direction": _value(
                    _first_present(
                        _field(execution, "direction"),
                        _field(plan, "direction"),
                    )
                ),
                "approved_by_risk": approved_by_risk,
                "execution_status": execution_status,
                "execution_message": _first_present(
                    _field(execution, "message"),
                    _field(explanation, "execution_message"),
                    "",
                ),
                "execution_mode": _value(_field(execution, "mode")),
                "action": _field(execution, "action"),
                "quantity": _field(execution, "quantity"),
                "current_weight": _first_present(
                    _field(execution, "current_weight"),
                    _field(risk, "current_position_weight"),
                ),
                "target_weight": _first_present(
                    _field(execution, "target_weight"),
                    _field(risk, "target_weight"),
                ),
                "final_weight": _first_present(
                    _field(execution, "final_weight"),
                    _field(risk, "final_weight"),
                ),
                "delta_weight": _first_present(
                    _field(execution, "delta_weight"),
                    _field(risk, "delta_weight"),
                ),
                "estimated_trade_value": _first_present(
                    _field(execution, "estimated_trade_value"),
                    _field(risk, "estimated_trade_value"),
                ),
                "portfolio_value": _field(risk, "portfolio_value"),
                "pending_broker_order": _field(execution, "pending_broker_order", False),
                "broker_confirmation_required": _field(
                    execution,
                    "broker_confirmation_required",
                    False,
                ),
                "submitted_to_broker": _field(execution, "submitted_to_broker", False),
                "submitted_quantity": _field(execution, "submitted_quantity", 0),
                "broker_order_id": _field(execution, "broker_order_id"),
                "broker_status": _field(execution, "broker_status", ""),
                "broker_message": _field(execution, "broker_message", ""),
                "broker_idempotency_key": _field(execution, "broker_idempotency_key", ""),
                "risk_clamped": _field(risk, "clamped", False),
                "suggested_action": _first_present(
                    _field(radar, "suggested_action"),
                    _field(plan, "entry_logic"),
                    _field(explanation, "summary"),
                    "",
                ),
            }
        )

    node_statuses = [
        _field(node, "status")
        for node in (status.get("nodes", {}) or {}).values()
    ]
    report_count = sum(
        1 for report_item in (status.get("reports", {}) or {}).values()
        if _field(report_item, "exists")
    )
    risk_adjusted = sum(1 for row in rows if row.get("risk_clamped") is True)
    risk_held = sum(1 for row in rows if row.get("approved_by_risk") is False)
    portfolio_decided = sum(
        1 for row in rows if row["execution_status"] == "portfolio_decided"
    )
    pending_broker = sum(1 for row in rows if row.get("pending_broker_order") is True)
    submitted_broker = sum(1 for row in rows if row.get("submitted_to_broker") is True)
    actionable = sum(1 for row in rows if row["status"] == "actionable")
    summary = _first_present(_field(report, "summary"), "")
    if not summary:
        summary = (
            f"Scanned {len(rows)} symbols; {actionable} actionable; "
            f"{portfolio_decided} portfolio decisions."
        )

    market_regime = _field(_field(report, "market_regime"), "regime")
    if market_regime is None:
        for plan in plan_by_symbol.values():
            market_regime = _field(plan, "market_regime")
            if market_regime is not None:
                break

    return {
        "summary": summary,
        "market_regime": _value(market_regime),
        "metrics": {
            "symbol_count": len(rows),
            "actionable_count": actionable,
            "risk_adjusted_count": risk_adjusted,
            "risk_held_count": risk_held,
            "portfolio_decided_count": portfolio_decided,
            "pending_broker_order_count": pending_broker,
            "submitted_broker_order_count": submitted_broker,
            "completed_nodes": sum(
                1 for item in node_statuses if item in {NODE_SUCCEEDED, NODE_SKIPPED}
            ),
            "failed_nodes": sum(1 for item in node_statuses if item == NODE_FAILED),
            "report_count": report_count,
        },
        "symbols": rows,
    }


def _index_by_symbol(items: Any) -> dict[str, Any]:
    if isinstance(items, dict):
        return {str(symbol): item for symbol, item in items.items()}
    indexed = {}
    for item in items or []:
        symbol = _field(item, "symbol")
        if symbol:
            indexed[str(symbol)] = item
    return indexed


def _unique_symbols(symbols: Iterable[Any]) -> list[str]:
    unique = []
    seen = set()
    for symbol in symbols:
        value = str(symbol).strip().upper()
        if value and value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _field(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return _value(value)
    return None


def _value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    return value


def _agent_io_sections(state: dict[str, Any]) -> list[tuple[str, Any, Any, str]]:
    symbol = (state.get("subscription_symbols") or [""])[0]
    radar_item = (state.get("radar_items") or [None])[0]
    technical = state.get("technical_positions", {}).get(symbol)
    technical_context = state.get("technical_contexts", {}).get(symbol)
    news_sentiment = state.get("news_sentiment_by_symbol", {}).get(symbol)
    fundamental = state.get("fundamental_analysis_by_symbol", {}).get(symbol)
    trade_plan = state.get("trade_plans", {}).get(symbol)
    risk_assessment = state.get("risk_assessments", {}).get(symbol)
    execution = state.get("execution_decisions", {}).get(symbol)
    risk_challenge = state.get("explanations", {}).get("risk_challenges", {}).get(symbol)
    return [
        (
            "Futu Portfolio Snapshot",
            {"portfolio_mode": state.get("portfolio_mode")},
            state.get("portfolio"),
            "futu_portfolio",
        ),
        (
            "Technical Position Agent",
            {
                "symbol": symbol,
                "price_history": state.get("price_history_by_symbol", {}).get(symbol),
            },
            {
                "technical_position": technical,
                "technical_context": technical_context,
            },
            "technical_position",
        ),
        (
            "News Sentiment Agent",
            {"symbol": symbol, "look_back_days": state.get("look_back_days")},
            news_sentiment,
            "news_sentiment",
        ),
        (
            "Fundamental Analysis Agent",
            {"symbol": symbol, "fundamental_rag": "available_to_agent_tool"},
            fundamental,
            "fundamental_analysis",
        ),
        (
            "Trader Agent",
            {
                "reviewed_opportunity": radar_item,
                "technical_position": technical,
                "technical_context": technical_context,
                "news_sentiment": news_sentiment,
                "fundamental": fundamental,
                "persona": state.get("persona_config"),
            },
            trade_plan,
            "trader",
        ),
        (
            "Risk Manager",
            {
                "persona": state.get("persona_config"),
                "portfolio": state.get("portfolio"),
                "trade_plan": trade_plan,
                "risk_challenge": risk_challenge,
            },
            risk_assessment,
            "risk_check",
        ),
        (
            "Portfolio Manager",
            {
                "trade_plan": trade_plan,
                "risk_assessment": risk_assessment,
                "mode": state.get("execution_mode"),
                "user_confirmed": state.get("user_confirmed"),
            },
            execution,
            "portfolio_manager",
        ),
        (
            "Run Explanation Agent",
            {
                "radar_items": state.get("radar_items", []),
                "technical_positions": state.get("technical_positions", {}),
                "trade_plans": state.get("trade_plans", {}),
                "risk_assessments": state.get("risk_assessments", {}),
                "execution_decisions": state.get("execution_decisions", {}),
                "risk_challenges": state.get("explanations", {}).get("risk_challenges", {}),
            },
            state.get("report"),
            "run_explanation",
        ),
    ]
