"""Persistent run tracking for copilot graph executions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Iterable


NODE_PENDING = "pending"
NODE_RUNNING = "running"
NODE_SUCCEEDED = "succeeded"
NODE_FAILED = "failed"
NODE_SKIPPED = "skipped"


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
            "started_at": now,
            "updated_at": now,
            "finished_at": None,
            "nodes": {
                node: {
                    "status": NODE_PENDING,
                    "started_at": None,
                    "finished_at": None,
                    "error": None,
                }
                for node in nodes
            },
            "reports": {},
            "errors": [],
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

    def finish(self, *, failed: bool = False) -> None:
        with self._lock:
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
        self.status["decision_summary"] = _build_decision_summary(state, self.status)
        lines = [
            "# 交易助手运行审计",
            "",
            f"- 运行 ID：`{self.status['run_id']}`",
            f"- 标的：`{', '.join(self.status['symbols'])}`",
            f"- 状态：`{_zh_node_status(self.status['status'])}`",
            f"- 当前节点：`{self.status.get('current_node') or '-'}`",
            f"- 状态文件：`{self.status_path}`",
        ]
        if error is not None:
            lines.append(f"- 异常：`{type(error).__name__}: {error}`")

        lines.extend(["", "## 节点进度", ""])
        for node_name, node in self.status["nodes"].items():
            suffix = f" ({node['error']})" if node.get("error") else ""
            lines.append(f"- `{node_name}`：{_zh_node_status(node['status'])}{suffix}")

        lines.extend(["", "## 报告", ""])
        if not self.status["reports"]:
            lines.append("- 未记录报告。")
        for key, report in sorted(self.status["reports"].items()):
            lines.append(
                f"- `{key}`：存在={report['exists']} 字节={report['bytes']} "
                f"路径=`{report['path']}`"
            )

        lines.extend(["", "## 错误", ""])
        graph_errors = state.get("errors", []) if state else []
        all_errors = [item["message"] for item in self.status["errors"]]
        all_errors.extend(str(item) for item in graph_errors)
        if not all_errors:
            lines.append("- 未记录错误。")
        else:
            for item in all_errors:
                lines.append(f"- `{item}`")

        if state:
            lines.extend(["", "## 智能体输入与输出", ""])
            for name, input_value, output_value, report_key in _agent_io_sections(state):
                report = self.status["reports"].get(report_key, {})
                lines.extend(
                    [
                        f"### {name}",
                        "",
                        f"- 报告：`{report.get('path', '未生成')}`",
                        "",
                        "输入：",
                        "```json",
                        json.dumps(_jsonable(input_value), ensure_ascii=False, indent=2),
                        "```",
                        "输出：",
                        "```json",
                        json.dumps(_jsonable(output_value), ensure_ascii=False, indent=2),
                        "```",
                    ]
                )

            lines.extend(["", "## 最终输出", "", "```json"])
            final_outputs = {
                "agent_reports": state.get("agent_reports", {}),
                "analyst_reports": state.get("analyst_reports", {}),
                "radar_items": state.get("radar_items", []),
                "technical_positions": state.get("technical_positions", {}),
                "fundamental_news_by_symbol": state.get("fundamental_news_by_symbol", {}),
                "trade_plans": state.get("trade_plans", {}),
                "risk_assessments": state.get("risk_assessments", {}),
                "execution_decisions": state.get("execution_decisions", {}),
                "risk_challenges": state.get("explanations", {}).get("risk_challenges", {}),
            }
            lines.append(json.dumps(_jsonable(final_outputs), ensure_ascii=False, indent=2))
            lines.append("```")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.record_report("run_audit", path)
        self.status["decision_summary"] = _build_decision_summary(state, self.status)
        self._touch()
        return str(path)

    def flush(self) -> None:
        with self._lock:
            self.status_path.write_text(
                json.dumps(self.status, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

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
            },
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _zh_node_status(status: str) -> str:
    labels = {
        NODE_PENDING: "待执行",
        NODE_RUNNING: "运行中",
        NODE_SUCCEEDED: "成功",
        NODE_FAILED: "失败",
        NODE_SKIPPED: "已跳过",
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
                        _field(plan, "direction"),
                        _field(execution, "direction"),
                    )
                ),
                "approved_by_risk": approved_by_risk,
                "execution_status": execution_status,
                "execution_message": _first_present(
                    _field(execution, "message"),
                    _field(explanation, "execution_message"),
                    "",
                ),
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
    risk_blocked = sum(
        1
        for row in rows
        if row["approved_by_risk"] is False
        or row["execution_status"] == "blocked_by_risk"
    )
    actionable = sum(1 for row in rows if row["status"] == "actionable")
    summary = _first_present(_field(report, "summary"), "")
    if not summary:
        summary = f"已扫描 {len(rows)} 个标的，{actionable} 个可行动，{risk_blocked} 个风险阻断。"

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
            "risk_blocked_count": risk_blocked,
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
    fundamental = state.get("fundamental_news_by_symbol", {}).get(symbol)
    trade_plan = state.get("trade_plans", {}).get(symbol)
    risk_assessment = state.get("risk_assessments", {}).get(symbol)
    execution = state.get("execution_decisions", {}).get(symbol)
    risk_challenge = state.get("explanations", {}).get("risk_challenges", {}).get(symbol)
    return [
        (
            "富途组合快照",
            {"portfolio_mode": state.get("portfolio_mode")},
            state.get("portfolio"),
            "futu_portfolio",
        ),
        (
            "机会雷达智能体",
            {
                "symbol": symbol,
                "price_history": state.get("price_history_by_symbol", {}).get(symbol),
                "look_back_days": state.get("look_back_days"),
                "trade_date": state.get("trade_date"),
            },
            radar_item,
            "opportunity_radar",
        ),
        (
            "技术位置智能体",
            {
                "symbol": symbol,
                "price_history": state.get("price_history_by_symbol", {}).get(symbol),
            },
            technical,
            "technical_position",
        ),
        (
            "基本面/新闻智能体",
            {"symbol": symbol, "provided_report": state.get("fundamental_news_by_symbol", {}).get(symbol)},
            fundamental,
            "fundamental_news",
        ),
        (
            "机会复核管理器",
            {
                "radar_item": radar_item,
                "technical_position": technical,
                "fundamental_news": fundamental,
            },
            radar_item,
            "opportunity_review",
        ),
        (
            "交易员智能体",
            {
                "reviewed_opportunity": radar_item,
                "technical_position": technical,
                "persona": state.get("persona_config"),
            },
            trade_plan,
            "trader",
        ),
        (
            "风控智能体",
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
            "执行提醒管理器",
            {
                "trade_plan": trade_plan,
                "risk_assessment": risk_assessment,
                "mode": state.get("execution_mode"),
                "user_confirmed": state.get("user_confirmed"),
            },
            execution,
            "execution_alert",
        ),
        (
            "运行解释智能体",
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
