"""Command-line entrypoint for a tracked copilot run."""

from __future__ import annotations

import argparse
import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from ai_trading_copilot.copilot.agents import PostTradeReviewLearningAgent
from ai_trading_copilot.copilot.config import DEFAULT_REPORT_OUTPUT_DIR
from ai_trading_copilot.copilot.config.defaults import DEFAULT_PERSONA_CONFIG
from ai_trading_copilot.copilot.domain.enums import AnalystType, ExecutionMode
from ai_trading_copilot.copilot.graph import CopilotLangGraph
from ai_trading_copilot.copilot.services.memory_store import DistilledMemoryStore
from ai_trading_copilot.copilot.services.rag_store import FundamentalRagStore
from ai_trading_copilot.copilot.services.run_tracker import RunTracker
from ai_trading_copilot.copilot.services.subscription_service import SubscriptionStore


DEFAULT_SUBSCRIPTIONS_FILE = Path(__file__).resolve().parents[1] / "config" / "subscriptions.json"
DEFAULT_MEMORY_FILE = Path(__file__).resolve().parents[1] / "config" / "memory.jsonl"
DEFAULT_ANALYSTS = [
    AnalystType.OPPORTUNITY_RADAR,
    AnalystType.TECHNICAL_POSITION,
    AnalystType.FUNDAMENTAL_NEWS,
]
ALL_ANALYST_VALUES = [analyst.value for analyst in DEFAULT_ANALYSTS]


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    symbols = _resolve_symbols(args)
    selected_analysts = _resolve_analysts(args)
    run_label = symbols[0] if len(symbols) == 1 else "BATCH"
    run_id = args.run_id or f"run_{run_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir or DEFAULT_REPORT_OUTPUT_DIR / run_id)
    params = {
        "subscription_symbols": symbols,
        "selected_analysts": selected_analysts,
        "price_history_by_symbol": None,
        "fundamental_news_by_symbol": None,
        "portfolio": None,
        "report_output_dir": str(output_dir),
        "trade_date": args.trade_date,
        "look_back_days": args.look_back_days,
        "mode": ExecutionMode(args.mode),
        "portfolio_mode": ExecutionMode(args.portfolio_mode),
        "user_confirmed": False,
    }
    defaults = {
        "persona": DEFAULT_PERSONA_CONFIG,
        "selected_analysts": [analyst.value for analyst in selected_analysts],
        "execution_mode_default": ExecutionMode.SIMULATION.value,
        "portfolio_mode_default": ExecutionMode.SIMULATION.value,
        "look_back_days_default": 90,
    }
    tracker = RunTracker(
        output_dir=output_dir,
        run_id=run_id,
        symbols=symbols,
        params=params,
        defaults=defaults,
        nodes=CopilotLangGraph.NODE_ORDER,
    )
    graph = _create_graph(
        CopilotLangGraph,
        run_tracker=tracker,
        memory_agent=create_default_memory_agent(),
    )
    state = None
    error = None
    try:
        state = graph.run(**params)
    except Exception as exc:  # pragma: no cover - exercised by integration style tests
        error = exc
        tracker.add_error(str(exc))
    finally:
        tracker.finish(failed=error is not None)
        audit_path = tracker.write_audit(state=state, error=error)

    print(json.dumps({"output_dir": str(output_dir), "audit_path": audit_path}, ensure_ascii=False))
    if error is not None:
        raise error
    return 0


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tracked AI trading copilot analysis.")
    parser.add_argument(
        "symbols",
        nargs="*",
        help="Subscribed symbols to analyze, e.g. CRCL or AAPL MSFT CRCL.",
    )
    parser.add_argument(
        "--symbols",
        dest="symbol_list",
        help="Comma-separated subscribed symbols to analyze, e.g. AAPL,MSFT,CRCL.",
    )
    parser.add_argument(
        "--subscriptions-file",
        default=str(DEFAULT_SUBSCRIPTIONS_FILE),
        help="JSON subscription file used for interactive selection.",
    )
    symbol_group = parser.add_mutually_exclusive_group()
    symbol_group.add_argument(
        "--select-symbols",
        action="store_true",
        help="Interactively select symbols from the subscription file.",
    )
    symbol_group.add_argument(
        "--all-subscribed",
        action="store_true",
        help="Analyze all symbols from the subscription file.",
    )
    parser.add_argument(
        "--analysts",
        type=_parse_analyst_arg,
        default="all",
        metavar="ANALYSTS",
        help=(
            "Comma-separated analysts to run, or 'all'. Choices: "
            + ", ".join(ALL_ANALYST_VALUES)
        ),
    )
    parser.add_argument(
        "--select-analysts",
        action="store_true",
        help="Interactively select analyst modules.",
    )
    parser.add_argument("--output-dir", help="Report output directory. Defaults to reports/run_SYMBOL_timestamp.")
    parser.add_argument("--run-id", help="Optional stable run id.")
    parser.add_argument("--trade-date", help="YYYY-MM-DD trade date. Defaults to today.")
    parser.add_argument("--look-back-days", type=int, default=90)
    parser.add_argument(
        "--mode",
        choices=[ExecutionMode.SIMULATION.value, ExecutionMode.LIVE.value],
        default=ExecutionMode.SIMULATION.value,
    )
    parser.add_argument(
        "--portfolio-mode",
        choices=[ExecutionMode.SIMULATION.value, ExecutionMode.LIVE.value],
        default=ExecutionMode.SIMULATION.value,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_symbols(args: argparse.Namespace) -> List[str]:
    explicit_symbols = _unique_symbols(
        [
            *args.symbols,
            *_parse_symbol_list(args.symbol_list),
        ]
    )
    subscriptions_file = Path(args.subscriptions_file)

    if args.all_subscribed:
        return _require_symbols(
            _load_subscription_symbols(subscriptions_file),
            f"No subscriptions found in {subscriptions_file}.",
        )

    if args.select_symbols:
        return _select_symbols_from_subscription_file(subscriptions_file)

    if explicit_symbols:
        return explicit_symbols

    loaded_symbols = _load_subscription_symbols(subscriptions_file)
    if loaded_symbols:
        return _select_symbols(loaded_symbols)

    raise SystemExit(
        "No symbols provided and no subscriptions were found. "
        "Pass symbols directly, use --symbols, or create a subscription file."
    )


def _resolve_analysts(args: argparse.Namespace) -> List[AnalystType]:
    configured = DEFAULT_ANALYSTS if args.analysts == "all" else list(args.analysts)
    if args.select_analysts:
        return _select_analysts(configured)
    return configured


def _parse_symbol_list(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _unique_symbols(values: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for value in values:
        symbol = value.strip().upper()
        if symbol and symbol not in seen:
            normalized.append(symbol)
            seen.add(symbol)
    return normalized


def _parse_analyst_arg(value: str) -> str | List[AnalystType]:
    cleaned = value.strip().lower()
    if cleaned == "all":
        return "all"

    analysts: List[AnalystType] = []
    for raw_item in cleaned.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            analyst = AnalystType(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid analyst '{item}'. Choices: all, {', '.join(ALL_ANALYST_VALUES)}"
            ) from exc
        if analyst not in analysts:
            analysts.append(analyst)

    if not analysts:
        raise argparse.ArgumentTypeError("--analysts must include at least one analyst or 'all'.")
    return analysts


def _load_subscription_symbols(path: Path) -> List[str]:
    if not path.exists():
        return []
    return SubscriptionStore(path).load().symbols


def _select_symbols_from_subscription_file(path: Path) -> List[str]:
    return _require_symbols(
        _select_symbols(
            _require_symbols(
                _load_subscription_symbols(path),
                f"No subscriptions found in {path}.",
            )
        ),
        "No symbols selected.",
    )


def _select_symbols(symbols: List[str]) -> List[str]:
    questionary = _load_questionary()
    selected = questionary.checkbox(
        "Select subscribed symbols to analyze:",
        choices=symbols,
    ).ask()
    return _require_symbols(selected or [], "No symbols selected.")


def _select_analysts(defaults: List[AnalystType]) -> List[AnalystType]:
    questionary = _load_questionary()
    selected = questionary.checkbox(
        "Select analysts to run:",
        choices=[
            questionary.Choice(
                title=value,
                value=value,
                checked=value in {analyst.value for analyst in defaults},
            )
            for value in ALL_ANALYST_VALUES
        ],
    ).ask()
    if not selected:
        raise SystemExit("No analysts selected.")
    return [AnalystType(value) for value in selected]


def _load_questionary():
    try:
        import questionary
    except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject
        raise SystemExit("Interactive selection requires the questionary package.") from exc
    return questionary


def _require_symbols(symbols: Iterable[str], message: str) -> List[str]:
    normalized = _unique_symbols(symbols)
    if not normalized:
        raise SystemExit(message)
    return normalized


def create_default_memory_agent(
    path: str | Path = DEFAULT_MEMORY_FILE,
    *,
    rag_chroma_dir: str | Path | None = None,
    auto_ingest_seed: bool = True,
) -> PostTradeReviewLearningAgent:
    memory_path = Path(path)
    rag_store = FundamentalRagStore(rag_chroma_dir or memory_path.parent / "rag_chroma")
    agent = PostTradeReviewLearningAgent(
        DistilledMemoryStore(memory_path),
        rag_store=rag_store,
    )
    if auto_ingest_seed:
        agent.ingest_seed_knowledge()
    return agent


def _create_graph(graph_cls: type, *, run_tracker: RunTracker, memory_agent):
    if _accepts_keyword(graph_cls, "memory_agent"):
        return graph_cls(run_tracker=run_tracker, memory_agent=memory_agent)
    return graph_cls(run_tracker=run_tracker)


def _accepts_keyword(callable_obj, name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == name
        for parameter in signature.parameters.values()
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
