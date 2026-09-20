"""Administration of durable delayed trade reviews."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai_trading_copilot.copilot.run import DEFAULT_MEMORY_DATABASE


def create_review_service(database_path):
    # Keep help/imports usable without initializing providers or the model.
    from ai_trading_copilot.copilot.services.delayed_review import create_review_service as factory
    return factory(database_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-database", type=Path, default=DEFAULT_MEMORY_DATABASE)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "run-due", "fills"):
        commands.add_parser(name)
    for name in ("show", "retry"):
        commands.add_parser(name).add_argument("review_id")
    commands.add_parser("import-snapshot").add_argument("file", type=Path)
    associate = commands.add_parser("associate")
    for name in ("environment", "account-id", "deal-id", "run-id", "symbol"):
        associate.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    try:
        service = create_review_service(args.memory_database)
        repository = service.repository
        if args.command == "list":
            result = {"items": repository.list_reviews()}
        elif args.command == "fills":
            result = {"items": repository.list_fills()}
        elif args.command == "run-due":
            result = service.run_due()
        elif args.command == "show":
            result = repository.detail(args.review_id)
            if result is None:
                raise ValueError("Review not found")
        elif args.command == "retry":
            if not repository.retry(args.review_id):
                raise ValueError("Review not found or cannot be retried")
            result = {"retried": True}
        elif args.command == "associate":
            repository.associate_fill(args.environment, args.account_id, args.deal_id, args.run_id, args.symbol)
            result = {"associated": True}
        else:
            state = json.loads(args.file.read_text(encoding="utf-8-sig"))
            if not isinstance(state, dict):
                raise ValueError("Snapshot must be a complete state object")
            required = {"run_id", "generated_at", "trade_plans", "risk_assessments", "execution_decisions"}
            if not required.issubset(state) or not (state.get("market_reports_by_symbol") or state.get("price_history_by_symbol")):
                raise ValueError("Complete snapshot requires run_id, generated_at, plans, risks, decisions and original market evidence")
            from ai_trading_copilot.copilot.domain.models import TradePlan
            from ai_trading_copilot.copilot.services.review_repository import timestamp
            timestamp(state["generated_at"])
            plans = state["trade_plans"]
            if isinstance(plans, dict):
                plans = list(plans.values())
            if not isinstance(plans, list) or not plans:
                raise ValueError("Snapshot must contain complete trade plans")
            for plan in plans:
                TradePlan.model_validate(plan)
            if any(not isinstance(state[key], (dict, list)) for key in ("risk_assessments", "execution_decisions")):
                raise ValueError("Snapshot risk assessments and decisions must be structured")
            result = {"registered": service.register_run(state)}
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
