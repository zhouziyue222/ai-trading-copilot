"""Command-line administration for the versioned memory system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from ai_trading_copilot.copilot.domain import MemoryStatus, RunOutcome
from ai_trading_copilot.copilot.run import (
    DEFAULT_MEMORY_DATABASE,
    create_default_memory_agent,
)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    agent = create_default_memory_agent(Path(args.memory_database))
    repository = agent.store.repository

    if args.command == "status":
        return _print({"database": str(repository.path), "counts": repository.counts_by_status()})
    if args.command == "list":
        statuses = [MemoryStatus(args.status)] if args.status else None
        items = agent.list_memories(statuses=statuses, symbol=args.symbol)
        return _print({"items": [item.model_dump(mode="json") for item in items]})
    if args.command == "versions":
        return _print(
            {
                "items": [
                    item.model_dump(mode="json")
                    for item in repository.versions(args.memory_id)
                ]
            }
        )
    if args.command == "transition":
        target = MemoryStatus(args.status)
        if target == MemoryStatus.APPROVED:
            if agent.evaluator is None:
                raise SystemExit("memory evaluator is unavailable")
            item = agent.evaluator.promote(
                args.memory_id,
                actor=args.actor,
                reason=args.reason,
            )
        else:
            item = agent.transition(
                args.memory_id,
                target,
                actor=args.actor,
                reason=args.reason,
            )
        return _print({"item": item.model_dump(mode="json")})
    if args.command == "evaluate":
        if agent.evaluator is None:
            raise SystemExit("memory evaluator is unavailable")
        evaluation = agent.evaluator.evaluate(args.memory_id)
        return _print({"evaluation": evaluation.model_dump(mode="json")})
    if args.command == "rollback":
        item = repository.rollback(
            args.memory_id,
            args.version,
            actor=args.actor,
            reason=args.reason,
        )
        return _print({"item": item.model_dump(mode="json")})
    if args.command == "outcome":
        outcome = RunOutcome(
            run_id=args.run_id,
            symbol=args.symbol,
            horizon_days=args.horizon_days,
            realized_return=args.realized_return,
            benchmark_return=args.benchmark_return,
            max_drawdown=args.max_drawdown,
            max_favorable_excursion=args.max_favorable_excursion,
            stopped_out=args.stopped_out,
            source=args.source,
            notes=args.notes,
        )
        return _print({"outcome": agent.record_outcome(outcome).model_dump(mode="json")})
    raise SystemExit(f"unknown command: {args.command}")


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage versioned copilot memories.")
    parser.add_argument("--memory-database", default=str(DEFAULT_MEMORY_DATABASE))
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status")
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--status", choices=[item.value for item in MemoryStatus])
    list_parser.add_argument("--symbol")

    versions = subparsers.add_parser("versions")
    versions.add_argument("memory_id")

    transition = subparsers.add_parser("transition")
    transition.add_argument("memory_id")
    transition.add_argument("status", choices=[item.value for item in MemoryStatus])
    _review_arguments(transition, default_reason="manual_cli_review")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("memory_id")

    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("memory_id")
    rollback.add_argument("version", type=int)
    _review_arguments(rollback, default_reason="manual_cli_rollback")

    outcome = subparsers.add_parser("outcome")
    outcome.add_argument("run_id")
    outcome.add_argument("symbol")
    outcome.add_argument("--horizon-days", type=int, default=5)
    outcome.add_argument("--realized-return", type=float)
    outcome.add_argument("--benchmark-return", type=float)
    outcome.add_argument("--max-drawdown", type=float)
    outcome.add_argument("--max-favorable-excursion", type=float)
    outcome.add_argument("--stopped-out", action=argparse.BooleanOptionalAction)
    outcome.add_argument("--source", default="manual_cli")
    outcome.add_argument("--notes", default="")

    return parser.parse_args(list(argv) if argv is not None else None)


def _review_arguments(parser: argparse.ArgumentParser, *, default_reason: str) -> None:
    parser.add_argument("--actor", default="local_cli_user")
    parser.add_argument("--reason", default=default_reason)


def _print(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
