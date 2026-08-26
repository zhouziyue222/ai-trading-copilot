"""Safety-gated Git promotion and rollback for harness prompt versions."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from typing import Iterable

from ai_trading_copilot.copilot.harness_benchmark import PROJECT_ROOT, run_benchmark


EDITABLE_SURFACE = "config/prompts"


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "benchmark":
        return _print(run_benchmark())
    if args.command == "promote":
        payload = _require_passing_benchmark()
        changed = _changed_prompt_files()
        if not changed:
            raise SystemExit("no prompt changes to promote")
        commit = _commit_prompt_version(args.message or _default_message("promote"))
        return _print({"commit": commit, "changed": changed, "benchmark": payload})
    if args.command == "rollback":
        if _changed_prompt_files():
            raise SystemExit("prompt files have uncommitted changes; commit or stash them first")
        target = _git("rev-parse", "--verify", f"{args.commit}^{{commit}}")
        _git("restore", "--source", target, "--", EDITABLE_SURFACE)
        try:
            payload = _require_passing_benchmark()
        except BaseException:
            _git("restore", "--source", "HEAD", "--", EDITABLE_SURFACE)
            raise
        if not _changed_prompt_files():
            return _print({"commit": _git("rev-parse", "HEAD"), "unchanged": True})
        commit = _commit_prompt_version(
            args.message or _default_message(f"rollback-to-{target[:12]}")
        )
        return _print({"commit": commit, "restored_from": target, "benchmark": payload})
    raise SystemExit(f"unknown command: {args.command}")


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark, promote, or roll back bounded harness prompt versions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("benchmark")
    promote = subparsers.add_parser("promote")
    promote.add_argument("--message")
    rollback = subparsers.add_parser("rollback")
    rollback.add_argument("commit")
    rollback.add_argument("--message")
    return parser.parse_args(list(argv) if argv is not None else None)


def _require_passing_benchmark() -> dict:
    payload = run_benchmark()
    hard_gates = {
        "pass_rate": 1.0,
        "safety_gate_pass_rate": 1.0,
        "memory_isolation_rate": 1.0,
        "prompt_contract_rate": 1.0,
    }
    failures = [
        f"{name}={payload.get(name)} (required {minimum})"
        for name, minimum in hard_gates.items()
        if float(payload.get(name, 0.0)) < minimum
    ]
    if failures:
        raise SystemExit("harness promotion blocked: " + "; ".join(failures))
    return payload


def _changed_prompt_files() -> list[str]:
    output = _git("status", "--porcelain", "--", EDITABLE_SURFACE)
    return [line[3:] for line in output.splitlines() if len(line) > 3]


def _commit_prompt_version(message: str) -> str:
    # A path-scoped commit preserves unrelated staged and unstaged user work.
    _git("add", "--", EDITABLE_SURFACE)
    _git("commit", "-m", message, "--", EDITABLE_SURFACE)
    return _git("rev-parse", "HEAD")


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit((completed.stderr or completed.stdout).strip())
    return completed.stdout.strip()


def _default_message(action: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"harness: {action} memory prompts {stamp}"


def _print(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
