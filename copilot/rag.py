# -*- coding: utf-8 -*-
"""CLI for managing the fundamental-only Chroma RAG knowledge base."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

from ai_trading_copilot.copilot.run import create_default_fundamental_research_retriever
from ai_trading_copilot.copilot.services.rag_store import DEFAULT_CHROMA_DIR


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    chroma_dir = args.chroma_dir

    if args.command == "_probe-status-worker":
        agent = create_default_fundamental_research_retriever(
            rag_chroma_dir=chroma_dir,
            auto_ingest_seed=False,
        )
        _print_json(agent.rag_status(probe=True))
        return 0

    if args.command == "status" and args.probe:
        _print_json(_probe_status_subprocess(chroma_dir))
        return 0

    agent = create_default_fundamental_research_retriever(
        rag_chroma_dir=chroma_dir,
        auto_ingest_seed=False,
    )

    if args.command == "status":
        _print_json(agent.rag_status(probe=False))
        return 0

    if args.command == "ingest-defaults":
        result = agent.ingest_seed_knowledge()
        _print_json(result)
        return 0

    if args.command == "ingest-text":
        text = Path(args.path).read_text(encoding="utf-8")
        result = agent.ingest_text_knowledge(
            title=args.title or Path(args.path).stem,
            text=text,
            source_type=args.source_type,
            source=str(Path(args.path)),
            symbols=_split_csv(args.symbols),
            tags=_split_csv(args.tags),
        )
        _print_json(result)
        return 0

    if args.command == "ingest-online":
        result = agent.ingest_online_fundamental_research(
            symbols=_split_csv(args.symbols),
            trade_date=args.trade_date,
            look_back_days=args.look_back_days,
        )
        _print_json(result)
        return 0

    if args.command == "query":
        docs = agent.retrieve_fundamental_rag_context(
            symbol=args.symbol,
            tags=_split_csv(args.tags),
            query=args.query,
            limit=args.limit,
        )
        _print_json([doc.model_dump(mode="json") for doc in docs])
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


def _probe_status_subprocess(chroma_dir: Path) -> dict:
    command = [
        sys.executable, "-X", "utf8",
        "-m",
        "ai_trading_copilot.copilot.rag",
        "--chroma-dir",
        str(chroma_dir),
        "_probe-status-worker",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "backend": "fundamental_chroma",
            "available": False,
            "probe_status": "failed",
            "collection": "ai_trading_copilot_fundamentals",
            "path": str(chroma_dir),
            "document_count": None,
            "error": f"RAG probe timed out after {exc.timeout} seconds.",
        }
    if completed.returncode != 0:
        return {
            "backend": "fundamental_chroma",
            "available": False,
            "probe_status": "failed",
            "collection": "ai_trading_copilot_fundamentals",
            "path": str(chroma_dir),
            "document_count": None,
            "error": (
                f"RAG probe worker exited with code {completed.returncode}. "
                f"{(completed.stderr or completed.stdout).strip()}"
            ).strip(),
        }
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {
            "backend": "fundamental_chroma",
            "available": False,
            "probe_status": "failed",
            "collection": "ai_trading_copilot_fundamentals",
            "path": str(chroma_dir),
            "document_count": None,
            "error": f"RAG probe returned invalid JSON: {exc}",
        }


def _print_json(payload) -> None:
    try:
        print(json.dumps(payload, ensure_ascii=False))
    except UnicodeEncodeError:
        print(json.dumps(payload, ensure_ascii=True))


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage AI Trading Copilot fundamental RAG knowledge base.")
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--probe", action="store_true", help="Open Chroma in a worker process and count documents.")
    subparsers.add_parser("ingest-defaults")
    subparsers.add_parser("_probe-status-worker", help=argparse.SUPPRESS)

    ingest_text = subparsers.add_parser("ingest-text")
    ingest_text.add_argument("path")
    ingest_text.add_argument("--title", default="")
    ingest_text.add_argument("--source-type", default="fundamental_research_note")
    ingest_text.add_argument("--symbols", default="")
    ingest_text.add_argument("--tags", default="manual,fundamentals,fundamental_research_note")

    ingest_online = subparsers.add_parser("ingest-online")
    ingest_online.add_argument("--symbols", required=True)
    ingest_online.add_argument("--trade-date", default=None)
    ingest_online.add_argument("--look-back-days", type=int, default=7)

    query = subparsers.add_parser("query", description="Debug query for the fundamental RAG index.")
    query.add_argument("--query", required=True)
    query.add_argument("--symbol", default=None)
    query.add_argument("--tags", default="")
    query.add_argument("--limit", type=int, default=5)

    return parser.parse_args(list(argv) if argv is not None else None)


def _split_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
