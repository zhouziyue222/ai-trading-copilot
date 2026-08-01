"""CLI for managing the fundamental-only Chroma RAG knowledge base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from ai_trading_copilot.copilot.run import DEFAULT_MEMORY_FILE, create_default_memory_agent


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    agent = create_default_memory_agent(
        args.memory_file,
        rag_chroma_dir=args.chroma_dir,
        auto_ingest_seed=False,
    )

    if args.command == "status":
        print(json.dumps(agent.rag_status(), ensure_ascii=False))
        return 0

    if args.command == "ingest-defaults":
        result = agent.ingest_seed_knowledge()
        print(json.dumps(result, ensure_ascii=False))
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
        print(json.dumps(result, ensure_ascii=False))
        return 0

    if args.command == "ingest-online":
        result = agent.ingest_online_fundamental_research(
            symbols=_split_csv(args.symbols),
            trade_date=args.trade_date,
            look_back_days=args.look_back_days,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0

    if args.command == "query":
        docs = agent.retrieve_fundamental_rag_context(
            symbol=args.symbol,
            tags=_split_csv(args.tags),
            query=args.query,
            limit=args.limit,
        )
        print(json.dumps([doc.model_dump(mode="json") for doc in docs], ensure_ascii=False))
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage AI Trading Copilot fundamental RAG knowledge base.")
    parser.add_argument("--memory-file", type=Path, default=DEFAULT_MEMORY_FILE)
    parser.add_argument("--chroma-dir", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status")
    subparsers.add_parser("ingest-defaults")

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
