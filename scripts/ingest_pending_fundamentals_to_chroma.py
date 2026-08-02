"""Ingest staged raw fundamental filings into the Chroma fundamental RAG store."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT.parent))

from ai_trading_copilot.copilot.services.rag_store import (  # noqa: E402
    DEFAULT_CHROMA_DIR,
    FundamentalRagStore,
    documents_from_text,
)


DEFAULT_MANIFEST = REPO_ROOT / "knowledge" / "fundamentals_pending" / "manifest.csv"
DEFAULT_TAGS = ("fundamentals", "sec_filing")


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = _repo_path(args.manifest)
    chroma_dir = _repo_path(args.chroma_dir) if args.chroma_dir else DEFAULT_CHROMA_DIR
    rows = _load_manifest(manifest_path)
    selected = _select_rows(rows, symbols=args.symbols, limit=args.limit)

    if not selected:
        print({"files": 0, "added": 0, "skipped": 0, "errors": ["no manifest rows selected"]})
        return 1

    store = FundamentalRagStore(chroma_dir)
    total_added = 0
    total_skipped = 0
    errors: list[str] = []
    planned_chunks = 0

    for index, row in enumerate(selected, start=1):
        raw_path = _repo_path(row["raw_path"])
        if not raw_path.exists():
            errors.append(f"{row.get('doc_id') or raw_path}: file not found")
            continue
        text = _extract_text(raw_path)
        if not text:
            errors.append(f"{row.get('doc_id') or raw_path}: empty extracted text")
            continue
        title = _title_for_row(row, raw_path)
        source = row.get("source_url") or str(raw_path)
        tags = [*DEFAULT_TAGS, row["source_type"].strip().lower(), row["symbol"].strip().upper()]
        docs = documents_from_text(
            title=title,
            text=text,
            source=source,
            source_type=row["source_type"],
            symbols=[row["symbol"]],
            tags=tags,
        )
        docs = [_with_manifest_metadata(doc, row, raw_path) for doc in docs]
        planned_chunks += len(docs)

        if args.dry_run:
            print(
                f"[{index}/{len(selected)}] {row['doc_id']} chunks={len(docs)} "
                f"chars={len(text)} dry_run"
            )
            continue

        result = store.upsert_documents(docs)
        total_added += result.added
        total_skipped += result.skipped
        errors.extend(f"{row['doc_id']}: {error}" for error in result.errors)
        print(
            f"[{index}/{len(selected)}] {row['doc_id']} added={result.added} "
            f"skipped={result.skipped} chunks={len(docs)}"
        )

    payload = {
        "files": len(selected),
        "planned_child_chunks": planned_chunks,
        "added": total_added,
        "skipped": total_skipped,
        "errors": errors,
        "chroma_dir": str(chroma_dir),
    }
    print(payload)
    return 1 if errors else 0


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--chroma-dir", type=Path, default=None)
    parser.add_argument("--symbols", default="", help="Comma-separated ticker filter.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def _repo_path(path: Path | str) -> Path:
    item = Path(path)
    return item if item.is_absolute() else REPO_ROOT / item


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _select_rows(rows: list[dict[str, str]], *, symbols: str, limit: int) -> list[dict[str, str]]:
    allowed = {symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip()}
    selected = [
        row
        for row in rows
        if row.get("status") in {"downloaded", "exists"}
        and row.get("source_type") in {"sec_10k", "sec_10q"}
        and (not allowed or row.get("symbol", "").upper() in allowed)
    ]
    if limit > 0:
        return selected[:limit]
    return selected


def _extract_text(path: Path) -> str:
    raw = path.read_bytes()
    soup = BeautifulSoup(raw, "html.parser")
    for node in soup(["script", "style", "noscript"]):
        node.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _title_for_row(row: dict[str, str], raw_path: Path) -> str:
    chinese_name = _chinese_name_from_path(raw_path)
    pieces = [
        row.get("symbol", "").strip().upper(),
        chinese_name,
        row.get("company_name", "").strip(),
        row.get("source_type", "").strip().upper(),
        row.get("fiscal_period", "").strip(),
    ]
    return " ".join(piece for piece in pieces if piece)


def _chinese_name_from_path(raw_path: Path) -> str:
    parts = raw_path.stem.split("_")
    return parts[1] if len(parts) >= 2 else ""


def _with_manifest_metadata(doc, row: dict[str, str], raw_path: Path):
    metadata = dict(doc.metadata)
    metadata.update(
        {
            "doc_id": row.get("doc_id", ""),
            "company_name": row.get("company_name", ""),
            "company_name_zh": _chinese_name_from_path(raw_path),
            "source_date": row.get("source_date", ""),
            "fiscal_period": row.get("fiscal_period", ""),
            "source_url": row.get("source_url", ""),
            "raw_path": str(raw_path),
            "raw_format": row.get("raw_format", ""),
            "filing_type": row.get("source_type", doc.source_type),
            "symbol": row.get("symbol", "").upper(),
        }
    )
    return doc.model_copy(update={"metadata": metadata})


if __name__ == "__main__":
    raise SystemExit(main())
