"""Normalize staged SEC filings into an embedding-free fundamental corpus.

This script reads raw files from ``knowledge/fundamentals_pending`` and writes
reviewable Markdown contexts into ``knowledge/fundamentals``. It does not create
embeddings and does not touch Chroma.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - stdlib fallback is tested through CLI use.
    BeautifulSoup = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "knowledge" / "fundamentals_pending" / "manifest.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "knowledge" / "fundamentals"
DEFAULT_EVAL_CASES = REPO_ROOT / "config" / "rag_eval_fundamentals_seed.json"
ALLOWED_SOURCE_TYPES = {"sec_10k", "sec_10q"}
DEFAULT_SYMBOLS = "MU,NVDA,AMD,AVGO,INTC,QCOM,TXN"
DEFAULT_METRICS = "profitability,cash_flow,balance_sheet"


METRIC_PROFILES = {
    "profitability": {
        "title": "Profitability",
        "positive_use": "questions about company-specific revenue, gross margin, operating income, net income, EPS, or profitability trend.",
        "keywords": (
            "revenue",
            "revenues",
            "net sales",
            "gross margin",
            "gross profit",
            "operating income",
            "operating loss",
            "net income",
            "net loss",
            "earnings per share",
            "diluted",
        ),
    },
    "cash_flow": {
        "title": "Cash Flow",
        "positive_use": "questions about operating cash flow, investing cash flow, financing cash flow, capital expenditures, or free-cash-flow pressure.",
        "keywords": (
            "cash flows",
            "net cash provided by operating activities",
            "net cash used in operating activities",
            "operating activities",
            "investing activities",
            "financing activities",
            "capital expenditures",
            "property and equipment",
            "free cash flow",
        ),
    },
    "balance_sheet": {
        "title": "Balance Sheet",
        "positive_use": "questions about liquidity, cash, assets, liabilities, debt, equity, inventories, or balance-sheet strength.",
        "keywords": (
            "balance sheet",
            "cash and cash equivalents",
            "short-term investments",
            "total assets",
            "inventories",
            "accounts receivable",
            "total liabilities",
            "debt",
            "stockholders' equity",
        ),
    },
    "risk_factors": {
        "title": "Risk Factors",
        "positive_use": "questions about SEC-disclosed business, market, operational, regulatory, supply-chain, or customer concentration risks.",
        "keywords": (
            "risk factors",
            "may adversely affect",
            "could adversely affect",
            "supply chain",
            "customer concentration",
            "competition",
            "regulatory",
            "geopolitical",
        ),
    },
}


@dataclass(frozen=True)
class ContextRecord:
    context_id: str
    doc_id: str
    symbol: str
    company_name: str
    source_type: str
    source_date: str
    fiscal_period: str
    metric_type: str
    title: str
    source_url: str
    raw_path: str
    markdown_path: str
    text: str
    snippet_count: int


class _SecHtmlTextExtractor(HTMLParser):
    _BLOCK_TAGS = {
        "address",
        "article",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "p",
        "section",
        "table",
        "td",
        "th",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        value = unescape(data)
        if value.strip():
            self.parts.append(value)

    def text(self) -> str:
        return _normalize_text(" ".join(self.parts))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = _select_rows(
        _load_manifest(_repo_path(args.manifest)),
        symbols=_split_csv(args.symbols),
        per_symbol=args.per_symbol,
        limit=args.limit,
    )
    metrics = _split_csv(args.metrics)
    output_dir = _repo_path(args.output_dir)
    contexts: list[ContextRecord] = []
    errors: list[str] = []

    for row in rows:
        raw_path = _repo_path(row["raw_path"])
        if not raw_path.exists():
            errors.append(f"{row.get('doc_id')}: raw file not found: {raw_path}")
            continue
        text = _extract_raw_text(raw_path)
        if not text:
            errors.append(f"{row.get('doc_id')}: no text extracted")
            continue
        for metric in metrics:
            profile = METRIC_PROFILES.get(metric)
            if profile is None:
                errors.append(f"{row.get('doc_id')}: unknown metric_type {metric}")
                continue
            snippets = _snippets_for_metric(
                text=text,
                keywords=profile["keywords"],
                max_snippets=args.max_snippets,
                max_chars=args.max_snippet_chars,
            )
            if len(snippets) < args.min_snippets:
                errors.append(
                    f"{row.get('doc_id')} {metric}: only {len(snippets)} snippets found"
                )
                continue
            context = _context_record(row, raw_path, output_dir, metric, snippets)
            contexts.append(context)
            if not args.dry_run:
                _write_context_markdown(context, profile)

    if not args.dry_run:
        _write_context_manifest(output_dir / "manifest.csv", contexts)
        _write_context_jsonl(output_dir / "eval_contexts.jsonl", contexts)
        _write_eval_cases(_repo_path(args.eval_cases_out), contexts)
        _write_report(output_dir / "reports" / "normalize_report.md", contexts, errors, args)

    payload = {
        "selected_filings": len(rows),
        "contexts": len(contexts),
        "symbols": sorted({context.symbol for context in contexts}),
        "metrics": sorted({context.metric_type for context in contexts}),
        "output_dir": str(output_dir),
        "eval_cases": str(_repo_path(args.eval_cases_out)),
        "errors": errors,
        "dry_run": args.dry_run,
        "embedding": "not_run",
        "chroma": "not_touched",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if errors and not contexts else 0


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-cases-out", type=Path, default=DEFAULT_EVAL_CASES)
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument("--per-symbol", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--min-snippets", type=int, default=1)
    parser.add_argument("--max-snippets", type=int, default=4)
    parser.add_argument("--max-snippet-chars", type=int, default=1200)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.per_symbol < 0:
        raise SystemExit("--per-symbol must be non-negative")
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.min_snippets < 0:
        raise SystemExit("--min-snippets must be non-negative")
    if args.max_snippets <= 0:
        raise SystemExit("--max-snippets must be positive")
    if args.max_snippet_chars < 200:
        raise SystemExit("--max-snippet-chars must be at least 200")
    return args


def _repo_path(path: Path | str) -> Path:
    item = Path(path)
    return item if item.is_absolute() else REPO_ROOT / item


def _split_csv(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _select_rows(
    rows: list[dict[str, str]],
    *,
    symbols: list[str],
    per_symbol: int,
    limit: int,
) -> list[dict[str, str]]:
    allowed_symbols = {symbol.upper() for symbol in symbols}
    selected: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for row in rows:
        symbol = row.get("symbol", "").upper()
        if allowed_symbols and symbol not in allowed_symbols:
            continue
        if row.get("status") not in {"downloaded", "exists"}:
            continue
        if row.get("source_type") not in ALLOWED_SOURCE_TYPES:
            continue
        if per_symbol and counts.get(symbol, 0) >= per_symbol:
            continue
        selected.append(row)
        counts[symbol] = counts.get(symbol, 0) + 1
        if limit and len(selected) >= limit:
            break
    return selected


def _extract_raw_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in {".htm", ".html"}:
        if BeautifulSoup is not None:
            return _extract_html_text_with_tables(raw)
        parser = _SecHtmlTextExtractor()
        parser.feed(raw)
        return parser.text()
    return _normalize_text(raw)


def _extract_html_text_with_tables(raw: str) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    for node in soup(["script", "style", "noscript"]):
        node.decompose()
    for table in soup.find_all("table"):
        rows: list[str] = []
        for row in table.find_all("tr"):
            cells = [
                _normalize_inline_text(cell.get_text(" ", strip=True))
                for cell in row.find_all(["th", "td"])
            ]
            cells = [cell for cell in cells if cell and cell != "$"]
            if len(cells) >= 2:
                rows.append(" | ".join(cells))
            elif cells:
                rows.append(cells[0])
        if rows:
            table.replace_with("\n" + "\n".join(rows) + "\n")
    return _normalize_text(soup.get_text("\n"))


def _normalize_inline_text(value: str) -> str:
    value = unescape(value).replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def _normalize_text(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t\f\v]+", " ", value)
    value = re.sub(r"\s*\n\s*", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _snippets_for_metric(
    *,
    text: str,
    keywords: Iterable[str],
    max_snippets: int,
    max_chars: int,
) -> list[str]:
    keyword_list = [keyword.lower() for keyword in keywords]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidates: list[tuple[int, int, int, str]] = []
    seen: set[str] = set()
    for index, line in enumerate(lines):
        lower = line.lower()
        if not any(keyword in lower for keyword in keyword_list):
            continue
        start = max(0, index - 8)
        end = min(len(lines), index + 18)
        snippet = _clean_snippet("\n".join(lines[start:end]), max_chars=max_chars)
        key = re.sub(r"\W+", " ", snippet.lower())[:240]
        if len(snippet) < 120 or key in seen or _is_navigation_snippet(snippet):
            continue
        seen.add(key)
        score = _snippet_score(snippet, keyword_list)
        candidates.append((score, start, end, snippet))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected: list[tuple[int, int, str]] = []
    for _score, start, end, snippet in candidates:
        if any(start <= selected_end and end >= selected_start for selected_start, selected_end, _ in selected):
            continue
        selected.append((start, end, snippet))
        if len(selected) >= max_snippets:
            break
    selected.sort(key=lambda item: item[0])
    return [snippet for _, _, snippet in selected]


def _snippet_score(snippet: str, keywords: list[str]) -> int:
    lower = snippet.lower()
    keyword_hits = sum(1 for keyword in keywords if keyword in lower)
    number_hits = len(re.findall(r"\$?\(?\d[\d,]*(?:\.\d+)?%?\)?", snippet))
    sec_statement_hits = sum(
        1
        for marker in (
            "consolidated statements",
            "consolidated statement",
            "results of operations",
            "management's discussion",
            "operating income by business unit",
            "statements of income",
            "statements of operations",
            "three months ended",
            "six months ended",
            "nine months ended",
            "year ended",
        )
        if marker in lower
    )
    return keyword_hits * 10 + min(number_hits, 20) + sec_statement_hits * 5


def _is_navigation_snippet(snippet: str) -> bool:
    lower = snippet.lower()
    if "table of contents" in lower and "item 1. | financial statements" in lower:
        return True
    if "exhibit" in lower and "signatures" in lower:
        return True
    if "cover page interactive data file" in lower:
        return True
    return False


def _clean_snippet(value: str, *, max_chars: int) -> str:
    value = _normalize_text(value)
    value = re.sub(r"\n{2,}", "\n", value)
    if len(value) <= max_chars:
        return value
    cut = value[:max_chars].rsplit("\n", 1)[0].strip()
    return cut or value[:max_chars].strip()


def _context_record(
    row: dict[str, str],
    raw_path: Path,
    output_dir: Path,
    metric: str,
    snippets: list[str],
) -> ContextRecord:
    symbol = row.get("symbol", "").upper()
    doc_id = row.get("doc_id", "")
    context_id = f"{doc_id}__{metric}"
    title = f"{symbol} {METRIC_PROFILES[metric]['title']} - {row.get('fiscal_period', '')}"
    text = "\n\n".join(snippets)
    markdown_path = output_dir / symbol / f"{context_id}.md"
    return ContextRecord(
        context_id=context_id,
        doc_id=doc_id,
        symbol=symbol,
        company_name=row.get("company_name", ""),
        source_type=row.get("source_type", ""),
        source_date=row.get("source_date", ""),
        fiscal_period=row.get("fiscal_period", ""),
        metric_type=metric,
        title=title,
        source_url=row.get("source_url", ""),
        raw_path=str(raw_path),
        markdown_path=str(markdown_path),
        text=text,
        snippet_count=len(snippets),
    )


def _write_context_markdown(context: ContextRecord, profile: dict) -> None:
    path = Path(context.markdown_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        f"context_id: {context.context_id}",
        f"doc_id: {context.doc_id}",
        f"symbol: {context.symbol}",
        f"company_name: {_yaml_string(context.company_name)}",
        f"source_type: {context.source_type}",
        f"source_date: {context.source_date}",
        f"fiscal_period: {_yaml_string(context.fiscal_period)}",
        f"metric_type: {context.metric_type}",
        "question_intent: entity_specific",
        "annotation_status: candidate_needs_human_review",
        f"source_url: {context.source_url}",
        f"raw_path: {_yaml_string(context.raw_path)}",
        "---",
        "",
        f"# {context.title}",
        "",
        "## Evaluation Use",
        "",
        f"- Positive context for {profile['positive_use']}",
        "- Hard negative for same-metric questions about a different company unless the query explicitly asks for peer comparison.",
        "- Source scope: official SEC filing excerpt; no news or market commentary is included.",
        "",
        "## Extracted SEC Snippets",
        "",
    ]
    for index, snippet in enumerate(context.text.split("\n\n"), start=1):
        lines.extend([f"### Snippet {index}", "", snippet, ""])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _write_context_manifest(path: Path, contexts: list[ContextRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "context_id",
        "doc_id",
        "symbol",
        "company_name",
        "source_type",
        "source_date",
        "fiscal_period",
        "metric_type",
        "title",
        "source_url",
        "raw_path",
        "markdown_path",
        "snippet_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for context in contexts:
            writer.writerow({field: getattr(context, field) for field in fields})


def _write_context_jsonl(path: Path, contexts: list[ContextRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for context in contexts:
            handle.write(
                json.dumps(
                    {
                        "context_id": context.context_id,
                        "title": context.title,
                        "text": context.text,
                        "symbol": context.symbol,
                        "company_name": context.company_name,
                        "source_type": context.source_type,
                        "source_date": context.source_date,
                        "fiscal_period": context.fiscal_period,
                        "metric_type": context.metric_type,
                        "source_url": context.source_url,
                        "markdown_path": context.markdown_path,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_eval_cases(path: Path, contexts: list[ContextRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    latest_by_symbol_metric: dict[tuple[str, str], ContextRecord] = {}
    for context in contexts:
        key = (context.symbol, context.metric_type)
        existing = latest_by_symbol_metric.get(key)
        if existing is None or context.source_date > existing.source_date:
            latest_by_symbol_metric[key] = context

    cases = []
    for (symbol, metric), context in sorted(latest_by_symbol_metric.items()):
        hard_negatives = [
            other.context_id
            for (other_symbol, other_metric), other in latest_by_symbol_metric.items()
            if other_metric == metric and other_symbol != symbol
        ][:8]
        metric_title = METRIC_PROFILES[metric]["title"].lower()
        cases.append(
            {
                "name": f"{symbol.lower()}_{metric}_latest_sec",
                "user_input": (
                    f"For {symbol}, what does the latest SEC filing say about {metric_title}? "
                    f"{symbol} 最新 SEC 财报如何说明{_metric_label_zh(metric)}？"
                ),
                "symbol": symbol,
                "intent": "entity_specific",
                "question_type": metric,
                "allowed_source_types": [context.source_type],
                "reference_context_ids": [context.context_id],
                "hard_negative_context_ids": hard_negatives,
                "expected_answer": (
                    f"Use only {symbol} SEC filing context for {metric_title}; do not answer from "
                    "same-metric filings of other companies unless the query asks for peer comparison."
                ),
                "annotation_status": "candidate_needs_human_review",
                "annotation_source": "deterministic_sec_filing_normalization",
                "annotation_granularity": "filing_metric_context",
            }
        )
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _metric_label_zh(metric: str) -> str:
    return {
        "profitability": "盈利能力",
        "cash_flow": "现金流",
        "balance_sheet": "资产负债表质量",
        "risk_factors": "风险因素",
    }.get(metric, metric)


def _write_report(
    path: Path,
    contexts: list[ContextRecord],
    errors: list[str],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_symbol: dict[str, int] = {}
    by_metric: dict[str, int] = {}
    for context in contexts:
        by_symbol[context.symbol] = by_symbol.get(context.symbol, 0) + 1
        by_metric[context.metric_type] = by_metric.get(context.metric_type, 0) + 1
    lines = [
        "# Fundamental Corpus Normalize Report",
        "",
        "- Scope: SEC 10-K/10-Q raw filings normalized into reviewable Markdown contexts.",
        "- Embedding: not run.",
        "- Chroma: not touched.",
        f"- Selected symbols: {args.symbols}",
        f"- Metrics: {args.metrics}",
        f"- Contexts: {len(contexts)}",
        f"- By symbol: {_format_counts(by_symbol)}",
        f"- By metric: {_format_counts(by_metric)}",
        "",
        "## Output Files",
        "",
        f"- Context manifest: `{path.parents[1] / 'manifest.csv'}`",
        f"- Eval contexts JSONL: `{path.parents[1] / 'eval_contexts.jsonl'}`",
        f"- Candidate eval cases: `{_repo_path(args.eval_cases_out)}`",
        "",
        "## Review Notes",
        "",
        "- Treat generated contexts as candidate annotations until a human confirms the snippets answer each evaluation query.",
        "- For entity-specific questions, same-metric contexts from other symbols are hard negatives.",
        "- Keep industry/news material in `knowledge/stock_research`; keep SEC filing contexts here.",
    ]
    if errors:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {error}" for error in errors)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "-"


if __name__ == "__main__":
    raise SystemExit(main())
