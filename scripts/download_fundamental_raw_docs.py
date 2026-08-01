"""Download raw official fundamental documents into the pending staging folder.

This script intentionally does not normalize text, create Markdown, compute
embeddings, or ingest anything into Chroma. It downloads raw SEC primary filing
documents and writes a manifest/report for later review.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests


SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVE_DOCUMENT_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{document}"

DEFAULT_OUTPUT_DIR = Path("knowledge") / "fundamentals_pending"
DEFAULT_FORMS = ("10-K", "10-Q")
DEFAULT_TICKERS = (
    "NVDA",
    "AMD",
    "AVGO",
    "MU",
    "INTC",
    "QCOM",
    "TXN",
    "AMAT",
    "LRCX",
    "KLAC",
    "MRVL",
    "WDC",
    "STX",
    "ON",
    "MCHP",
    "ADI",
    "NXPI",
    "MPWR",
    "SWKS",
    "QRVO",
    "TER",
    "COHR",
    "ACLS",
    "LSCC",
    "DIOD",
    "ALGM",
    "RMBS",
    "SITM",
)
CHINESE_COMPANY_NAMES = {
    "ADI": "亚德诺半导体",
    "AMAT": "应用材料",
    "AMD": "超威半导体",
    "AVGO": "博通",
    "INTC": "英特尔",
    "KLAC": "科磊",
    "LRCX": "泛林集团",
    "MCHP": "微芯科技",
    "MRVL": "迈威尔科技",
    "MU": "美光科技",
    "NVDA": "英伟达",
    "NXPI": "恩智浦半导体",
    "ON": "安森美",
    "QCOM": "高通",
    "STX": "希捷科技",
    "TXN": "德州仪器",
    "WDC": "西部数据",
}
MANIFEST_FIELDS = (
    "doc_id",
    "symbol",
    "company_name",
    "source_type",
    "source_date",
    "fiscal_period",
    "raw_format",
    "raw_path",
    "source_url",
    "status",
    "notes",
)


@dataclass(frozen=True)
class FilingCandidate:
    symbol: str
    company_name: str
    cik: str
    form: str
    filing_date: str
    report_date: str
    accession_number: str
    primary_document: str

    @property
    def source_type(self) -> str:
        if self.form == "10-K":
            return "sec_10k"
        if self.form == "10-Q":
            return "sec_10q"
        return "fundamental_research_note"

    @property
    def source_url(self) -> str:
        accession = self.accession_number.replace("-", "")
        return SEC_ARCHIVE_DOCUMENT_URL.format(
            cik_int=int(self.cik),
            accession=accession,
            document=self.primary_document,
        )

    @property
    def doc_id(self) -> str:
        base = f"{self.symbol}_{self.report_date or self.filing_date}_{self.form}_{self.accession_number}"
        return _slug(base)

    @property
    def raw_format(self) -> str:
        suffix = Path(self.primary_document).suffix.lower().lstrip(".")
        return suffix or "unknown"

    @property
    def fiscal_period(self) -> str:
        if self.form == "10-K":
            return f"FY ending {self.report_date}" if self.report_date else ""
        if self.form == "10-Q":
            return f"Quarter ending {self.report_date}" if self.report_date else ""
        return self.report_date


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    reports_dir = output_dir / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": args.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        }
    )

    ticker_map = _load_ticker_map(session, args.user_agent, args.timeout)
    candidates: list[FilingCandidate] = []
    for ticker in args.tickers:
        ticker = ticker.strip().upper()
        if not ticker:
            continue
        item = ticker_map.get(ticker)
        if item is None:
            candidates.append(
                FilingCandidate(
                    symbol=ticker,
                    company_name="",
                    cik="0",
                    form="",
                    filing_date="",
                    report_date="",
                    accession_number="",
                    primary_document="",
                )
            )
            continue
        candidates.extend(
            _filing_candidates_for_ticker(
                session=session,
                ticker=ticker,
                cik=item["cik"],
                company_name=item["company_name"],
                forms=set(args.forms),
                per_ticker=args.per_ticker,
                timeout=args.timeout,
            )
        )
        time.sleep(args.delay)
        if len([candidate for candidate in candidates if candidate.cik != "0"]) >= args.max_docs:
            break

    rows = []
    downloaded = 0
    for candidate in candidates:
        if downloaded >= args.max_docs:
            break
        if candidate.cik == "0":
            rows.append(_error_row(candidate.symbol, "ticker not found in SEC company_tickers.json"))
            continue
        destination = raw_dir / candidate.symbol / _raw_filename(candidate)
        destination.parent.mkdir(parents=True, exist_ok=True)
        status, notes = _download_raw_document(
            session=session,
            url=candidate.source_url,
            destination=destination,
            timeout=args.timeout,
            delay=args.delay,
            force=args.force,
        )
        if status in {"downloaded", "exists"}:
            downloaded += 1
        rows.append(
            {
                "doc_id": candidate.doc_id,
                "symbol": candidate.symbol,
                "company_name": candidate.company_name,
                "source_type": candidate.source_type,
                "source_date": candidate.filing_date,
                "fiscal_period": candidate.fiscal_period,
                "raw_format": candidate.raw_format,
                "raw_path": str(destination),
                "source_url": candidate.source_url,
                "status": status,
                "notes": notes,
            }
        )

    manifest_path = output_dir / "manifest.csv"
    _write_manifest(manifest_path, rows)
    _write_report(reports_dir / "download_report.md", rows, args)
    print(
        json.dumps(
            {
                "target": args.max_docs,
                "downloaded_or_existing": downloaded,
                "rows": len(rows),
                "manifest": str(manifest_path),
                "report": str(reports_dir / "download_report.md"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if downloaded >= args.max_docs else 1


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw SEC fundamental documents only.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-docs", type=int, default=100)
    parser.add_argument("--per-ticker", type=int, default=5)
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--forms", default=",".join(DEFAULT_FORMS))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--user-agent",
        default=os.getenv(
            "SEC_USER_AGENT",
            "ai-trading-copilot raw fundamental dataset contact@example.com",
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.tickers = [item.strip().upper() for item in args.tickers.split(",") if item.strip()]
    args.forms = [item.strip().upper() for item in args.forms.split(",") if item.strip()]
    if args.max_docs <= 0:
        raise SystemExit("--max-docs must be positive")
    if args.per_ticker <= 0:
        raise SystemExit("--per-ticker must be positive")
    return args


def _load_ticker_map(session: requests.Session, user_agent: str, timeout: float) -> dict[str, dict[str, str]]:
    response = session.get(
        SEC_COMPANY_TICKERS_URL,
        timeout=timeout,
        headers={"User-Agent": user_agent, "Host": "www.sec.gov"},
    )
    response.raise_for_status()
    payload = response.json()
    output = {}
    for item in payload.values():
        ticker = str(item["ticker"]).upper()
        output[ticker] = {
            "cik": str(item["cik_str"]).zfill(10),
            "company_name": str(item["title"]),
        }
    return output


def _filing_candidates_for_ticker(
    *,
    session: requests.Session,
    ticker: str,
    cik: str,
    company_name: str,
    forms: set[str],
    per_ticker: int,
    timeout: float,
) -> list[FilingCandidate]:
    response = session.get(
        SEC_SUBMISSIONS_URL.format(cik=cik),
        timeout=timeout,
        headers={"Host": "data.sec.gov"},
    )
    response.raise_for_status()
    recent = response.json()["filings"]["recent"]
    candidates = []
    for index, form in enumerate(recent.get("form", [])):
        if form.upper() not in forms:
            continue
        primary_document = str(recent["primaryDocument"][index])
        if not primary_document:
            continue
        candidates.append(
            FilingCandidate(
                symbol=ticker,
                company_name=company_name,
                cik=cik,
                form=form.upper(),
                filing_date=str(recent["filingDate"][index]),
                report_date=str(recent["reportDate"][index]),
                accession_number=str(recent["accessionNumber"][index]),
                primary_document=primary_document,
            )
        )
        if len(candidates) >= per_ticker:
            break
    return candidates


def _download_raw_document(
    *,
    session: requests.Session,
    url: str,
    destination: Path,
    timeout: float,
    delay: float,
    force: bool,
) -> tuple[str, str]:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return "exists", f"{destination.stat().st_size} bytes"
    time.sleep(delay)
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:
        return "error", str(exc)
    destination.write_bytes(response.content)
    return "downloaded", f"{len(response.content)} bytes"


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def _write_report(path: Path, rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    counts: dict[str, int] = {}
    formats: dict[str, int] = {}
    for row in rows:
        counts[row.get("status", "")] = counts.get(row.get("status", ""), 0) + 1
        formats[row.get("raw_format", "")] = formats.get(row.get("raw_format", ""), 0) + 1

    lines = [
        "# Fundamental Raw Document Download Report",
        "",
        f"- Target documents: {args.max_docs}",
        f"- Rows: {len(rows)}",
        f"- Status counts: {_format_counts(counts)}",
        f"- Raw format counts: {_format_counts(formats)}",
        "- Scope: raw files only; no text extraction, Markdown normalization, embedding, or Chroma ingestion.",
        "",
        "## Documents",
        "",
        "| # | Symbol | 中文名 | Company | Source Type | Source Date | Fiscal Period | Raw Format | Status | Raw Path |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    row.get("symbol", ""),
                    CHINESE_COMPANY_NAMES.get(row.get("symbol", ""), ""),
                    _escape_md(row.get("company_name", "")),
                    row.get("source_type", ""),
                    row.get("source_date", ""),
                    _escape_md(row.get("fiscal_period", "")),
                    row.get("raw_format", ""),
                    row.get("status", ""),
                    _escape_md(row.get("raw_path", "")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _error_row(symbol: str, message: str) -> dict[str, str]:
    return {
        "doc_id": "",
        "symbol": symbol,
        "company_name": "",
        "source_type": "",
        "source_date": "",
        "fiscal_period": "",
        "raw_format": "",
        "raw_path": "",
        "source_url": "",
        "status": "error",
        "notes": message,
    }


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _raw_filename(candidate: FilingCandidate) -> str:
    chinese_name = CHINESE_COMPANY_NAMES.get(candidate.symbol, "")
    prefix = f"{candidate.symbol}_{chinese_name}" if chinese_name else candidate.symbol
    remainder = candidate.doc_id.removeprefix(f"{candidate.symbol}_")
    return f"{prefix}_{remainder}.{candidate.raw_format}"


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "-"


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
