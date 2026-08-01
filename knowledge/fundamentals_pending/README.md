# Fundamental Raw Document Staging

This folder stores raw fundamental source documents before they are normalized
or ingested into Chroma.

Files here are not read by `ai-trading-copilot-rag ingest-defaults`. The default
RAG seed directory remains `knowledge/fundamentals/`.

## Raw-Only Scope

- Store original downloaded files only: PDF, HTML, TXT, XLS/XLSX, or ZIP.
- Do not extract text here.
- Do not create Markdown chunks here.
- Do not run embedding or Chroma ingestion from this folder.

## Layout

```text
knowledge/fundamentals_pending/
  manifest.csv
  raw/
    {SYMBOL}/
```

Raw filenames should include the ticker and Chinese company name:

```text
{SYMBOL}_{中文公司名}_{report_date}_{form}_{accession}.htm
```

Example:

```text
NVDA_英伟达_2026-04-26_10-Q_0001045810-26-000052.htm
```

## Allowed Source Types

- `earnings_report`
- `earnings_call_transcript`
- `sec_10q`
- `sec_10k`
- `annual_report`
- `investor_day`
- `company_guidance`
- `fundamental_research_note`

Move reviewed and normalized Markdown files to `knowledge/fundamentals/` only
when they are ready for RAG ingestion.
