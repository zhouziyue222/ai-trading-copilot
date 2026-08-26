# AI Trading Copilot

AI Trading Copilot is a standalone, subscription-list-driven multi-agent trading
copilot for US stocks and ETFs. It can be copied out of the original
TradingAgents repository and installed from this directory with `pip install -e .`.

Implemented MVP slices:

- User persona and subscription-list domain models
- Subscription status machine and JSON-backed subscription store
- Deterministic market regime, technical-position, and opportunity-radar agents
- Fundamental/news risk review manager
- Trader agent that creates structured trade plans for actionable setups
- ai-hedge-fund v2-style Risk Manager that clamps target weights by single-name and gross-exposure limits
- Portfolio Manager that turns risk-adjusted weights into final hold/buy/sell/reduce decisions
- Fundamental-only Chroma RAG knowledge base with OpenAI embeddings, BM25 keyword recall, RRF fusion, rerank, and structured semantic parent-child chunking
- Futu/yfinance market data adapters and OHLCV CSV parsing
- End-to-end subscription opportunity workflow

Install:

```bash
python -m pip install -e .
```

Development install and verification:

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

Evaluation install:

```bash
python -m pip install -e ".[dev,eval]"
```

Web UI:

```bash
ai-trading-copilot-ui
```

Then open `http://127.0.0.1:8000`. The UI keeps execution in simulation mode and only allows
live portfolio access as read-only context.

Local trace is always written under each run report directory as `trace.json` and
`trace.md`. For interactive OpenTelemetry debugging with Jaeger:

```powershell
.\scripts\start-observability.ps1
.\scripts\start-ui-with-otel.ps1
```

Then run the UI as usual and open `http://127.0.0.1:16686`. Search for service
`ai-trading-copilot` and filter by `run.id` to inspect the node/tool/LLM waterfall.
Full LLM content and reasoning details stay in the local `trace.json`; Jaeger only
receives compact metadata and hashes.

CLI examples:

```bash
# Single symbol, legacy-compatible usage
ai-trading-copilot CRCL

# Multiple symbols
ai-trading-copilot AAPL MSFT CRCL
ai-trading-copilot --symbols AAPL,MSFT,CRCL

# Select symbols from config/subscriptions.json
ai-trading-copilot --select-symbols

# Analyze every symbol in the subscription file
ai-trading-copilot --all-subscribed

# Run a specific analyst combination
ai-trading-copilot CRCL --analysts news_sentiment,technical_position,fundamental_analysis
ai-trading-copilot CRCL --select-analysts
```

RAG memory:

The copilot reads optional distilled trading memories from `config/memory.jsonl`
and stores fundamental research chunks in a persistent Chroma database at
`config/rag_chroma`. OpenAI-compatible embeddings use `OPENAI_API_KEY` or
`DASHSCOPE_API_KEY` and default to `text-embedding-v4`. When Chroma or
embeddings are unavailable, the
workflow degrades to the JSONL/keyword memory retrieval path instead of blocking
the analysis.

Retrieval uses a hybrid RAG pipeline: the original request is rewritten by the
available LLM, expanded into base/risk/technical/fundamental queries, sent
through vector recall and BM25 keyword recall, then merged with Reciprocal Rank
Fusion (RRF). The fundamental analyst receives the fused Chroma results as
background evidence.

Example line:

```json
{"memory_type":"strategy_performance","lesson":"Wait for support confirmation before adding size on AAPL pullbacks.","symbols":["AAPL"],"tags":["pullback","risk"],"source_run_id":"run_AAPL_20260514_090000","source_path":"reports/run_AAPL_20260514_090000/run_audit.md","created_at":"2026-05-14T09:00:00+08:00","confidence":0.8}
```

Seed the default stock-research knowledge base:

```bash
ai-trading-copilot-rag ingest-defaults
```

Default seed ingestion uses reviewed Markdown files from `knowledge/fundamentals`
and `knowledge/stock_research`; raw files under `knowledge/fundamentals_pending`
are not ingested.

Add a local research note:

```bash
ai-trading-copilot-rag ingest-text docs/my_aapl_note.md --symbols AAPL --tags earnings,risk
```

Fetch online fundamental text into Chroma:

```bash
ai-trading-copilot-rag ingest-online --symbols AAPL,MSFT --look-back-days 7
```

Inspect or override the active Chroma database path:

```bash
ai-trading-copilot-rag --chroma-dir config/rag_chroma status
ai-trading-copilot-rag --chroma-dir config/rag_chroma status --probe
```

On Windows PowerShell, use UTF-8 mode when inspecting large RAG query payloads
that may contain Chinese text or SEC filing whitespace:

```powershell
$env:PYTHONUTF8='1'
ai-trading-copilot-rag query --query "MU gross margin HBM guidance" --symbol MU --tags fundamentals --limit 3
```

`status` avoids opening Chroma and may report `available: null`; use
`status --probe` or the dashboard refresh button to verify the active database.

Evaluation:

```bash
# Deterministic agent workflow smoke evaluation
ai-trading-copilot-eval agent-smoke --symbols AAPL --format markdown

# Standalone analyst evaluation
ai-trading-copilot-eval analyst-smoke --analysts news_sentiment,technical_position,fundamental_analysis --symbols AAPL --format markdown

# Fundamental analyst evaluation with RAGAS sample capture
ai-trading-copilot-eval fundamental-eval --symbols AAPL --backend ragas --answer-quality auto --format markdown

# RAGAS-style before/after comparison: BM25 baseline vs optimized hybrid retrieval
ai-trading-copilot-eval rag-compare --backend ragas --answer-quality auto --top-k 5 --format markdown

```

`rag-compare` runs Chroma access in a worker process so native Chroma failures are
reported as structured evaluation failures instead of crashing the parent process.
Retrieval quality is the hard gate. `faithfulness` and `answer_relevancy` are
reported when RAGAS and evaluator credentials are available, but they are soft
metrics unless `--answer-quality required` is passed. Vector recall remains
disabled by default; pass `--use-vector` when `OPENAI_API_KEY`/compatible
embedding credentials are configured.
`fundamental-eval` runs only the fundamental analyst and records RAGAS-compatible
samples under the evaluation output directory. `rag-compare` records the baseline
and optimized retrieval samples separately in the same JSONL schema.

The Web UI also exposes these RAG actions in the run-control panel. The
fundamental analyst receives Chroma results as background context; fresh tool
data and Risk Manager limits remain authoritative inputs to the Portfolio Manager.

See `docs/vector_memory_plan.md` for the vector database design notes.

Safety boundary:

- This package does not support live broker orders.
- Portfolio Manager decisions are recommendations during the run.
- Futu `SIMULATE` orders are submitted only after result-page confirmation.
- The project does not call `unlock_trade`.
- Existing `tradingagents/` source files are not modified by this project.
