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
- Hard risk agent for subscription, stop-loss, position, drawdown, instrument, and chasing gates
- Execution/alert manager that defaults to simulation and requires confirmation for live mode
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

Web UI:

```bash
ai-trading-copilot-ui
```

Then open `http://127.0.0.1:8000`. The UI keeps execution in simulation mode and only allows
live portfolio access as read-only context.

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
```

Evaluation:

```bash
# Deterministic agent workflow smoke evaluation
ai-trading-copilot-eval agent-smoke --symbols AAPL --format markdown

# Standalone analyst evaluation
ai-trading-copilot-eval analyst-smoke --analysts news_sentiment,technical_position,fundamental_analysis --symbols AAPL --format markdown

# RAG before/after comparison: single-query BM25 baseline vs optimized hybrid retrieval
ai-trading-copilot-eval rag-compare --top-k 5 --format markdown

```

`rag-compare` defaults to offline mode and disables external vector embedding calls.
Pass `--use-vector` when `OPENAI_API_KEY`/compatible embedding credentials are
configured and you want to include the vector recall channel in the optimized run.

The Web UI also exposes these RAG actions in the run-control panel. The
fundamental analyst receives Chroma results as background context; fresh tool
data and hard risk rules remain authoritative.

See `docs/vector_memory_plan.md` for the vector database design notes.

Safety boundary:

- This package does not place broker orders.
- Live execution can only become `LIVE_READY` after explicit user confirmation.
- Existing `tradingagents/` source files are not modified by this project.
