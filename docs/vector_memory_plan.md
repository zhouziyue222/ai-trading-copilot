# Vector Memory Database Plan

## Goal

Add a retrieval layer that lets the fundamental/news agent reuse company
fundamental documents instead of relying only on the latest run inputs. The
default backend is OpenAI-compatible embeddings plus a persistent local Chroma
database. Distilled post-trade lessons remain in JSONL memory and are not
synced into Chroma.

## Current Implementation

- Historical lesson source: `config/memory.jsonl`
- Seed fundamental documents: `knowledge/fundamentals/`
- Vector database: `config/rag_chroma`
- Runtime service: `copilot.services.FundamentalRagStore`
- Embedding model: `text-embedding-3-small`
- Agent entry point: `PostTradeReviewLearningAgent`
- Retrieval cap: 5 context items per symbol

Only the fundamental/news analyst retrieves Chroma RAG context. Opportunity
radar and technical position prompts use market tools and deterministic
analysis, while post-trade lessons continue through JSONL memory retrieval.
Retrieval is hybrid: embedding vector recall and BM25 keyword recall run as
separate child-chunk recall channels, RRF merges the ranked lists, rerank orders
the fused candidates, and results are aggregated back to parent context.

## Data Model

Historical memory remains compact JSONL:

- `memory_type`: user behavior, strategy performance, or symbol characteristic
- `lesson`: distilled text, capped at 500 characters
- `symbols`, `tags`, `source_run_id`, `source_path`, `confidence`

Chroma stores child `RagDocument` chunks from fundamental sources only:

- `id`, `title`, `text`
- `source_type`: `earnings_report`, `earnings_call_transcript`, `sec_10q`,
  `sec_10k`, `annual_report`, `investor_day`, `company_guidance`, or
  `fundamental_research_note`
- `source`, `symbols`, `tags`, `created_at`
- metadata including `chunk_strategy=structured_semantic_parent_child_v1`,
  `parent_id`, `parent_text`, `section_title`, `section_path`, `section_type`,
  fiscal/source fields, `embedding_model`, and content hash

Chroma is a rebuildable index. Fundamental source files and online fundamental
fetches are durable inputs. JSONL memories are separate durable inputs.

## Retrieval Flow

1. Load distilled memories from JSONL for history-aware downstream decisions.
2. In the fundamental/news node, build a base query from symbol, tags, and
   analyst query text.
3. Ask the available LLM to rewrite the query into JSON fields:
   `rewritten_query`, `risk_query`, `technical_query`, and
   `fundamental_query`. If the LLM is unavailable or returns invalid JSON, use
   deterministic risk/technical/fundamental keyword expansions.
4. Run vector recall for each expanded query using embeddings and Chroma.
5. Run BM25/keyword recall for each expanded query by reading Chroma documents
   and scoring title, text, source type, symbols, and tags.
6. Filter/boost by symbol and tags, then fuse all vector and BM25 ranked lists
   with Reciprocal Rank Fusion (RRF).
7. Rerank fused child candidates with `local_cross_feature_v1`.
8. Aggregate child matches by `parent_id` and inject parent context only into
   the fundamental/news prompt with a clear boundary: RAG is background; live
   tools remain authoritative.

## Update Paths

- CLI seed import: `ai-trading-copilot-rag ingest-defaults`
- CLI local file import: `ai-trading-copilot-rag ingest-text <path> --symbols AAPL`
- CLI online import: `ai-trading-copilot-rag ingest-online --symbols AAPL,MSFT`
- CLI query: `ai-trading-copilot-rag query --symbol AAPL --query "earnings risk"`
- Web UI: RAG panel for status, seed import, online fetch, and manual text ingest.

## Guardrails

- Retrieved context must be limited to company fundamental material.
- Retrieved context must not bypass hard risk checks.
- Full reports are not stuffed into prompts; documents are chunked and capped.
- Chroma database files are not source-controlled and can be rebuilt.
- Future hosted vector databases should preserve metadata filters for symbol,
  tag, source type, and source path.
