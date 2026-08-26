# Fundamental Corpus Normalize Report

- Scope: SEC 10-K/10-Q raw filings normalized into reviewable Markdown contexts.
- Embedding: not run.
- Chroma: not touched.
- Selected symbols: MU,NVDA,AMD,AVGO,INTC
- Metrics: profitability,cash_flow,balance_sheet
- Contexts: 30
- By symbol: AMD=6, AVGO=6, INTC=6, MU=6, NVDA=6
- By metric: balance_sheet=10, cash_flow=10, profitability=10

## Output Files

- Context manifest: `E:\ai_trading_copilot\knowledge\fundamentals\manifest.csv`
- Eval contexts JSONL: `E:\ai_trading_copilot\knowledge\fundamentals\eval_contexts.jsonl`
- Candidate eval cases: `E:\ai_trading_copilot\config\rag_eval_fundamentals_seed.json`

## Review Notes

- Treat generated contexts as candidate annotations until a human confirms the snippets answer each evaluation query.
- For entity-specific questions, same-metric contexts from other symbols are hard negatives.
- Keep industry/news material in `knowledge/stock_research`; keep SEC filing contexts here.
