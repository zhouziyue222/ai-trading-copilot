import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "normalize_pending_fundamentals.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("normalize_pending_fundamentals", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_pending_fundamentals_writes_reviewable_contexts(tmp_path):
    module = _load_script()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    mu_raw = raw_dir / "mu.htm"
    nvda_raw = raw_dir / "nvda.htm"
    mu_raw.write_text(
        """
        <html><body>
        <h1>Consolidated Statements of Operations</h1>
        <table>
        <tr><td>Revenue</td><td>$8,000</td></tr>
        <tr><td>Gross margin</td><td>41%</td></tr>
        <tr><td>Operating income</td><td>$2,000</td></tr>
        <tr><td>Net income</td><td>$1,700</td></tr>
        <tr><td>Diluted earnings per share</td><td>$1.25</td></tr>
        </table>
        </body></html>
        """,
        encoding="utf-8",
    )
    nvda_raw.write_text(
        """
        <html><body>
        <h1>Consolidated Statements of Operations</h1>
        <table>
        <tr><td>Revenue</td><td>$30,000</td></tr>
        <tr><td>Gross margin</td><td>70%</td></tr>
        <tr><td>Operating income</td><td>$18,000</td></tr>
        <tr><td>Net income</td><td>$16,000</td></tr>
        <tr><td>Diluted earnings per share</td><td>$3.20</td></tr>
        </table>
        </body></html>
        """,
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "doc_id": "MU_2026-05-28_10-Q_TEST",
                "symbol": "MU",
                "company_name": "MICRON TECHNOLOGY INC",
                "source_type": "sec_10q",
                "source_date": "2026-06-25",
                "fiscal_period": "Quarter ending 2026-05-28",
                "raw_format": "htm",
                "raw_path": str(mu_raw),
                "source_url": "https://example.test/mu",
                "status": "exists",
                "notes": "",
            }
        )
        writer.writerow(
            {
                "doc_id": "NVDA_2026-04-26_10-Q_TEST",
                "symbol": "NVDA",
                "company_name": "NVIDIA CORP",
                "source_type": "sec_10q",
                "source_date": "2026-05-20",
                "fiscal_period": "Quarter ending 2026-04-26",
                "raw_format": "htm",
                "raw_path": str(nvda_raw),
                "source_url": "https://example.test/nvda",
                "status": "exists",
                "notes": "",
            }
        )

    output_dir = tmp_path / "fundamentals"
    eval_cases = tmp_path / "rag_eval_fundamentals_seed.json"
    result = module.main(
        [
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--eval-cases-out",
            str(eval_cases),
            "--symbols",
            "MU,NVDA",
            "--per-symbol",
            "1",
            "--metrics",
            "profitability",
        ]
    )

    assert result == 0
    mu_context = output_dir / "MU" / "MU_2026-05-28_10-Q_TEST__profitability.md"
    assert mu_context.exists()
    text = mu_context.read_text(encoding="utf-8")
    assert "annotation_status: candidate_needs_human_review" in text
    assert "metric_type: profitability" in text
    assert "Gross margin" in text
    assert not (tmp_path / "rag_chroma").exists()

    cases = json.loads(eval_cases.read_text(encoding="utf-8"))
    mu_case = next(case for case in cases if case["symbol"] == "MU")
    assert mu_case["reference_context_ids"] == ["MU_2026-05-28_10-Q_TEST__profitability"]
    assert mu_case["hard_negative_context_ids"] == [
        "NVDA_2026-04-26_10-Q_TEST__profitability"
    ]
