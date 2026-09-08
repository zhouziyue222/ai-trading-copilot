import pytest

from ai_trading_copilot.copilot.agents.llm_tools import strip_trailing_json_object
from ai_trading_copilot.copilot.config.prompts import render_prompt
from ai_trading_copilot.copilot.services.reporting import prepare_report


@pytest.mark.parametrize("payload", [
    '```json\n{"reason": "字符串中含有 } 与 {", "nested": {"a": [1, 2]}}\n```',
    '{"direction": "watch", "targets": [110, 118]}',
    '```\n{"direction": "watch"}\n```',
    '[{"direction": "watch"}]',
    '```json\n{"unfinished":',
])
def test_report_removes_machine_payload_without_losing_evidence(payload):
    narrative = '## 结论\n\n继续观察，支撑位100美元。\n\n[新闻来源](https://example.com/news)'
    result = strip_trailing_json_object(narrative + '\n\n' + payload)
    assert result == narrative


def test_publication_gate_marks_missing_values_and_is_idempotent():
    result = prepare_report('# 技术分析\n\n- 支撑：None\n- 均线：N/A\n| 股价 | - |\n')
    assert result.count('【待补充】') == 4
    assert '信息概括' in result and '结论' in result
    assert prepare_report(result) == result


def test_empty_machine_only_report_does_not_invent_conclusion():
    result = prepare_report('{"direction": "buy"}')
    assert '证据不足' in result and 'buy' not in result


def test_markdown_links_and_numeric_evidence_survive_cleanup():
    content = '## 信息概括\n\n[来源](https://example.com)\n\n## 结论\n\n目标110美元，权重0.00%，区间[100, 110]美元。'
    # A JSON array is deliberately excluded even when embedded in prose.
    result = prepare_report(content)
    assert '[来源](https://example.com)' in result
    assert '目标110美元，权重0.00%' in result


def test_every_prompt_includes_professional_chinese_report_contract():
    for name in ['trader.v1', 'technical_position.v1', 'technical_position.debug.v1',
                 'opportunity_radar.v1', 'news_sentiment.v1', 'fundamental_analyst.v1',
                 'portfolio_memory_advisor.v1']:
        prompt = render_prompt(name)
        assert '【待补充】' in prompt
        assert '信息概括' in prompt and '静默自检' in prompt
        assert '结论与结构化决策/风险约束一致' in prompt


def test_saved_stage_report_removes_fenced_json(tmp_path):
    from pathlib import Path
    from ai_trading_copilot.copilot.graph.copilot_langgraph import CopilotLangGraph

    graph = CopilotLangGraph(enable_default_llm=False, report_output_dir=tmp_path)
    path = graph._save_agent_report(
        state={}, stage='3_trader', agent_name='trader',
        content='## 信息概括\n\n当前价格100美元。\n\n## 结论\n\n观察。\n\n```json\n{"direction":"watch"}\n```',
    )
    report = Path(path).read_text(encoding='utf-8')
    assert '当前价格100美元' in report and '观察' in report
    assert 'json' not in report and 'direction' not in report


def test_missing_price_history_never_displays_placeholder_price():
    from ai_trading_copilot.copilot.agents.technical_position_agent import TechnicalPositionAgent

    result = TechnicalPositionAgent().analyze_with_report(symbol='MU', bars=[])
    assert '【待补充】' in result.report
    assert '1.00' not in result.report
