from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ai_trading_copilot.copilot import memory_eval


class Model:
    model_name = "test-fixture-not-a-real-model"

    def __init__(self, *, broken=False, malformed=False):
        self.prompts = []
        self.broken = broken
        self.malformed = malformed

    def bind_tools(self, tools):
        return self

    def invoke(self, prompt):
        text = prompt if isinstance(prompt, str) else prompt[0].content
        self.prompts.append(text)
        if self.broken:
            raise RuntimeError("fixture model unavailable")
        if self.malformed:
            return SimpleNamespace(content="No structured answer.", tool_calls=[])
        if "proceed, scale, or hold" in text:
            payload = {"decision": "scale", "scale": 0.5, "memory_citations": ["support@1", "invented@1"]}
        else:
            payload = {
                "direction": "buy", "position_weight": 0.2, "entry_logic": "Fixture decision.",
                "memory_citations": ["support@1", "invented@1"],
            }
        return SimpleNamespace(content=json.dumps(payload), tool_calls=[], usage_metadata={"total_tokens": 100})


def test_offline_evaluation_has_honest_retrieval_and_learning_metrics(monkeypatch):
    def forbidden():
        raise AssertionError("offline evaluation must not construct a real model")

    monkeypatch.setattr("ai_trading_copilot.copilot.config.llm.create_default_deepseek_llm", forbidden)
    result = memory_eval.run_evaluation()
    assert result["case_count"] == 9
    assert result["model_pairs"] == []
    assert result["mean_decision_constraint_pass_rate_delta"] is None
    assert result["learning_contract"]["passed"]
    assert result["metrics"]["recall_at_3"] == 1
    # A known same-symbol false positive remains visible, rather than being
    # dropped from the evaluation to manufacture a perfect result.
    assert result["metrics"]["irrelevant_injection_rate"] > 0
    assert result["metrics"]["invalid_citation_rate"] == 0
    assert len(result["code_tree_sha256"]) == 64
    assert all(row["case_id"] != "development-basic" for row in result["retrieval"])


def test_live_trial_uses_real_decision_pipeline_and_reports_filtered_citations(tmp_path):
    case = memory_eval.load_cases(memory_eval.DEFAULT_CASES, "evaluation")[0]
    model = Model()
    result = memory_eval._model_trial(case, tmp_path, model, with_memory=True)
    assert result["completed"]
    assert result["llm_calls"] == 2
    assert result["total_tokens"] == 200
    assert result["plan"]["position_weight"] == 0.2
    assert result["risk"]["final_weight"] == 0.2
    assert result["decision"]["final_weight"] == 0.1
    assert result["decision"]["requires_user_confirmation"]
    assert not result["decision"]["submitted_to_broker"]
    assert result["raw_invalid_citation_count"] == 2
    assert result["invalid_citation_count"] == 0
    assert all("allowed_directions" not in prompt and "relevant_ids" not in prompt for prompt in model.prompts)


def test_without_memory_arm_has_no_memory_context_or_advisor_call(tmp_path):
    case = memory_eval.load_cases(memory_eval.DEFAULT_CASES, "evaluation")[0]
    model = Model()
    result = memory_eval._model_trial(case, tmp_path, model, with_memory=False)
    assert result["completed"]
    assert result["llm_calls"] == 1
    assert result["retrieved_ids"] == []
    assert result["decision"]["final_weight"] == 0.2
    assert result["plan"]["memory_citations"] == []
    assert "Confirm support retest before entry." not in model.prompts[0]


def test_raw_citation_audit_does_not_borrow_another_consumers_retrieval(tmp_path):
    from ai_trading_copilot.copilot.domain import DistilledMemory

    case = memory_eval.load_cases(memory_eval.DEFAULT_CASES, "evaluation")[0]
    case.memories.append(DistilledMemory(
        memory_id="pm-only", memory_type="strategy_performance", scope="strategy",
        status="approved", lesson="Rebalance allocations.", tags=["portfolio"],
    ))

    class CrossConsumerModel(Model):
        def invoke(self, prompt):
            response = super().invoke(prompt)
            payload = json.loads(response.content)
            payload["memory_citations"] = ["pm-only@1"]
            response.content = json.dumps(payload)
            return response

    result = memory_eval._model_trial(case, tmp_path, CrossConsumerModel(), with_memory=True)
    assert result["completed"]
    assert result["raw_invalid_citation_count"] == 1
    assert result["plan"]["memory_citations"] == []
    assert result["decision"]["memory_citations"] == ["pm-only@1"]


@pytest.mark.parametrize("kind", ["broken", "malformed"])
def test_model_failure_does_not_receive_deterministic_fallback_credit(tmp_path, kind):
    case = memory_eval.load_cases(memory_eval.DEFAULT_CASES, "evaluation")[0]
    result = memory_eval._model_trial(case, tmp_path, Model(**{kind: True}), with_memory=True)
    assert not result["completed"]
    assert result["constraint_pass_rate"] is None
    assert result["decision_constraint_pass_rate"] is None
    assert result["trader_fallback"]


def test_live_requires_credentials_and_does_not_emit_success(monkeypatch, capsys):
    monkeypatch.setattr("ai_trading_copilot.copilot.config.llm.create_default_deepseek_llm", lambda: None)
    assert memory_eval.main(["--live"]) == 2
    assert "DEEPSEEK_API_KEY" in json.loads(capsys.readouterr().out)["error"]


def test_memory_retrieval_failure_is_not_credited_as_a_memory_trial(tmp_path, monkeypatch):
    case = memory_eval.load_cases(memory_eval.DEFAULT_CASES, "evaluation")[0]

    def fail(*args, **kwargs):
        raise RuntimeError("fixture retrieval unavailable")

    monkeypatch.setattr(memory_eval.MemoryRetrievalSession, "prefetch", fail)
    result = memory_eval._model_trial(case, tmp_path, Model(), with_memory=True)
    assert not result["completed"]
    assert result["memory_retrieval_error"]
    assert result["decision_constraint_pass_rate"] is None


def test_live_pair_report_preserves_case_inputs_and_incomplete_pairs(tmp_path, monkeypatch):
    fixture = json.loads(memory_eval.DEFAULT_CASES.read_text(encoding="utf-8"))
    fixture["cases"] = fixture["cases"][:1]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    monkeypatch.setattr("ai_trading_copilot.copilot.config.llm.create_default_deepseek_llm", lambda: Model(broken=True))
    report = memory_eval.run_evaluation(path, live=True)
    assert len(report["model_pairs"]) == 1
    assert report["completed_pairs"] == 0
    assert report["mean_constraint_pass_rate_delta"] is None
    assert json.loads(path.read_text(encoding="utf-8")) == fixture


def test_fixture_validation_and_cli_output(tmp_path, capsys):
    output = tmp_path / "result.json"
    assert memory_eval.main(["--split", "development", "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(capsys.readouterr().out)
    fixture = json.loads(memory_eval.DEFAULT_CASES.read_text(encoding="utf-8"))
    fixture["cases"][0]["relevant_ids"] = ["missing-id"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    assert memory_eval.main(["--cases", str(path)]) == 2


@pytest.mark.parametrize("payload", [[], {"format_version": "memory_effect_cases.v1"}])
def test_invalid_case_container_is_reported_as_configuration_error(tmp_path, payload, capsys):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert memory_eval.main(["--cases", str(path)]) == 2
    assert "error" in json.loads(capsys.readouterr().out)


def test_native_deepseek_long_response_is_not_scored_from_truncated_trace(tmp_path, monkeypatch):
    from ai_trading_copilot.copilot.agents import llm_tools

    class NativeModel(Model):
        model_name = "deepseek-fixture"
        openai_api_base = "https://example.invalid"
        openai_api_key = "test-only-key"
        extra_body = {"thinking": {"type": "enabled"}}

        def invoke(self, prompt):
            response = super().invoke(prompt)
            response.content = "Long final explanation. " * 1000 + "\n" + response.content
            return response

    model = NativeModel()

    def create(**kwargs):
        response = model.invoke(kwargs["messages"][0]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(message=response)], usage=SimpleNamespace(total_tokens=100))

    monkeypatch.setattr(llm_tools, "_create_openai_client", lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ))
    case = memory_eval.load_cases(memory_eval.DEFAULT_CASES, "evaluation")[0]
    enabled = memory_eval._model_trial(case, tmp_path / "with", model, with_memory=True)
    disabled = memory_eval._model_trial(case, tmp_path / "without", model, with_memory=False)
    assert enabled["completed"] and disabled["completed"]
    assert enabled["total_tokens"] == 200
    assert disabled["total_tokens"] == 100
    assert not enabled["trader_fallback"] and not enabled["portfolio_fallback"]
