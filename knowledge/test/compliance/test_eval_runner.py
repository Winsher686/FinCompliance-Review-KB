"""评测运行器与 LLM-as-Judge 的测试（knowledge/eval/run_eval.py）。

守两件事：

1. **离线必须能跑**：不配置 LLM 时 `--judge` 只打印「已跳过」，绝不因为缺少外部依赖而失败；
2. **在线路径可测**：把 LLM 客户端换成桩，验证 JSON 解析、评分聚合、异常样本的降级处理。

全部打桩，不发任何真实网络请求。
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pytest

from knowledge.eval.dataset import load_samples
from knowledge.eval.run_eval import _render, _judge_prompt, main, run_judge
from knowledge.eval.dataset import evaluate_dataset


class _StubJudgeClient:
    """按预设脚本返回字符串的假 LLM 客户端（模拟真实 client.invoke_text 的输出）。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke_json(self, messages, default=None, **kwargs):
        from knowledge.utils.llm_client import parse_json_response
        self.calls.append(messages)
        if not self.responses:
            return default
        return parse_json_response(self.responses.pop(0), default=default)


def _only_first(n=4):
    return load_samples()[:n]


# ==================== 离线可跑 ====================

def test_judge_skips_when_no_llm(monkeypatch):
    """没有可用 LLM 时必须 skipped，而不是抛异常。"""
    monkeypatch.setattr("knowledge.utils.llm_client.get_llm_client", lambda *a, **k: None)
    result = run_judge(_only_first())
    assert result["status"] == "skipped"
    assert "langchain_deepseek" in result["reason"], "跳过原因应说明缺少什么，而不是静默跳过"
    assert result["items"] == []


def test_judge_skips_when_dependency_missing(monkeypatch):
    """连 llm_client 都不存在（依赖未装）也要能优雅退化。"""
    import knowledge.utils.llm_client as lc

    def _boom(*_a, **_k):
        raise ImportError("langchain_deepseek not installed")

    monkeypatch.setattr(lc, "get_llm_client", _boom)
    result = run_judge(_only_first())
    assert result["status"] == "skipped"
    assert "不可用" in result["reason"]


def test_render_marks_skipped_judge():
    report = evaluate_dataset(_only_first())
    text = _render(report, {"status": "skipped", "reason": "未配置 LLM", "items": []})
    assert "LLM-as-Judge" in text and "已跳过" in text
    assert "rule_hit_f1" in text


def test_main_exits_zero_offline(capsys):
    """离线直接跑 main()：确定性指标全绿 → 退出码 0。"""
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "评测结果" in out
    assert "样本数" in out


# ==================== 在线路径（打桩） ====================

def test_judge_aggregates_scores(monkeypatch):
    samples = _only_first(3)
    stub = _StubJudgeClient([
        '{"score": 5, "reason": "证据充分", "unsupported": []}',
        '```json\n{"score": 3, "reason": "证据略薄", "unsupported": ["R4 结论"]}\n```',   # 带围栏
        '评分如下：{"score": 4, "reason": "基本一致"}',                                  # 前后有说明文字
    ])
    monkeypatch.setattr("knowledge.utils.llm_client.get_llm_client", lambda *a, **k: stub)

    result = run_judge(samples)
    assert result["status"] == "ok"
    assert result["scored"] == 3
    assert result["mean_score"] == pytest.approx(4.0)
    assert result["pass_rate_ge4"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["items"][1]["unsupported"] == ["R4 结论"]
    assert len(stub.calls) == 3


def test_judge_tolerates_bad_items(monkeypatch):
    """单条样本返回垃圾时只影响该条，其余照常统计。"""
    samples = _only_first(3)
    stub = _StubJudgeClient(["完全不是 JSON", '{"reason": "缺 score"}', '{"score": 5, "reason": "ok"}'])
    monkeypatch.setattr("knowledge.utils.llm_client.get_llm_client", lambda *a, **k: stub)

    result = run_judge(samples)
    assert result["scored"] == 1
    assert result["mean_score"] == 5.0
    assert len(result["failed_calls"]) == 2


def test_judge_prompt_contains_evidence_and_expectation():
    sample = load_samples()[0]
    from knowledge.eval.dataset import predict_sample
    prompt = _judge_prompt(sample, predict_sample(sample))
    assert sample["question"] in prompt
    assert "系统给出的审查结论" in prompt
    assert "参考期望" in prompt
    assert "R1" in prompt


def test_judge_accepts_generated_reports(monkeypatch):
    """可选：把真实生成的报告文本一并送评。"""
    samples = _only_first(1)
    stub = _StubJudgeClient(['{"score": 2, "reason": "报告与证据不符"}'])
    monkeypatch.setattr("knowledge.utils.llm_client.get_llm_client", lambda *a, **k: stub)

    report_text = "本次审查未发现任何合规风险。"
    run_judge(samples, reports={samples[0]["id"]: report_text})
    sent = stub.calls[0]
    joined = str(sent)
    assert report_text in joined


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
