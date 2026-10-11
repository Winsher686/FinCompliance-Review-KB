"""评测数据集与指标回归测试（knowledge/eval/）。

这一组测试守的是「评测集本身的可信度」：
- 确定性指标必须满分（规则命中 / 风险等级 / HITL）——任何一条不达标说明规则或数据有问题；
- 元信息（evaluation.json）必须能从数据重算出来，杜绝「文档写了 50 条、文件里只有 8 条」这类漂移；
- 相对日期 token 必须可解析且**不随时间腐烂**（注入 today 复现边界）。

运行：pytest knowledge/test/compliance/test_eval_dataset.py -q
     或 python knowledge/test/compliance/test_eval_dataset.py
"""

import json
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from knowledge.eval.dataset import (
    dataset_meta, dataset_path, evaluate_dataset, expected_hitl, load_samples,
    predict_sample, resolve_date_token, resolve_dates,
)
from knowledge.processor.review_process.nodes.intent_recognize_node import VALID_INTENTS

_EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "eval"))

#: 评测集的规模下限（L3 要求 ≥ 50 条）
MIN_SAMPLES = 50


def _load_meta():
    with open(os.path.join(_EVAL_DIR, "evaluation.json"), "r", encoding="utf-8") as f:
        return json.load(f)


# ==================== 规模与结构 ====================

def test_dataset_scale_and_shape():
    samples = load_samples()
    assert len(samples) >= MIN_SAMPLES, f"评测集规模不足：{len(samples)} < {MIN_SAMPLES}"

    ids = [s["id"] for s in samples]
    assert len(ids) == len(set(ids)), "样本 id 重复"

    required = {"id", "company", "review_type", "question", "expect", "relations"}
    for s in samples:
        missing = required - set(s)
        assert not missing, f"{s.get('id')} 缺少字段：{missing}"
        assert set(s["expect"]) >= {"intent", "rules", "risk_level", "hitl"}, f"{s['id']} expect 字段不完整"
        assert s["expect"]["intent"] in VALID_INTENTS, f"{s['id']} 意图非法：{s['expect']['intent']}"
        for rel in s["relations"]:
            assert rel.get("head") and rel.get("rel") and rel.get("tail"), f"{s['id']} 关系字段不完整"


def test_dataset_covers_all_rules_and_both_polarities():
    """每条规则都要有正样本，且必须存在负样本（只有正样本测不出误报）。"""
    samples = load_samples()
    positives = {r for s in samples for r in s["expect"]["rules"]}
    assert positives == {"R1", "R2", "R3", "R4", "R5", "R6"}, f"规则覆盖不全：{sorted(positives)}"

    negatives = [s for s in samples if not s["expect"]["rules"]]
    assert len(negatives) >= 10, f"负样本太少（{len(negatives)}），无法度量误报"
    highs = [s for s in samples if s["expect"]["risk_level"] == "高"]
    assert len(highs) >= 5, "高风险样本过少，HITL 路径覆盖不足"


# ==================== 确定性指标 ====================

def test_deterministic_metrics_are_perfect():
    """规则命中 / 风险等级 / HITL 是确定性结论，必须 100% 一致。"""
    report = evaluate_dataset(load_samples())
    assert report.rule_hit_f1 == 1.0, "规则命中 F1 未达标：\n" + "\n".join(report.mismatches)
    assert report.rule_hit_precision == 1.0 and report.rule_hit_recall == 1.0
    assert report.risk_level_accuracy == 1.0
    assert report.hitl_accuracy == 1.0
    assert not report.false_positive_samples, f"存在误报样本：{report.false_positive_samples}"
    assert not report.missed_samples, f"存在漏检样本：{report.missed_samples}"
    # 每条规则都要在正样本上被召回（0 召回意味着该规则实际不可用）
    assert all(v == 1.0 for v in report.rule_recall_by_id.values()), report.rule_recall_by_id


def test_intent_fallback_accuracy_reported():
    """意图：离线仅关键词兜底，如实统计并打印，同时设下限防回归。"""
    report = evaluate_dataset(load_samples())
    print(f"[INFO] 意图（关键词兜底）准确率: {report.intent_accuracy:.1%}；偏差分布: {report.intent_confusion}")
    assert 0.0 <= report.intent_accuracy <= 1.0
    assert report.intent_accuracy >= 0.6, "关键词兜底意图准确率跌破 0.6，疑似规则被改坏"


def test_report_is_json_serialisable():
    report = evaluate_dataset(load_samples())
    payload = json.dumps(report.to_dict(), ensure_ascii=False)
    assert '"rule_hit_f1"' in payload
    assert "评测结果" not in payload      # to_dict 只给机器读，Markdown 走 to_markdown
    assert report.to_markdown().count("- 样本数") == 1


# ==================== 元信息一致性 ====================

def test_dataset_meta_consistency():
    """evaluation.json 的统计块必须能从 qa_pairs 重算出来（防文档漂移）。"""
    meta = _load_meta()
    samples = load_samples()
    recomputed = dataset_meta(samples)

    assert meta["total"] == recomputed["total"] == len(samples)
    assert meta["by_review_type"] == recomputed["by_review_type"]
    assert meta["by_intent"] == recomputed["by_intent"]
    assert meta["expected_rules"] == recomputed["expected_rules"]
    assert meta["expected_risk_levels"] == recomputed["expected_risk_levels"]
    assert meta["hitl_expectation"]["expected_hitl_ids"] == recomputed["expected_hitl_ids"]


def test_expected_scores_match_actual():
    """evaluation.json 里记录的期望分数必须与当前实现一致（改数据就要同步改分数）。"""
    meta = _load_meta()
    report = evaluate_dataset(load_samples())
    for key, actual in (
        ("intent_accuracy_offline", report.intent_accuracy),
        ("rule_hit_f1", report.rule_hit_f1),
        ("risk_level_accuracy", report.risk_level_accuracy),
        ("hitl_accuracy", report.hitl_accuracy),
    ):
        assert meta["expected_scores"][key] == round(actual, 4), f"{key} 与数据不一致"


def test_dataset_path_is_configurable(monkeypatch):
    monkeypatch.setenv("EVAL_QA_FILE", "/tmp/custom.jsonl")
    assert dataset_path() == "/tmp/custom.jsonl"
    monkeypatch.delenv("EVAL_QA_FILE")
    assert dataset_path().endswith(os.path.join("eval", "qa_pairs.jsonl"))


# ==================== 相对日期（防数据集腐烂） ====================

def test_resolve_relative_date_tokens():
    today = date(2026, 10, 11)
    assert resolve_date_token("@today", today) == "2026-10-11"
    assert resolve_date_token("@-30d", today) == "2026-09-11"
    assert resolve_date_token("@+365d", today) == "2027-10-11"
    assert resolve_date_token("@-2y", today) == "2024-10-11"
    # 非 token 原样返回（绝对日期、空值、数字都不受影响）
    assert resolve_date_token("2025-08-12", today) == "2025-08-12"
    assert resolve_date_token("", today) == ""
    assert resolve_date_token(None, today) is None


def test_resolve_dates_is_recursive():
    struct = {"a": ["@-1d", {"b": "@+1d"}], "c": 3}
    out = resolve_dates(struct, date(2026, 1, 10))
    assert out["a"][0] == "2026-01-09"
    assert out["a"][1]["b"] == "2026-01-11"
    assert out["c"] == 3


def test_r3_window_is_time_stable():
    """R3 的「近三年」窗口不应随真实时间推移而失效（相对日期 token 的意义）。

    注意：规则引擎用的是**真实今天**做窗口判定，所以数据集必须按真实今天解析；
    token 的价值在于「无论哪一天读取，@-2y 都是那一天的两年前」，永远落在 3 年窗口内。
    """
    sample = {
        "id": "t", "company": "X", "review_type": "violation", "question": "是否违规未整改？",
        "expect": {"intent": "compliance", "rules": ["R3"], "risk_level": "高", "hitl": True},
        "relations": [{"head": "X", "rel": "SANCTIONED_BY", "tail": "平台治理中心",
                       "props": {"reason": "虚假宣传", "date": "@-2y", "status": "未整改"}}],
    }
    # ① 引擎层：按真实今天解析时 R3 必须命中
    assert predict_sample(resolve_dates(sample, date.today()))["rules"] == {"R3"}

    # ② 机制层：解析结果恒等于「基准日 - 2 年」，因此永远是近 3 年内的事
    for base in (date(2026, 1, 1), date(2033, 7, 20)):
        stamp = resolve_dates(sample, base)["relations"][0]["props"]["date"]
        assert stamp == (base - timedelta(days=365 * 2)).isoformat()


def test_date_boundary_samples_reproduce():
    """资质到期边界：按真实今天解析后，@-1d 恒为过期、@+200d 恒为有效（不随日期腐烂）。"""
    today = date.today()
    expired = resolve_dates({"relations": [
        {"head": "X", "rel": "HOLDS_CERT", "tail": "食品经营许可证", "props": {"expire_date": "@-1d"}},
    ]}, today)
    valid = resolve_dates({"relations": [
        {"head": "X", "rel": "HOLDS_CERT", "tail": "食品经营许可证", "props": {"expire_date": "@+200d"}},
    ]}, today)
    assert predict_sample(expired)["rules"] == {"R2"}
    assert predict_sample(valid)["rules"] == set()


# ==================== HITL 口径 ====================

def test_expected_hitl_matches_node_rule():
    """HITL 期望值必须与 `compliance_check_node` 的判定口径一致。"""
    assert expected_hitl("general", "高") is True
    assert expected_hitl("general", "无") is False
    assert expected_hitl("report", "无") is True
    assert expected_hitl("violation", "无") is True
    assert expected_hitl("product_compliance", "低") is True
    assert expected_hitl("banned_word", "中") is False


def test_hitl_and_risk_are_consistent_in_dataset():
    """数据自洽：凡期望 hitl=True 的样本，必须由「高风险」或「HITL 审查类型」解释。"""
    for s in load_samples():
        expected = s["expect"]["hitl"]
        derived = expected_hitl(s["review_type"], s["expect"]["risk_level"])
        assert expected == derived, f"{s['id']} 的 hitl 期望与触发规则不符（应为 {derived}）"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("评测数据集测试全部通过")
