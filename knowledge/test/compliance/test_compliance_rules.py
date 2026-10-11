"""电商合规规则引擎测试（knowledge/processor/review_process/rules/compliance_rules.py）。

覆盖：
- 数值/比例/日期解析工具；
- R1 违禁词宣传、R2 资质缺失或过期、R3 违规未整改、R4 问题货源、
  R5 类目错放与价格违规、R6 品牌授权链穿透；
- 风险等级汇总（取最严重项）。

全部离线可跑（纯 Python，无 Milvus/Neo4j/LLM 依赖）。
"""

import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from knowledge.processor.review_process.rules.compliance_rules import (
    parse_amount, _parse_ratio, _parse_date, _is_true,
    build_context, evaluate_rules, overall_risk_level, SEVERITY_ORDER,
)

_MERCHANT = "广州XX贸易有限公司"


def _ids(findings):
    return {f.rule_id for f in findings}


def _evidence(findings, rule_id):
    return next(f for f in findings if f.rule_id == rule_id).evidence


# ==================== 解析工具 ====================

def test_parse_amount():
    assert parse_amount("1.2亿元") == 1.2e8
    assert parse_amount("5,000万") == 5000 * 1e4
    assert parse_amount("3000000") == 3000000.0
    assert parse_amount(1500) == 1500.0
    assert parse_amount("") is None
    assert parse_amount(None) is None


def test_parse_ratio():
    assert _parse_ratio("70%") == 0.7
    assert _parse_ratio("0.7") == 0.7
    assert _parse_ratio(0.5) == 0.5
    assert _parse_ratio("80") == 0.8      # 无百分号且数值 > 1 → 视为百分数
    assert _parse_ratio("") is None
    assert _parse_ratio(None) is None


def test_parse_date():
    assert _parse_date("2025-12-31") == date(2025, 12, 31)
    assert _parse_date("2025/12/31") == date(2025, 12, 31)
    assert _parse_date("2025年12月31日") == date(2025, 12, 31)
    assert _parse_date("2025-06") == date(2025, 6, 1)
    assert _parse_date("非法日期") is None
    assert _parse_date(None) is None


def test_is_true():
    for value in ("1", "true", "YES", "是", "有"):
        assert _is_true(value)
    for value in ("0", "false", "", "否", "无"):
        assert not _is_true(value)


# ==================== R1 违禁词/极限词宣传 ====================

def test_r1_banned_word_triggered():
    ctx = build_context(relations=[
        {"head": _MERCHANT, "rel": "PROMOTES", "tail": "深海鱼油软胶囊宣传",
         "props": {"banned_words": "国家级,最安全"}},
    ])
    findings = evaluate_rules(ctx)
    assert "R1" in _ids(findings)
    assert overall_risk_level(findings) == "高"
    assert _evidence(findings, "R1")


def test_r1_not_triggered_without_banned_words():
    ctx = build_context(relations=[
        {"head": _MERCHANT, "rel": "PROMOTES", "tail": "普通促销",
         "props": {"discount_rate": "10%"}},
    ])
    assert "R1" not in _ids(evaluate_rules(ctx))


# ==================== R2 资质缺失或过期 ====================

def test_r2_missing_and_expired():
    past = (date.today() - timedelta(days=30)).isoformat()
    ctx = build_context(relations=[
        {"head": _MERCHANT, "rel": "HOLDS_CERT", "tail": "食品经营许可证",
         "props": {"cert_type": "食品经营许可证", "status": "缺失"}},
        {"head": _MERCHANT, "rel": "HOLDS_CERT", "tail": "化妆品特证",
         "props": {"cert_type": "化妆品特证", "expire_date": past}},
    ])
    findings = evaluate_rules(ctx)
    assert "R2" in _ids(findings)
    assert len(_evidence(findings, "R2")) == 2
    assert overall_risk_level(findings) == "中"


def test_r2_not_triggered_when_valid():
    future = (date.today() + timedelta(days=365)).isoformat()
    ctx = build_context(relations=[
        {"head": _MERCHANT, "rel": "HOLDS_CERT", "tail": "食品经营许可证",
         "props": {"cert_type": "食品经营许可证", "expire_date": future, "status": "有效"}},
    ])
    assert "R2" not in _ids(evaluate_rules(ctx))


# ==================== R3 违规未整改 ====================

def test_r3_violation_unrectified():
    recent = (date.today() - timedelta(days=60)).isoformat()
    rels = [{"head": _MERCHANT, "rel": "SANCTIONED_BY", "tail": "平台治理中心",
             "props": {"reason": "虚假宣传", "amount": "5000元", "date": recent, "status": "未整改"}}]
    assert "R3" in _ids(evaluate_rules(build_context(relations=rels)))
    # 已整改（disclosed 关键字命中）→ 不命中
    ctx = build_context(relations=rels, disclosed={"已整改"})
    assert "R3" not in _ids(evaluate_rules(ctx))


def test_r3_not_triggered_outdated_or_rectified():
    old = (date.today() - timedelta(days=365 * 5)).isoformat()
    recent = (date.today() - timedelta(days=10)).isoformat()
    outdated = build_context(relations=[
        {"head": _MERCHANT, "rel": "SANCTIONED_BY", "tail": "平台治理中心",
         "props": {"reason": "历史违规", "amount": "1000元", "date": old, "status": "未整改"}},
    ])
    rectified = build_context(relations=[
        {"head": _MERCHANT, "rel": "SANCTIONED_BY", "tail": "平台治理中心",
         "props": {"reason": "虚假宣传", "amount": "5000元", "date": recent, "status": "已整改"}},
    ])
    assert "R3" not in _ids(evaluate_rules(outdated))
    assert "R3" not in _ids(evaluate_rules(rectified))


# ==================== R4 问题货源 ====================

def test_r4_problem_supplier():
    ctx = build_context(relations=[
        {"head": "YY供应链有限公司", "rel": "SUPPLIES", "tail": _MERCHANT,
         "props": {"product": "深海鱼油原料", "direction": "upstream"}},
        {"head": "YY供应链有限公司", "rel": "COMPLAINED_IN", "tail": "商品质量纠纷",
         "props": {"case_no": "TS-2024-0123", "cause": "商品质量纠纷", "amount": "3万元"}},
    ])
    findings = evaluate_rules(ctx)
    assert "R4" in _ids(findings)
    assert overall_risk_level(findings) == "中"


def test_r4_not_triggered_without_complaint():
    ctx = build_context(relations=[
        {"head": "YY供应链有限公司", "rel": "SUPPLIES", "tail": _MERCHANT, "props": {}},
    ])
    assert "R4" not in _ids(evaluate_rules(ctx))


# ==================== R5 类目错放与价格违规 ====================

def test_r5_misplaced_category_and_fake_price():
    ctx = build_context(relations=[
        {"head": "深海鱼油软胶囊", "rel": "BELONGS_TO", "tail": "普通食品",
         "props": {"category_code": "C1002", "is_misplaced": "true"}},
        {"head": _MERCHANT, "rel": "PROMOTES", "tail": "深海鱼油软胶囊促销",
         "props": {"discount_rate": "80%", "is_fake_price": "true"}},
    ])
    findings = evaluate_rules(ctx)
    assert "R5" in _ids(findings)
    assert len(_evidence(findings, "R5")) == 2
    assert overall_risk_level(findings) == "中"


def test_r5_discount_ratio_threshold():
    over = build_context(relations=[
        {"head": _MERCHANT, "rel": "PROMOTES", "tail": "促销", "props": {"discount_rate": "60%"}},
    ])
    under = build_context(relations=[
        {"head": _MERCHANT, "rel": "PROMOTES", "tail": "促销", "props": {"discount_rate": "30%"}},
    ])
    assert "R5" in _ids(evaluate_rules(over))       # 60% ≥ 阈值 50%
    assert "R5" not in _ids(evaluate_rules(under))  # 30% < 阈值 50%


# ==================== R6 品牌授权链穿透 ====================

def test_r6_brand_chain_from_relations():
    ctx = build_context(relations=[
        {"head": "森田", "rel": "AUTHORIZED_BY", "tail": "ZZ旗舰店", "props": {"level": "一级"}},
        {"head": "ZZ旗舰店", "rel": "SHARED_BRAND", "tail": _MERCHANT, "props": {"brand": "森田"}},
    ])
    findings = evaluate_rules(ctx)
    assert "R6" in _ids(findings)
    assert overall_risk_level(findings) == "低"


def test_r6_from_graph_paths():
    ctx = build_context(
        relations=[{"head": "森田", "rel": "AUTHORIZED_BY", "tail": "ZZ旗舰店", "props": {}},
                   {"head": "ZZ旗舰店", "rel": "AUTHORIZED_BY", "tail": _MERCHANT, "props": {}}],
        graph_paths=[{"nodes": ["森田", "ZZ旗舰店", _MERCHANT],
                      "rels": ["AUTHORIZED_BY", "AUTHORIZED_BY"], "hops": 2}],
    )
    assert "R6" in _ids(evaluate_rules(ctx))


def test_r6_single_level_not_triggered():
    ctx = build_context(relations=[
        {"head": "森田", "rel": "AUTHORIZED_BY", "tail": _MERCHANT, "props": {"level": "一级"}},
    ])
    assert "R6" not in _ids(evaluate_rules(ctx))


# ==================== 汇总与边界 ====================

def test_no_findings_means_no_risk():
    ctx = build_context(company=_MERCHANT, metrics={"gmv": 5_000_000.0, "listing_count": 120}, relations=[])
    findings = evaluate_rules(ctx)
    assert findings == []
    assert overall_risk_level(findings) == "无"


def test_overall_risk_takes_max_severity():
    ctx = build_context(relations=[
        {"head": _MERCHANT, "rel": "HOLDS_CERT", "tail": "食品经营许可证", "props": {"status": "缺失"}},  # R2 中
        {"head": _MERCHANT, "rel": "PROMOTES", "tail": "宣传", "props": {"banned_words": "最安全"}},       # R1 高
    ])
    findings = evaluate_rules(ctx)
    assert {"R1", "R2"} <= _ids(findings)
    assert overall_risk_level(findings) == "高"


def test_relations_of_accepts_type_key():
    """relations_of 兼容 rel / type 两种键名（图谱与导入侧口径不同）。"""
    ctx = build_context(relations=[{"head": "A", "type": "PROMOTES", "tail": "B",
                                    "props": {"banned_words": "最佳"}}])
    assert "R1" in _ids(evaluate_rules(ctx))


def test_severity_order_complete():
    assert SEVERITY_ORDER["高"] > SEVERITY_ORDER["中"] > SEVERITY_ORDER["低"] > SEVERITY_ORDER["无"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("电商合规规则测试全部通过")
