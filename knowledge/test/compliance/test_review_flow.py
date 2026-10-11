"""审查流程（LangGraph + HITL）测试。

校验：
1. 高风险命中 → 图被 interrupt 暂停（status=pending_approval）；
2. Command(resume=...) 恢复 → 生成报告、review_status=approved；
3. 无风险 → 直接走完并完成（completed，无人工复核）。

全部离线可跑：检索/LLM 缺省时由各节点兜底，规则引擎为纯 Python。
"""

import os
import sys
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from knowledge.processor.review_process.main_graph import review_app, build_thread_config
from knowledge.processor.review_process.state import create_default_state

_MERCHANT = "广州XX贸易有限公司"

# 违禁词宣传（R1 高）+ 资质缺失（R2 中）→ 整体高风险，应触发 HITL
_HIGH_RISK_RELATIONS = [
    {"head": _MERCHANT, "rel": "PROMOTES", "tail": "深海鱼油软胶囊宣传",
     "props": {"date": "2024-03", "banned_words": "国家级,最安全"}},
    {"head": _MERCHANT, "rel": "HOLDS_CERT", "tail": "食品经营许可证",
     "props": {"cert_type": "食品经营许可证", "status": "缺失"}},
]


def _run(relations, review_type="banned_word", metrics=None):
    review_id = f"test-{uuid.uuid4()}"
    state = create_default_state(
        review_id=review_id, task_id="test-task",
        company=_MERCHANT, review_type=review_type,
        question="该店铺商品宣传是否使用违禁词？资质是否齐全？", is_stream=False,
        metrics=metrics if metrics is not None else {"gmv": 5_000_000.0, "listing_count": 120},
        relations=relations,
    )
    snapshot = review_app.invoke(state, config=build_thread_config(review_id))
    return review_id, snapshot


def test_high_risk_triggers_hitl_interrupt():
    review_id, snap = _run(_HIGH_RISK_RELATIONS)
    assert bool(snap.get("__interrupt__")), "高风险应触发 interrupt"
    assert snap.get("risk_level") == "高"
    assert snap.get("needs_human_review") is True
    # 证据链（关系型兜底）应已构建
    assert len(snap.get("evidence_chain") or []) >= 1
    # 中断时报告尚未生成
    assert not (snap.get("report") or "").strip()


def test_resume_after_interrupt_generates_report():
    from langgraph.types import Command
    review_id, snap = _run(_HIGH_RISK_RELATIONS)
    assert bool(snap.get("__interrupt__"))
    resumed = review_app.invoke(
        Command(resume={"approved": True, "reviewer": "测试员", "comment": "同意整改"}),
        config=build_thread_config(review_id),
    )
    assert resumed.get("review_status") == "approved"
    assert (resumed.get("report") or "").strip(), "审批后应生成报告"
    approval = resumed.get("approval") or {}
    assert approval.get("approved") is True
    assert approval.get("reviewer") == "测试员"


def test_low_risk_completes_without_hitl():
    _, snap = _run(relations=[], review_type="general", metrics={})
    assert not bool(snap.get("__interrupt__"))
    assert snap.get("risk_level") == "无"
    assert snap.get("review_status") == "completed"
    assert (snap.get("report") or "").strip()


def test_review_type_report_always_triggers_hitl():
    """L2 回归：review_type=report 属 HITL_REVIEW_TYPES 默认集合，即使无风险也需人工复核。"""
    _, snap = _run(relations=[], review_type="report", metrics={})
    assert bool(snap.get("__interrupt__")), "report 审查类型必须触发 interrupt"
    assert snap.get("needs_human_review") is True


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("审查流程测试全部通过")
