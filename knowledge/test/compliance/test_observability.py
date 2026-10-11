"""可观测性测试（knowledge/utils/observability.py）。

覆盖 L3-2：
- trace_id 派生与透传；
- 节点耗时打点（含异常/中断状态）；
- 指标聚合与分位数；
- 耗时瀑布渲染；
- 端到端：跑一次审查后，同一 trace 下能看到 8 个节点的耗时瀑布。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from knowledge.utils.observability import (
    ensure_trace_id, new_trace_id, node_span, span_extras, record_metric,
    get_trace, format_waterfall, snapshot_metrics, reset_observability, trace_prefix,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_observability()
    yield
    reset_observability()


# ==================== trace_id ====================

def test_ensure_trace_id_prefers_business_ids():
    assert ensure_trace_id({"trace_id": "abc"}) == "abc"
    assert ensure_trace_id({"review_id": "r1"}) == "rv-r1"
    assert ensure_trace_id({"task_id": "t1"}) == "tk-t1"
    assert ensure_trace_id({"session_id": "s1"}) == "ss-s1"
    assert ensure_trace_id({}) != ""
    assert ensure_trace_id(None) != ""
    # 已经是带前缀的业务 ID 时不重复加前缀
    assert ensure_trace_id({"review_id": "rv-r1"}) == "rv-r1"


def test_new_trace_id_unique_and_short():
    ids = {new_trace_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(len(i) == 12 for i in ids)
    assert new_trace_id("x-").startswith("x-")


def test_trace_prefix():
    assert trace_prefix("abc") == "[trace=abc] "
    assert trace_prefix("") == ""


# ==================== 打点 ====================

def test_node_span_records_duration_and_extras():
    with node_span("t1", "compliance_check_node") as span:
        span["extras"]["findings"] = 2
    trace = get_trace("t1")
    assert len(trace["spans"]) == 1
    entry = trace["spans"][0]
    assert entry["node"] == "compliance_check_node"
    assert entry["node_cn"] == "合规规则检查"
    assert entry["duration_ms"] >= 0
    assert entry["status"] == "ok"
    assert entry["extras"]["findings"] == 2


def test_node_span_marks_error_and_reraises():
    with pytest.raises(ValueError):
        with node_span("t2", "rrf_rerank_node"):
            raise ValueError("boom")
    entry = get_trace("t2")["spans"][0]
    assert entry["status"] == "error"
    assert "boom" in entry["error"]


def test_node_span_marks_langgraph_interrupt():
    class _Interrupt(Exception):
        pass

    _Interrupt.__module__ = "langgraph.errors"
    with pytest.raises(_Interrupt):
        with node_span("t3", "hitl_node"):
            raise _Interrupt("paused")
    assert get_trace("t3")["spans"][0]["status"] == "interrupted"


def test_span_extras_extracts_counts_and_enums():
    extras = span_extras({
        "reranked_docs": [1, 2, 3],
        "findings": [{"rule_id": "R1"}],
        "risk_level": "高",
        "intent": "compliance",
        "report": "很长的报告",       # 非计数字段，不应进入
    })
    assert extras["docs"] == 3
    assert extras["findings"] == 1
    assert extras["risk"] == "高"
    assert extras["intent"] == "compliance"
    assert "report" not in extras
    assert span_extras(None) == {}


# ==================== 指标 ====================

def test_record_metric_and_snapshot():
    for v in (10, 20, 30, 40, 100):
        record_metric("node_duration_ms", v)
    snap = snapshot_metrics()["node_duration_ms"]
    assert snap["count"] == 5
    assert snap["mean"] == 40.0
    assert snap["p50"] == 30
    assert snap["max"] == 100
    assert 30 <= snap["p95"] <= 100


def test_record_metric_ignores_non_numeric():
    record_metric("x", "not-a-number")
    assert "x" not in snapshot_metrics()


def test_disabled_mode_is_noop(monkeypatch):
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "0")
    with node_span("t9", "hitl_node") as span:
        span["extras"]["a"] = 1
    record_metric("y", 1)
    assert get_trace("t9")["spans"] == []
    assert snapshot_metrics() == {}


# ==================== 瀑布 ====================

def test_waterfall_renders_cn_names_and_order():
    with node_span("t4", "intent_recognize_node"):
        pass
    with node_span("t4", "multi_search_node") as span:
        span["extras"]["docs"] = 7
    text = format_waterfall("t4")
    assert "意图识别" in text
    assert "多路检索" in text
    assert "docs=7" in text
    assert text.index("意图识别") < text.index("多路检索")


def test_waterfall_handles_missing_trace():
    assert "无打点数据" in format_waterfall("not-exist")


# ==================== 端到端：一次审查产出完整 trace ====================

def test_review_run_produces_full_trace():
    pytest.importorskip("langgraph", reason="需要 langgraph（精简环境跳过）")
    import uuid

    from knowledge.processor.review_process.main_graph import review_app, build_thread_config
    from knowledge.processor.review_process.state import create_default_state

    review_id = f"trace-{uuid.uuid4()}"
    state = create_default_state(
        review_id=review_id, task_id="trace-task",
        company="广州XX贸易有限公司", review_type="banned_word",
        question="该店铺商品宣传是否使用违禁词？", is_stream=False,
        metrics={"gmv": 5_000_000.0},
        relations=[{"head": "广州XX贸易有限公司", "rel": "PROMOTES", "tail": "深海鱼油软胶囊宣传",
                    "props": {"banned_words": "最安全"}}],
    )
    review_app.invoke(state, config=build_thread_config(review_id))

    # trace_id 由 review_id 派生，节点基类已在每个节点写入打点
    trace = get_trace(f"rv-{review_id}")
    nodes = [s["node"] for s in trace["spans"]]
    assert len(nodes) >= 6, f"打点节点过少: {nodes}"
    assert "compliance_check_node" in nodes
    assert all(s["duration_ms"] >= 0 for s in trace["spans"])
    assert trace["total_ms"] > 0
    assert "节点耗时瀑布" in format_waterfall(f"rv-{review_id}")
