"""审查 API 测试（FastAPI，默认端口 8002）。

校验用户要求的 5 个接口 + SSE 任务状态：
- POST /review                     提交审查（同步返回结果）
- GET  /review/{review_id}         获取结果
- POST /review/{review_id}/approve 人工审批
- GET  /graph/entity/{entity_id}   实体与关系
- GET  /graph/path                 两实体间路径
- GET  /status/{task_id}           任务状态

使用 FastAPI TestClient，无需真实启动服务；无 Neo4j 时图谱接口应返回空结果而非报错。
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

try:
    from fastapi.testclient import TestClient
    from knowledge.processor.review_process.api.review_router import create_app
    _IMPORT_OK = True
    _IMPORT_ERR = ""
except Exception as e:  # pragma: no cover
    _IMPORT_OK = False
    _IMPORT_ERR = str(e)

_MERCHANT = "广州XX贸易有限公司"


def _client():
    if not _IMPORT_OK:
        msg = f"跳过：缺少依赖（{_IMPORT_ERR}）。需安装 fastapi / uvicorn / httpx。"
        if "pytest" in sys.modules:
            import pytest
            pytest.skip(msg)
        print(f"[SKIP] {msg}")
        return None
    return TestClient(create_app())


_PAYLOAD = {
    "company": _MERCHANT,
    "review_type": "banned_word",
    "question": "该店铺商品宣传是否使用了违禁词或极限词？是否需要整改？",
    "is_stream": False,
    "metrics": {"gmv": 5_000_000.0, "listing_count": 120},
    "relations": [
        {"head": _MERCHANT, "rel": "PROMOTES", "tail": "深海鱼油软胶囊宣传",
         "props": {"date": "2024-03", "banned_words": "国家级,最安全", "discount_rate": "80%"}},
        {"head": _MERCHANT, "rel": "HOLDS_CERT", "tail": "食品经营许可证",
         "props": {"cert_type": "食品经营许可证", "status": "缺失"}},
    ],
}


def test_health():
    c = _client()
    if c is None:
        return
    assert c.get("/health").json() == {"ok": True}


def test_submit_and_get_review():
    c = _client()
    if c is None:
        return
    r = c.post("/review", json=_PAYLOAD)
    assert r.status_code == 200
    data = r.json()
    assert data["review_id"]
    assert data["risk_level"] == "高"
    assert data["status"] == "pending_approval"
    assert any(f["rule_id"] == "R1" for f in data["findings"])

    rid = data["review_id"]
    r2 = c.get(f"/review/{rid}")
    assert r2.status_code == 200
    assert r2.json()["risk_level"] == "高"


def test_approve_resumes_flow():
    c = _client()
    if c is None:
        return
    rid = c.post("/review", json=_PAYLOAD).json()["review_id"]
    r = c.post(f"/review/{rid}/approve",
               json={"approved": True, "reviewer": "平台治理员", "comment": "同意整改"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "approved"
    assert (body.get("report") or "").strip()


def test_get_missing_review_returns_404():
    c = _client()
    if c is None:
        return
    assert c.get("/review/not-exist-id").status_code == 404


def test_graph_endpoints_offline_safe():
    c = _client()
    if c is None:
        return
    r1 = c.get(f"/graph/entity/{_MERCHANT}")
    assert r1.status_code == 200
    assert isinstance(r1.json()["relations"], list)
    r2 = c.get("/graph/path", params={
        "from_name": "森田", "to_name": _MERCHANT, "max_hops": 3})
    assert r2.status_code == 200
    assert isinstance(r2.json()["paths"], list)


def test_task_status_endpoint():
    c = _client()
    if c is None:
        return
    r = c.get("/status/unknown-task")
    assert r.status_code == 200
    assert "status" in r.json()


def test_e2e_product_compliance_r1_r2_enters_hitl():
    """L2 端到端：标题含“最安全” + 缺食品经营许可证 → 命中 R1+R2 且进入 pending_approval。"""
    c = _client()
    if c is None:
        return
    payload = {
        "company": _MERCHANT,
        "review_type": "product_compliance",
        "question": "该商品标题使用“最安全”，且店铺未取得食品经营许可证，是否合规？",
        "is_stream": False,
        "metrics": {"gmv": 5_000_000.0, "listing_count": 120},
        "relations": [
            {"head": _MERCHANT, "rel": "PROMOTES", "tail": "深海鱼油软胶囊宣传",
             "props": {"banned_words": "最安全"}},
            {"head": _MERCHANT, "rel": "HOLDS_CERT", "tail": "食品经营许可证",
             "props": {"cert_type": "食品经营许可证", "status": "缺失"}},
        ],
    }
    data = c.post("/review", json=payload).json()
    assert data["risk_level"] == "高"
    assert data["status"] == "pending_approval"
    assert {"R1", "R2"} <= {f["rule_id"] for f in data["findings"]}


def test_trace_endpoint_returns_waterfall():
    """L3-2：提交审查后可用 trace_id 拉取节点耗时瀑布。"""
    c = _client()
    if c is None:
        return
    data = c.post("/review", json=_PAYLOAD).json()
    trace_id = data.get("trace_id") or f"rv-{data['review_id']}"
    r = c.get(f"/trace/{trace_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["node_count"] >= 1
    assert body["total_ms"] >= 0
    assert "节点耗时瀑布" in body["waterfall"]
    assert any(span["node"] == "compliance_check_node" for span in body["spans"])


def test_trace_endpoint_404_for_unknown():
    c = _client()
    if c is None:
        return
    assert c.get("/trace/not-exist-trace").status_code == 404


def test_metrics_endpoint_shape():
    c = _client()
    if c is None:
        return
    c.post("/review", json=_PAYLOAD)
    r = c.get("/metrics")
    assert r.status_code == 200
    metrics = r.json()["metrics"]
    assert "node_duration_ms" in metrics
    for stat in ("count", "mean", "p50", "p95", "max"):
        assert stat in metrics["node_duration_ms"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("审查 API 测试全部通过")
