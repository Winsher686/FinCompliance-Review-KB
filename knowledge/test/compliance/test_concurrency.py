"""多路检索并发调度测试（knowledge/processor/review_process/nodes/multi_search_node.py）。

这一组测试不是「能不能跑通」，而是**证明并发确实发生了、且容错边界符合预期**：

1. 三路检索的起跑时刻互相重叠（并发证据），墙钟 ≈ max(单路) 而非 sum(单路)；
2. 任一路抛异常，其余两路结果照常返回（分支级异常隔离）；
3. 任一路超时，节点在超时上限附近返回，不把整条审查流程拖住；
4. 分支耗时被打点到指标，能回答「到底哪一路慢」。

离线可跑：全部用可控 sleep 的假分支，不依赖 Milvus / Neo4j / BGE-M3。
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pytest

from knowledge.processor.review_process.config import ReviewConfig
from knowledge.processor.review_process.nodes.multi_search_node import MultiSearchNode
from knowledge.utils.observability import reset_observability, snapshot_metrics


class _StubSearchNode(MultiSearchNode):
    """把三路检索替换成可配置延迟 / 可配置失败的假分支。"""

    def __init__(self, latencies=None, failures=None, config=None):
        super().__init__(config=config)
        self.latencies = dict(latencies or {})
        self.failures = dict(failures or {})
        self.entered_at = {}
        self.exited_at = {}

    # ---- 假分支 ----
    def _stub(self, branch, value):
        self.entered_at[branch] = time.monotonic()
        try:
            if branch in self.failures:
                raise RuntimeError(self.failures[branch])
            time.sleep(self.latencies.get(branch, 0))
            return value
        finally:
            self.exited_at[branch] = time.monotonic()

    def _vector_search(self, text, company):
        return self._stub("vector", [{"entity": {"chunk_id": "vec-1"}, "distance": 0.11}])

    def _sparse_search(self, text, company):
        return self._stub("sparse", [{"entity": {"chunk_id": "spa-1"}, "distance": 0.22}])

    def _kg_search(self, query, company):
        return self._stub("kg", (
            [{"entity": {"chunk_id": "graph_path_0"}, "distance": 1.0}],
            ["[森田] -(AUTHORIZED_BY)-> ZZ旗舰店"],
            [{"nodes": ["森田", "ZZ旗舰店"], "rels": ["AUTHORIZED_BY"], "hops": 1,
              "path_text": "森田 -[AUTHORIZED_BY]-> ZZ旗舰店"}],
        ))


def _config(branch_timeout: float = 20.0) -> ReviewConfig:
    cfg = ReviewConfig()
    cfg.search_branch_timeout = branch_timeout
    cfg.search_max_workers = 3
    return cfg


def _state() -> dict:
    return {
        "review_id": "rev-concurrency",
        "question": "森田品牌授权店铺是否存在共用品牌的情况",
        "rewritten_query": "森田 授权 店铺 共用品牌",
        "company": "广州XX贸易有限公司",
        "hyde_doc": "假设性文档：森田品牌授权给 ZZ旗舰店。",
    }


@pytest.fixture(autouse=True)
def _clean_metrics():
    reset_observability()
    yield
    reset_observability()


# ==================== 并发确实发生 ====================

def test_three_branches_start_concurrently():
    """三路检索的起跑时刻应互相重叠（真并发的直接证据）。"""
    node = _StubSearchNode(latencies={"vector": 0.30, "sparse": 0.30, "kg": 0.30}, config=_config())
    node.process(_state())

    assert set(node.entered_at) == {"vector", "sparse", "kg"}
    spread_ms = (max(node.entered_at.values()) - min(node.entered_at.values())) * 1000
    # 串行调度下三路的进入时刻会相差约一个分支耗时（≈300ms）；并发应在几十毫秒内齐发
    assert spread_ms < 150, f"三路未并发齐发，进入时刻跨度 {spread_ms:.0f}ms"


def test_parallel_wall_time_is_max_not_sum():
    """墙钟耗时应接近最慢一路，而不是三路之和。"""
    per_branch = 0.30
    node = _StubSearchNode(latencies={"vector": per_branch, "sparse": per_branch, "kg": per_branch},
                           config=_config())
    began = time.monotonic()
    result = node.process(_state())
    elapsed = time.monotonic() - began

    sequential = per_branch * 3
    assert elapsed < sequential * 0.75, f"并发收益不足：{elapsed:.2f}s vs 串行 {sequential:.2f}s"
    assert result["search_wall_ms"] == pytest.approx(per_branch * 1000, abs=250)


def test_results_still_mapped_to_correct_fields():
    """并发不改变字段契约：向量→embedding_chunks、稀疏→sparse_chunks、图谱→kg_*。"""
    node = _StubSearchNode(config=_config())
    result = node.process(_state())

    assert [c["entity"]["chunk_id"] for c in result["embedding_chunks"]] == ["vec-1"]
    assert [c["entity"]["chunk_id"] for c in result["sparse_chunks"]] == ["spa-1"]
    assert len(result["kg_chunks"]) == 1
    assert result["kg_triples"][0].startswith("[森田]")
    assert result["kg_paths"][0]["hops"] == 1


# ==================== 分支级容错 ====================

def test_one_branch_failure_keeps_others():
    """稀疏检索挂掉时，向量与图谱结果必须照常返回。"""
    node = _StubSearchNode(failures={"sparse": "milvus timeout"}, config=_config())
    result = node.process(_state())

    assert result["sparse_chunks"] == []
    assert result["embedding_chunks"], "向量检索结果不应被其他分支的失败影响"
    assert result["kg_triples"], "图谱结果不应被其他分支的失败影响"


def test_all_branches_failure_still_returns_shape():
    """三路全挂也不能抛异常，必须返回完整的空结构（离线可跑的前提）。"""
    node = _StubSearchNode(failures={"vector": "x", "sparse": "y", "kg": "z"}, config=_config())
    result = node.process(_state())

    assert result["embedding_chunks"] == [] and result["sparse_chunks"] == []
    assert result["kg_chunks"] == [] and result["kg_triples"] == [] and result["kg_paths"] == []


def test_branch_timeout_is_bounded_and_degraded():
    """慢分支超时后：整体在超时上限附近返回，该分支降级为空。"""
    node = _StubSearchNode(latencies={"vector": 0.05, "sparse": 0.05, "kg": 3.0},
                           config=_config(branch_timeout=0.4))
    began = time.monotonic()
    results, durations, errors = node.parallel_retrieve(node._build_jobs(_state()))
    elapsed = time.monotonic() - began

    assert elapsed < 1.5, f"超时分支拖住了整体：{elapsed:.2f}s"
    assert "timeout" in errors.get("kg", "")
    assert durations["kg"] == pytest.approx(400, abs=200)
    assert results["vector"] and results["sparse"]          # 正常分支不受影响
    assert results.get("kg") is None                        # 超时分支降级


def test_process_degrades_on_timeout_without_raising():
    """通过 process() 走一遍：超时分支降级为空列表而不是抛异常。"""
    node = _StubSearchNode(latencies={"kg": 3.0}, config=_config(branch_timeout=0.4))
    result = node.process(_state())

    assert result["kg_chunks"] == [] and result["kg_triples"] == [] and result["kg_paths"] == []
    assert result["embedding_chunks"]


# ==================== 打点 ====================

def test_branch_durations_are_recorded_as_metrics():
    """分支耗时进指标：回答「检索慢在哪一路」。"""
    node = _StubSearchNode(latencies={"vector": 0.05, "sparse": 0.10, "kg": 0.15}, config=_config())
    result = node.process(_state())

    metrics = snapshot_metrics()
    assert metrics["search_branch_ms"]["count"] >= 3
    assert metrics["search_wall_ms"]["count"] == 1
    assert set(result["search_branch_ms"]) == {"vector", "sparse", "kg"}
    assert result["search_branch_ms"]["vector"] >= 50


def test_span_extras_compacts_branch_ms():
    """瀑布旁注把分支耗时压成 `kg:150|sparse:100|vector:50`（慢的在前）。"""
    from knowledge.utils.observability import span_extras
    extras = span_extras({"search_branch_ms": {"vector": 50, "sparse": 100, "kg": 150},
                          "search_wall_ms": 152.4})
    assert extras["branches"] == "kg:150|sparse:100|vector:50"
    assert extras["wall_ms"] == 152


# ==================== 图谱分支的批量下推 ====================

def test_kg_branch_uses_batch_apis(monkeypatch):
    """图谱分支必须走批量接口（UNWIND 一次往返），不允许回退到逐实体查询。"""
    calls = {"batch_relations": 0, "batch_paths": 0, "single": 0}

    import knowledge.utils.neo4j_util as neo

    def _fake_batch_relations(names, item_name=None, limit_per_entity=50, total_limit=None):
        calls["batch_relations"] += 1
        return {name: [{"head": name, "rel": "AUTHORIZED_BY", "tail": "ZZ旗舰店", "props": {}}]
                for name in names}

    def _fake_batch_paths(pairs, item_name=None, max_hops=3, limit_per_pair=10, total_limit=None):
        calls["batch_paths"] += 1
        assert pairs, "实体对不应为空"
        return [{"nodes": ["森田", "ZZ旗舰店"], "rels": ["AUTHORIZED_BY"], "hops": 1,
                 "path_text": "森田 -[AUTHORIZED_BY]-> ZZ旗舰店"}]

    monkeypatch.setattr(neo, "get_entity_relations_batch", _fake_batch_relations)
    monkeypatch.setattr(neo, "find_paths_batch", _fake_batch_paths)
    monkeypatch.setattr(neo, "get_entity_relations", _no_single_node(calls))

    # 用真实 MultiSearchNode（不做桩），只把图谱工具的批量接口换成计数桩
    node = MultiSearchNode(config=_config())
    kg_chunks, triples, paths = node._kg_search("森田品牌授权 ZZ旗舰店 共用品牌", "广州XX贸易有限公司")

    assert calls["batch_relations"] == 1, "实体关系应一次批量取回"
    assert calls["batch_paths"] == 1, "路径应一次批量取回"
    assert calls["single"] == 0, "不应回退到逐实体查询（N+1）"
    assert triples and paths and kg_chunks


def _no_single_node(calls):
    """逐实体查询的哨兵：一旦被调用即说明批量下推失效。"""
    def _no_single(*_args, **_kwargs):
        calls["single"] += 1
        return []
    return _no_single


def test_kg_branch_degrades_without_neo4j():
    """无 Neo4j 驱动时图谱分支返回空三元组，而不是抛异常（离线可跑）。"""
    node = MultiSearchNode(config=_config())
    kg_chunks, triples, paths = node._kg_search("森田品牌授权 ZZ旗舰店 共用品牌", "广州XX贸易有限公司")
    assert kg_chunks == [] and triples == [] and paths == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
