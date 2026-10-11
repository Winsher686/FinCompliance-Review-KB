"""检索并发化 / Neo4j N+1 消除 的前后对比基准（离线可跑，无需 Milvus / Neo4j）。

回答两个问题，并给出可复现的数字：

1. **三路检索并发 vs 串行**：同一组「单路耗时」，串行墙钟 = sum，并发墙钟 = max。
2. **Neo4j 往返次数**：逐实体/逐实体对查询（N+1）vs 批量下推（UNWIND），
   在给定跨机房 RTT 下的墙钟差异。

用法::

    python knowledge/eval/bench_search_concurrency.py
    python knowledge/eval/bench_search_concurrency.py --rtt 80 --repeat 5

说明（避免误读数字）：脚本里的检索分支与 Neo4j 往返都用 `time.sleep` **模拟**网络/模型耗时，
因此它衡量的是**调度与往返结构**带来的差异（这才是代码改动能决定的部分），
而不是某台机器的绝对性能。真实部署下把 --branch-ms / --rtt 换成实测值即可重跑。
"""

import argparse
import os
import statistics
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from knowledge.processor.review_process.config import ReviewConfig
from knowledge.processor.review_process.nodes.multi_search_node import MultiSearchNode

# 三路检索的模拟单路耗时（ms）：向量 / 稀疏 / 图谱
DEFAULT_BRANCH_MS = {"vector": 400, "sparse": 300, "kg": 900}
DEFAULT_RTT_MS = 50
# 基准用的审查问题与主体：确保能抽出 3 个实体（→ 3 个实体对）
BENCH_QUERY = "森田品牌授权 ZZ旗舰店 与 广州XX贸易有限公司 是否共用品牌"
BENCH_COMPANY = "广州XX贸易有限公司"


# ==================== 1. 检索并发 ====================

class _TimedSearchNode(MultiSearchNode):
    """把三路检索换成按 branch_ms 睡眠的桩，用于测量调度开销。"""

    def __init__(self, branch_ms: Dict[str, float], config: ReviewConfig):
        super().__init__(config=config)
        self.branch_ms = branch_ms

    def _sleep_branch(self, branch: str) -> None:
        time.sleep(self.branch_ms.get(branch, 0) / 1000.0)

    def _vector_search(self, text, company):
        return self._run("vector", [{"entity": {"chunk_id": "vec"}, "distance": 0.1}])

    def _sparse_search(self, text, company):
        return self._run("sparse", [{"entity": {"chunk_id": "spa"}, "distance": 0.2}])

    def _kg_search(self, query, company):
        return self._run("kg", (
            [{"entity": {"chunk_id": "graph_path_0"}, "distance": 1.0}],
            ["[森田] -(AUTHORIZED_BY)-> ZZ旗舰店"],
            [{"nodes": ["森田", "ZZ旗舰店"], "rels": ["AUTHORIZED_BY"], "hops": 1, "path_text": "p"}],
        ))

    def _run(self, branch: str, value: Any):
        time.sleep(self.branch_ms.get(branch, 0) / 1000.0)
        return value


def _measure_sequential(node: _TimedSearchNode, state: Dict[str, Any]) -> float:
    """按原实现顺序调用三路检索。"""
    query = (state.get("rewritten_query") or state.get("question") or "").strip()
    company = (state.get("company") or "").strip()
    began = time.monotonic()
    node._vector_search(f"{query}\n{state.get('hyde_doc', '')}".strip(), company)
    node._sparse_search(query, company)
    node._kg_search(query, company)
    return (time.monotonic() - began) * 1000


def _measure_parallel(node: _TimedSearchNode, state: Dict[str, Any]) -> float:
    """走真实并发调度入口。"""
    began = time.monotonic()
    node.parallel_retrieve(node._build_jobs(state))
    return (time.monotonic() - began) * 1000


def bench_search(branch_ms: Dict[str, float], repeat: int) -> Dict[str, Any]:
    cfg = ReviewConfig()
    cfg.search_branch_timeout = max(60.0, sum(branch_ms.values()) / 1000.0 * 4)
    cfg.search_max_workers = 3
    node = _TimedSearchNode(branch_ms, cfg)
    state = {
        "review_id": "bench",
        "question": "森田品牌授权店铺是否存在共用品牌的情况",
        "rewritten_query": "森田 授权 店铺 共用品牌",
        "company": "广州XX贸易有限公司",
        "hyde_doc": "假设性文档",
    }
    seq = [_measure_sequential(node, state) for _ in range(repeat)]
    par = [_measure_parallel(node, state) for _ in range(repeat)]
    return {
        "branch_ms": branch_ms,
        "sequential_ms": statistics.median(seq),
        "parallel_ms": statistics.median(par),
        "speedup": statistics.median(seq) / max(statistics.median(par), 1e-6),
    }


# ==================== 2. Neo4j 往返 ====================

class _FakeResult:
    def __init__(self, records: List[Dict[str, Any]]):
        self._records = records

    def data(self) -> List[Dict[str, Any]]:
        return self._records


class _FakeTx:
    """记录每次 run()，并按 rtt 模拟一次网络往返。"""

    def __init__(self, rtt_ms: float, stats: Dict[str, Any]):
        self.rtt_ms = rtt_ms
        self.stats = stats

    def run(self, cypher: str, **params: Any) -> _FakeResult:
        self.stats["roundtrips"] += 1
        time.sleep(self.rtt_ms / 1000.0)
        if "UNWIND $names" in cypher:
            return _FakeResult([
                {"seed": name, "head": name, "rel": "AUTHORIZED_BY", "tail": "ZZ旗舰店", "props": {}}
                for name in params.get("names", [])
            ])
        if "UNWIND $pairs" in cypher:
            pair = (params.get("pairs") or [{}])[0]
            return _FakeResult([{
                "nodes": [pair.get("from_name", "A"), pair.get("to_name", "B")],
                "rels": ["AUTHORIZED_BY"], "props": [{}], "hops": 1,
            }])
        if "startNode(r)" in cypher:
            name = params.get("name", "")
            return _FakeResult([{"head": name, "rel": "AUTHORIZED_BY", "tail": "ZZ旗舰店", "props": {}}])
        return _FakeResult([{
            "nodes": [params.get("from_name", "A"), params.get("to_name", "B")],
            "rels": ["AUTHORIZED_BY"], "props": [{}], "hops": 1,
        }])


class _FakeSession:
    def __init__(self, rtt_ms: float, stats: Dict[str, Any]):
        self.rtt_ms = rtt_ms
        self.stats = stats

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def execute_read(self, fn):
        return fn(_FakeTx(self.rtt_ms, self.stats))


def _install_fake_session(neo, rtt_ms: float) -> Dict[str, Any]:
    stats = {"sessions": 0, "roundtrips": 0}

    def _session():
        stats["sessions"] += 1
        return _FakeSession(rtt_ms, stats)

    neo._session = _session
    return stats


def _baseline_kg_search(query: str, company: str, max_relations: int = 50,
                        max_hops: int = 3, max_paths: int = 5) -> List[Dict[str, Any]]:
    """整改前 `MultiSearchNode._kg_search` 的实现（逐实体 / 逐实体对各开一次 session）。

    仅用于对比基准，保留原实现的结构：实体数 n → 关系 1 次 + 路径 C(n,2) 次往返。
    """
    from knowledge.utils.neo4j_util import find_path_between, get_entity_relations

    entities = MultiSearchNode._extract_entities(query, company)
    relations: List[Dict[str, Any]] = []
    if company:
        relations = get_entity_relations(company, item_name=company, limit=max_relations)
    if not relations and entities:
        relations = get_entity_relations(entities[0], limit=max_relations)

    names = entities[:3]
    paths: List[Dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            paths.extend(find_path_between(names[i], names[j], max_hops=max_hops, limit=max_paths))
            if len(paths) >= max_paths:
                break
    return paths[:max_paths]


def bench_neo4j(rtt_ms: float, query: str, company: str) -> Dict[str, Any]:
    """对比「逐实体/逐实体对查询」与「UNWIND 批量下推」的往返次数与墙钟。

    两端都跑真实的图谱分支代码（整改前实现以函数形式保留在本脚本内），
    只把 session 换成「按 rtt 睡眠并计数」的假实现，因此差异只来自**往返结构**。
    """
    import knowledge.utils.neo4j_util as neo

    cfg = ReviewConfig()
    cfg.kg_max_relations = 50
    cfg.kg_max_paths = 5
    cfg.kg_max_path_hops = 3
    entities = MultiSearchNode._extract_entities(query, company)

    # ---- 整改前：逐实体 / 逐实体对（N+1） ----
    stats_n1 = _install_fake_session(neo, rtt_ms)
    began = time.monotonic()
    _baseline_kg_search(query, company)
    n1_ms = (time.monotonic() - began) * 1000
    n1_roundtrips = stats_n1["roundtrips"]

    # ---- 整改后：UNWIND 批量下推 ----
    stats_batch = _install_fake_session(neo, rtt_ms)
    node = MultiSearchNode(config=cfg)
    began = time.monotonic()
    node._kg_search(query, company)
    batch_ms = (time.monotonic() - began) * 1000
    batch_roundtrips = stats_batch["roundtrips"]

    return {
        "rtt_ms": rtt_ms,
        "entities": len(entities),
        "pairs": len(neo.build_entity_pairs(entities, max_entities=3)),
        "n1_roundtrips": n1_roundtrips,
        "n1_ms": n1_ms,
        "batch_roundtrips": batch_roundtrips,
        "batch_ms": batch_ms,
        "saved_pct": (1 - batch_ms / max(n1_ms, 1e-6)) * 100,
    }


# ==================== 输出 ====================

def _render_search(res: Dict[str, Any]) -> str:
    b = res["branch_ms"]
    lines = [
        "| 方案 | 单路耗时 向量/稀疏/图谱 | 墙钟(ms) | 相对串行 |",
        "| --- | --- | --- | --- |",
        f"| 串行（整改前） | {b['vector']:.0f}/{b['sparse']:.0f}/{b['kg']:.0f} | "
        f"{res['sequential_ms']:.0f} | 1.00x |",
        f"| 并发（整改后） | {b['vector']:.0f}/{b['sparse']:.0f}/{b['kg']:.0f} | "
        f"{res['parallel_ms']:.0f} | **{res['speedup']:.2f}x** |",
    ]
    return "\n".join(lines)


def _render_neo4j(res: Dict[str, Any]) -> str:
    return "\n".join([
        "| 方案 | 往返次数 | 墙钟(ms) | 提速 |",
        "| --- | --- | --- | --- |",
        f"| 逐实体 + 逐实体对（整改前, N+1） | {res['n1_roundtrips']} | {res['n1_ms']:.0f} | 1.00x |",
        f"| UNWIND 批量下推（整改后） | {res['batch_roundtrips']} | {res['batch_ms']:.0f} | "
        f"**{res['n1_ms'] / max(res['batch_ms'], 1e-6):.2f}x**（省 {res['saved_pct']:.0f}% 耗时） |",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="检索并发 / Neo4j 往返 前后对比基准")
    parser.add_argument("--branch-ms", default="400,300,900",
                        help="三路检索模拟耗时（ms），顺序：向量,稀疏,图谱")
    parser.add_argument("--rtt", type=float, default=DEFAULT_RTT_MS, help="模拟的 Neo4j 单次往返耗时（ms）")
    parser.add_argument("--repeat", type=int, default=3, help="每项重复次数（取中位数）")
    args = parser.parse_args()

    vec, sparse, kg = (float(x) for x in args.branch_ms.split(","))
    branch_ms = {"vector": vec, "sparse": sparse, "kg": kg}

    print("=" * 72)
    print("检索并发基准（MultiSearchNode：向量 + 稀疏 + 图谱）")
    print("=" * 72)
    search = bench_search(branch_ms, args.repeat)
    print(_render_search(search))
    print()

    print("=" * 72)
    print(f"Neo4j 往返基准（模拟 RTT = {args.rtt:.0f} ms，主体：{BENCH_COMPANY}）")
    print("=" * 72)
    graph = bench_neo4j(args.rtt, BENCH_QUERY, BENCH_COMPANY)
    print(_render_neo4j(graph))
    print()
    print("注：单路耗时与 RTT 均为 -- 参数模拟值，衡量的是「调度/往返结构」差异，非机器绝对性能。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
