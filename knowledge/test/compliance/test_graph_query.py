"""图谱查询工具测试（knowledge/utils/neo4j_util.py）。

重点校验：跳数钳制、Cypher 生成（变长路径跳数为拼接、非参数）、
结果整理函数、以及**批量查询（N+1 消除）**的语句与分组逻辑。
无需 Neo4j 驱动（纯逻辑部分 + 无驱动时的降级路径）。
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from knowledge.utils.neo4j_util import (
    clamp_hops, build_path_cypher, build_entity_relations_cypher,
    build_entity_profile_cypher, simplify_path_records, simplify_relation_records,
    build_entity_relations_batch_cypher, build_paths_batch_cypher,
    dedupe_names, build_entity_pairs, group_relation_records, dedupe_paths,
    get_entity_relations_batch, find_paths_batch,
    DEFAULT_MAX_HOPS, MAX_ALLOWED_HOPS,
)


def test_clamp_hops_bounds():
    assert clamp_hops(3) == 3
    assert clamp_hops(0) == 1
    assert clamp_hops(99) == MAX_ALLOWED_HOPS
    assert clamp_hops("bad") == DEFAULT_MAX_HOPS
    assert clamp_hops(None) == DEFAULT_MAX_HOPS


def test_path_cypher_uses_clamped_hops():
    cypher = build_path_cypher(99)  # 应被钳到 5
    assert f"[*1..{MAX_ALLOWED_HOPS}]" in cypher
    assert "$from_name" in cypher and "$to_name" in cypher
    # 排除内部关系
    assert "MENTIONED_IN" in cypher


def test_entity_relations_cypher():
    cypher = build_entity_relations_cypher()
    assert "startNode(r) = n" in cypher
    assert "properties(r)" in cypher


def test_entity_profile_cypher():
    cypher = build_entity_profile_cypher()
    assert "labels(n)" in cypher and "properties(n)" in cypher


def test_simplify_path_records():
    raw = [{
        "nodes": ["森田", "ZZ旗舰店", "广州XX贸易有限公司"],
        "rels": ["AUTHORIZED_BY", "SHARED_BRAND"],
        "props": [{"level": "一级"}, {"brand": "森田"}],
        "hops": 2,
    }]
    paths = simplify_path_records(raw)
    assert len(paths) == 1
    assert paths[0]["hops"] == 2
    assert "森田" in paths[0]["path_text"]
    assert "AUTHORIZED_BY" in paths[0]["path_text"]


def test_simplify_relation_records():
    raw = [
        {"head": "森田", "rel": "AUTHORIZED_BY", "tail": "ZZ旗舰店", "props": {"level": "一级"}},
        {"head": "", "rel": "AUTHORIZED_BY", "tail": "X", "props": {}},  # 无效，应被过滤
    ]
    rels = simplify_relation_records(raw)
    assert len(rels) == 1
    assert rels[0]["head"] == "森田" and rels[0]["props"]["level"] == "一级"


def test_simplify_empty_records():
    assert simplify_path_records([]) == []
    assert simplify_relation_records(None) == []


# ==================== 批量查询（N+1 消除） ====================

def test_entity_relations_batch_cypher_uses_unwind():
    """关系批量查询：UNWIND 种子 + 子查询内 per_limit（保证语义与逐条查询一致）。"""
    cypher = build_entity_relations_batch_cypher()
    assert "UNWIND $names AS seed" in cypher
    assert "$per_limit" in cypher
    assert "RETURN seed" in cypher
    assert "properties(r)" in cypher
    # 没有 total_limit 时不应出现外层 LIMIT
    assert "$total_limit" not in cypher

    bounded = build_entity_relations_batch_cypher(with_item_name=True, with_total_limit=True)
    assert "$total_limit" in bounded
    assert "m.item_name = $item_name" in bounded


def test_paths_batch_cypher_clamps_hops():
    """路径批量查询：跳数仍为拼接（不可参数化），且被钳制到 [1, 5]。"""
    cypher = build_paths_batch_cypher(max_hops=99)
    assert "UNWIND $pairs AS pair" in cypher
    assert f"[*1..{MAX_ALLOWED_HOPS}]" in cypher
    assert "pair.from_name" in cypher and "pair.to_name" in cypher
    assert "MENTIONED_IN" in cypher


def test_dedupe_names_keeps_order():
    assert dedupe_names(["A", "B", "A", "  ", None, "C"]) == ["A", "B", "C"]
    assert dedupe_names(["A", "B", "C"], max_count=2) == ["A", "B"]
    assert dedupe_names(None) == []


def test_build_entity_pairs_is_combination():
    """两两组合：n 个实体 → C(n,2) 个实体对，取代原来的嵌套 for 循环。"""
    pairs = build_entity_pairs(["森田", "ZZ旗舰店", "广州XX贸易有限公司"])
    assert len(pairs) == 3
    assert pairs[0] == {"from_name": "森田", "to_name": "ZZ旗舰店"}
    assert {"from_name": "ZZ旗舰店", "to_name": "广州XX贸易有限公司"} in pairs
    # 单实体无需配对，避免空查询
    assert build_entity_pairs(["只有一个"]) == []
    # 超过 max_entities 时截断（限制组合爆炸）
    assert len(build_entity_pairs(["A", "B", "C", "D"], max_entities=3)) == 3


def test_group_relation_records():
    raw = [
        {"seed": "森田", "head": "森田", "rel": "AUTHORIZED_BY", "tail": "ZZ旗舰店", "props": {"level": "一级"}},
        {"seed": "森田", "head": "森田", "rel": "SHARED_BRAND", "tail": "广州XX贸易有限公司", "props": {}},
        {"seed": "ZZ旗舰店", "head": "ZZ旗舰店", "rel": "SELLS", "tail": "深海鱼油软胶囊", "props": {}},
        {"seed": "", "head": "X", "rel": "Y", "tail": "Z", "props": {}},        # 缺种子 → 丢弃
        {"seed": "森田", "head": "", "rel": "Y", "tail": "Z", "props": {}},      # 关系不完整 → 丢弃
    ]
    grouped = group_relation_records(raw)
    assert set(grouped) == {"森田", "ZZ旗舰店"}
    assert len(grouped["森田"]) == 2
    assert grouped["森田"][0]["props"]["level"] == "一级"
    assert grouped["ZZ旗舰店"][0]["rel"] == "SELLS"


def test_dedupe_paths_removes_reverse_duplicate():
    """无向路径会被 (A,B) 与 (B,A) 各命中一次，需按节点+关系序列去重。"""
    path = {"nodes": ["森田", "ZZ旗舰店"], "rels": ["AUTHORIZED_BY"], "hops": 1,
            "path_text": "森田 -[AUTHORIZED_BY]-> ZZ旗舰店"}
    other = {"nodes": ["森田", "广州XX贸易有限公司"], "rels": ["SHARED_BRAND"], "hops": 1,
             "path_text": "森田 -[SHARED_BRAND]-> 广州XX贸易有限公司"}
    assert len(dedupe_paths([path, dict(path), other])) == 2


def test_batch_query_without_driver_returns_empty():
    """无 Neo4j 驱动时必须降级为空结果（离线可跑），而不是抛异常。"""
    assert get_entity_relations_batch(["森田", "ZZ旗舰店"]) == {}
    assert find_paths_batch([{"from_name": "森田", "to_name": "ZZ旗舰店"}]) == []
    # 空输入不应触发查询
    assert get_entity_relations_batch([]) == {}
    assert find_paths_batch([]) == []
    assert find_paths_batch([{"from_name": "", "to_name": "X"}]) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("图谱查询工具测试全部通过")
