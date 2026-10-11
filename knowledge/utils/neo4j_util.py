"""Neo4j 工具（驱动管理 + 电商图谱查询）。

设计要点（电商合规）：
1. 驱动导入改为「可选」——未安装 neo4j 驱动时模块仍可导入，
   便于纯逻辑单元测试离线运行（真正的查询函数在无驱动时返回空结果）。
2. 图谱查询：多跳路径（品牌→授权店铺→商家）、实体关系、实体画像。
   这些能力支撑「品牌与商家之间是否存在授权链路径」「店铺通过哪些层级共用品牌」等合规问题。
3. **批量查询（N+1 消除）**：一次审查要遍历「主体 + 2~3 个实体」的一跳关系、以及实体两两之间的路径。
   若逐个实体开 session，就是典型的 N+1 往返（实体数 × 网络 RTT），在跨机房部署下延迟线性放大。
   因此提供 ``get_entity_relations_batch`` / ``find_paths_batch``：用 ``UNWIND $names`` /
   ``UNWIND $pairs`` 把 N 次往返压成 1 次，并用子查询内的 ``LIMIT $per_limit`` 保证每个种子
   的返回量仍然受控（等价于逐条查询的 limit 语义）。

注意：Cypher 中变长路径 `[*1..N]` 的跳数不能用参数占位，必须把经过校验的
整数拼接进语句（本项目对 max_hops 做了 1..5 的钳制，避免注入与性能问题）。
"""

import os
import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# 驱动为可选依赖，缺失时模块仍可导入（离线测试友好）
try:  # pragma: no cover - 取决于运行环境
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover
    GraphDatabase = None  # type: ignore

_neo4j_driver = None

# 图谱查询的默认上限
DEFAULT_MAX_HOPS = 3
MAX_ALLOWED_HOPS = 5
DEFAULT_PATH_LIMIT = 10
DEFAULT_RELATION_LIMIT = 50


def get_neo4j_driver():
    """
    获取 Neo4j 驱动实例（单例模式）
    """
    global _neo4j_driver
    if GraphDatabase is None:
        logger.warning("neo4j 驱动未安装，无法连接图数据库（如需图谱能力请安装 neo4j）")
        return None
    try:
        if _neo4j_driver is None:
            uri = os.getenv("NEO4J_URI")
            username = os.getenv("NEO4J_USERNAME")
            password = os.getenv("NEO4J_PASSWORD")

            logger.info(f"正在初始化 Neo4j 驱动，连接 URI: {uri}")
            _neo4j_driver = GraphDatabase.driver(
                uri=uri,
                auth=(username, password)
            )
            # Neo4j 驱动默认是懒加载，这行代码能确保如果账号密码错误或网络不通，当场就会抛出异常，而不是等到插入数据时才报错。
            _neo4j_driver.verify_connectivity()
            logger.info("Neo4j 驱动初始化成功并已验证连接！")
        return _neo4j_driver

    except Exception as e:
        # exc_info=True 会在日志中打印出完整的 Error Traceback 堆栈，方便排查到底是密码错还是网络不通
        logger.error(msg=f"初始化 Neo4j 驱动失败: {e}", exc_info=True)
        return None


# ==================== 纯逻辑辅助（可离线单元测试） ====================

def clamp_hops(max_hops: Any) -> int:
    """把跳数钳制到 [1, MAX_ALLOWED_HOPS]，非法值回退默认。"""
    try:
        hops = int(max_hops)
    except (TypeError, ValueError):
        return DEFAULT_MAX_HOPS
    return max(1, min(hops, MAX_ALLOWED_HOPS))


def build_path_cypher(max_hops: int = DEFAULT_MAX_HOPS, with_item_name: bool = False) -> str:
    """构造「两实体间多跳路径」查询语句。

    Args:
        max_hops: 最长跳数（已在调用方钳制）
        with_item_name: 是否附加 item_name（主体）过滤
    """
    hops = clamp_hops(max_hops)
    item_filter = "AND a.item_name = $item_name AND b.item_name = $item_name" if with_item_name else ""
    return f"""
MATCH p = (a:Entity {{name: $from_name}})-[*1..{hops}]-(b:Entity {{name: $to_name}})
WHERE a <> b
  AND ALL(r IN relationships(p) WHERE type(r) <> 'MENTIONED_IN')
  {item_filter}
RETURN
  [n IN nodes(p) | n.name] AS nodes,
  [r IN relationships(p) | type(r)] AS rels,
  [r IN relationships(p) | properties(r)] AS props,
  length(p) AS hops
ORDER BY hops ASC
LIMIT $limit
"""


def build_entity_relations_cypher(with_item_name: bool = False) -> str:
    """构造「实体一跳关系」查询语句（双向、带关系属性）。"""
    item_filter = "AND m.item_name = $item_name" if with_item_name else ""
    return f"""
MATCH (n:Entity {{name: $name}})-[r]-(m:Entity)
WHERE type(r) <> 'MENTIONED_IN'
  {item_filter}
RETURN
  CASE WHEN startNode(r) = n THEN n.name ELSE m.name END AS head,
  type(r) AS rel,
  CASE WHEN startNode(r) = n THEN m.name ELSE n.name END AS tail,
  properties(r) AS props
LIMIT $limit
"""


def build_entity_relations_batch_cypher(with_item_name: bool = False, with_total_limit: bool = False) -> str:
    """构造「多实体一跳关系」批量查询语句（UNWIND 消除 N+1）。

    与 :func:`build_entity_relations_cypher` 语义等价，区别在于：
    - 一次往返即可取回多个种子的关系（``UNWIND $names AS seed``）；
    - ``LIMIT $per_limit`` 写在子查询内，因此是**每个种子**各自的条数上限，
      而不是总量上限（若写在外层，先到的种子会吃掉全部配额）。

    Args:
        with_item_name: 是否附加 item_name（主体）过滤
        with_total_limit: 是否附加外层总条数上限（防止实体数放大后总量失控）
    """
    item_filter = "AND m.item_name = $item_name" if with_item_name else ""
    total_clause = "LIMIT $total_limit" if with_total_limit else ""
    return f"""
UNWIND $names AS seed
CALL {{
  WITH seed
  MATCH (n:Entity {{name: seed}})-[r]-(m:Entity)
  WHERE type(r) <> 'MENTIONED_IN'
    {item_filter}
  RETURN
    CASE WHEN startNode(r) = n THEN n.name ELSE m.name END AS head,
    type(r) AS rel,
    CASE WHEN startNode(r) = n THEN m.name ELSE n.name END AS tail,
    properties(r) AS props
  LIMIT $per_limit
}}
RETURN seed, head, rel, tail, props
{total_clause}
"""


def build_paths_batch_cypher(max_hops: int = DEFAULT_MAX_HOPS, with_item_name: bool = False,
                             with_total_limit: bool = False) -> str:
    """构造「实体两两多跳路径」批量查询语句（UNWIND 消除 N×N 次往返）。

    调用方先构造 ``$pairs = [{"from_name": a, "to_name": b}, ...]``，
    一次往返取回所有实体对的路径。``LIMIT $per_limit`` 位于子查询内，语义为「每个实体对的上限」。
    """
    hops = clamp_hops(max_hops)
    item_filter = "AND a.item_name = $item_name AND b.item_name = $item_name" if with_item_name else ""
    total_clause = "LIMIT $total_limit" if with_total_limit else ""
    return f"""
UNWIND $pairs AS pair
CALL {{
  WITH pair
  MATCH p = (a:Entity {{name: pair.from_name}})-[*1..{hops}]-(b:Entity {{name: pair.to_name}})
  WHERE a <> b
    AND ALL(r IN relationships(p) WHERE type(r) <> 'MENTIONED_IN')
    {item_filter}
  RETURN
    [n IN nodes(p) | n.name] AS nodes,
    [r IN relationships(p) | type(r)] AS rels,
    [r IN relationships(p) | properties(r)] AS props,
    length(p) AS hops
  ORDER BY hops ASC
  LIMIT $per_limit
}}
RETURN nodes, rels, props, hops
{total_clause}
"""


def build_entity_profile_cypher() -> str:
    """构造「实体画像」查询语句（节点标签、属性、描述）。"""
    return """
MATCH (n:Entity {name: $name})
RETURN n.name AS name, labels(n) AS labels, properties(n) AS props
LIMIT 1
"""


def simplify_path_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 Neo4j 返回的原始路径记录整理为统一结构。

    Returns:
        [{"nodes": [...], "rels": [...], "props": [...], "hops": n, "path_text": "A -[HOLDS]-> B"}]
    """
    paths: List[Dict[str, Any]] = []
    for row in records or []:
        nodes = [str(n) for n in (row.get("nodes") or [])]
        rels = [str(r) for r in (row.get("rels") or [])]
        props = list(row.get("props") or [])
        if not nodes or len(nodes) < 2:
            continue
        # 拼装可读路径：A -[HOLDS]-> B -[PENALIZED_BY]-> C
        text_parts = [nodes[0]]
        for idx, rel in enumerate(rels):
            if idx + 1 < len(nodes):
                text_parts.append(f"-[{rel}]-> {nodes[idx + 1]}")
        paths.append({
            "nodes": nodes,
            "rels": rels,
            "props": props,
            "hops": int(row.get("hops") or len(rels)),
            "path_text": " ".join(text_parts),
        })
    return paths


def simplify_relation_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """整理实体关系查询结果。"""
    relations: List[Dict[str, Any]] = []
    for row in records or []:
        head = (row.get("head") or "").strip()
        rel = (row.get("rel") or "").strip()
        tail = (row.get("tail") or "").strip()
        if not (head and rel and tail):
            continue
        relations.append({
            "head": head, "rel": rel, "tail": tail,
            "props": row.get("props") or {},
        })
    return relations


def dedupe_names(names: Optional[Sequence[Any]], max_count: Optional[int] = None) -> List[str]:
    """实体名去重（保持原顺序）+ 可选截断，供批量查询构造参数使用。"""
    out: List[str] = []
    for raw in names or []:
        name = str(raw or "").strip()
        if name and name not in out:
            out.append(name)
    return out[:max_count] if max_count else out


def build_entity_pairs(names: Optional[Sequence[Any]], max_entities: int = 3) -> List[Dict[str, str]]:
    """把实体名列表展开为两两组合（无序对），用于批量路径查询。

    取代原本 ``for i ... for j in range(i+1, ...)`` 的嵌套循环：
    组合逻辑被抽成纯函数，可离线单测，且能一次性交给 Neo4j 用一次往返查完。
    """
    uniq = dedupe_names(names, max_count=max_entities)
    return [
        {"from_name": uniq[i], "to_name": uniq[j]}
        for i in range(len(uniq))
        for j in range(i + 1, len(uniq))
    ]


def group_relation_records(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """把批量关系查询结果按种子实体分组。

    Returns:
        {"广州XX贸易有限公司": [{head, rel, tail, props}, ...], ...}
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in records or []:
        seed = str(row.get("seed") or "").strip()
        if not seed:
            continue
        rels = simplify_relation_records([row])
        if rels:
            grouped.setdefault(seed, []).extend(rels)
    return grouped


def dedupe_paths(paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按「节点序列 + 关系序列」去重（无向路径会被 A→B 与 B→A 各命中一次）。"""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for path in paths or []:
        key = (tuple(path.get("nodes") or []), tuple(path.get("rels") or []))
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


# ==================== 图数据库查询（需驱动） ====================

def _session():
    """获取 Neo4j session；不可用时抛出异常由调用方兜底。"""
    driver = get_neo4j_driver()
    if driver is None:
        raise RuntimeError("Neo4j 驱动不可用")
    return driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j"))


def find_path_between(from_name: str, to_name: str, item_name: Optional[str] = None,
                      max_hops: int = DEFAULT_MAX_HOPS, limit: int = DEFAULT_PATH_LIMIT) -> List[Dict[str, Any]]:
    """查询两个实体之间的多跳路径（如 品牌→授权店铺→商家）。

    Args:
        from_name: 起点实体名
        to_name: 终点实体名
        item_name: 主体（商家/店铺）过滤维度，可选
        max_hops: 最大跳数（1..5）
        limit: 返回路径条数上限

    Returns:
        simplify_path_records 结构；发生错误或驱动不可用时返回 []。
    """
    if not from_name or not to_name:
        return []
    cypher = build_path_cypher(max_hops, with_item_name=bool(item_name))
    params: Dict[str, Any] = {"from_name": from_name, "to_name": to_name, "limit": int(limit)}
    if item_name:
        params["item_name"] = item_name
    try:
        with _session() as session:
            records = session.execute_read(lambda tx: tx.run(cypher, **params).data())
        return simplify_path_records(records)
    except Exception as e:
        logger.warning(f"查询实体路径失败({from_name} -> {to_name})，返回空结果: {e}")
        return []


def get_entity_relations(entity_name: str, item_name: Optional[str] = None,
                         limit: int = DEFAULT_RELATION_LIMIT) -> List[Dict[str, Any]]:
    """查询某实体的一跳关系（双向，含关系属性）。"""
    if not entity_name:
        return []
    cypher = build_entity_relations_cypher(with_item_name=bool(item_name))
    params: Dict[str, Any] = {"name": entity_name, "limit": int(limit)}
    if item_name:
        params["item_name"] = item_name
    try:
        with _session() as session:
            records = session.execute_read(lambda tx: tx.run(cypher, **params).data())
        return simplify_relation_records(records)
    except Exception as e:
        logger.warning(f"查询实体关系失败({entity_name})，返回空结果: {e}")
        return []


def get_entity_relations_batch(entity_names: Optional[Sequence[Any]], item_name: Optional[str] = None,
                               limit_per_entity: int = DEFAULT_RELATION_LIMIT,
                               total_limit: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    """批量查询多个实体的一跳关系（**一次往返**，替代逐个实体开 session 的 N+1）。

    Args:
        entity_names: 种子实体名列表（自动去重保序）
        item_name: 主体（商家/店铺）过滤维度，可选
        limit_per_entity: 每个种子实体的关系条数上限
        total_limit: 可选的外层总条数上限（实体数放大时兜底）

    Returns:
        {实体名: [关系...]}；驱动不可用或查询失败时返回 {}。
    """
    names = dedupe_names(entity_names)
    if not names:
        return {}
    cypher = build_entity_relations_batch_cypher(
        with_item_name=bool(item_name), with_total_limit=bool(total_limit),
    )
    params: Dict[str, Any] = {"names": names, "per_limit": int(limit_per_entity)}
    if item_name:
        params["item_name"] = item_name
    if total_limit:
        params["total_limit"] = int(total_limit)
    try:
        with _session() as session:
            records = session.execute_read(lambda tx: tx.run(cypher, **params).data())
        return group_relation_records(records)
    except Exception as e:
        logger.warning(f"批量查询实体关系失败({names})，返回空结果: {e}")
        return {}


def find_paths_batch(pairs: Optional[Sequence[Dict[str, str]]], item_name: Optional[str] = None,
                     max_hops: int = DEFAULT_MAX_HOPS, limit_per_pair: int = DEFAULT_PATH_LIMIT,
                     total_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """批量查询多个实体对之间的多跳路径（**一次往返**）。

    调用方可用 :func:`build_entity_pairs` 构造 ``pairs``。
    结果按“路径节点序列”去重，避免同一无向路径被多个实体对重复返回。
    """
    valid: List[Dict[str, str]] = [
        {"from_name": str(p.get("from_name") or "").strip(), "to_name": str(p.get("to_name") or "").strip()}
        for p in (pairs or [])
        if str(p.get("from_name") or "").strip() and str(p.get("to_name") or "").strip()
    ]
    if not valid:
        return []
    cypher = build_paths_batch_cypher(
        max_hops=max_hops, with_item_name=bool(item_name), with_total_limit=bool(total_limit),
    )
    params: Dict[str, Any] = {"pairs": valid, "per_limit": int(limit_per_pair)}
    if item_name:
        params["item_name"] = item_name
    if total_limit:
        params["total_limit"] = int(total_limit)
    try:
        with _session() as session:
            records = session.execute_read(lambda tx: tx.run(cypher, **params).data())
    except Exception as e:
        logger.warning(f"批量查询实体路径失败({len(valid)} 对)，返回空结果: {e}")
        return []

    # 去重：无向路径可能同时命中 (A,B) 与 (B,A)
    return dedupe_paths(simplify_path_records(records))


def get_entity_profile(entity_name: str) -> Dict[str, Any]:
    """查询实体画像（标签与属性）。"""
    if not entity_name:
        return {}
    cypher = build_entity_profile_cypher()
    try:
        with _session() as session:
            records = session.execute_read(lambda tx: tx.run(cypher, name=entity_name).data())
    except Exception as e:
        logger.error(f"查询实体画像失败({entity_name}): {e}")
        return {}
    if not records:
        return {}
    row = records[0]
    return {
        "name": row.get("name", entity_name),
        "labels": row.get("labels") or [],
        "props": row.get("props") or {},
    }
