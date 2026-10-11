"""审查流程状态类型定义（电商商品/商家合规）。

定义完整的审查状态结构与辅助函数，覆盖 8 个节点所需的全部字段。
"""

from typing import TypedDict, List, Dict, Any
import copy


class ReviewGraphState(TypedDict, total=False):
    """电商合规审查图状态。

    字段分组：
    - 任务标识：review_id / task_id / session_id
    - 输入：company / review_type / question
    - 中间：intent / rewritten_query / hyde_doc / 各路检索结果 / rrf / rerank
    - 证据与合规：evidence_chain / relations / findings / risk_level
    - HITL：needs_human_review / review_status / approval
    - 输出：report / conclusion / metrics / error
    """

    # ==================== 任务标识 ====================
    review_id: str          # 审查 ID（同时作为 LangGraph thread_id）
    task_id: str            # 任务 ID（SSE / 状态轮询）
    session_id: str         # 会话 ID
    trace_id: str           # 链路追踪 ID（可观测性：节点耗时瀑布 / 指标聚合）

    # ==================== 输入 ====================
    company: str            # 主体（商家/店铺）名，等价于知识库 item_name 维度
    review_type: str        # 审查类型：banned_word / qualification / violation / supply_chain / price_category / general / report
    question: str           # 用户问题描述
    is_stream: bool         # 是否流式（SSE）

    # ==================== 意图与改写 ====================
    intent: str             # fact / path / compliance / report
    rewritten_query: str    # 改写后的问题
    hyde_doc: str           # 假设性文档

    # ==================== 多路检索 ====================
    embedding_chunks: list  # 向量检索结果
    sparse_chunks: list     # 稀疏/BM25 检索结果
    kg_chunks: list         # 图谱命中切片
    kg_triples: list        # 图谱一跳关系文本
    kg_paths: list          # 图谱多跳路径（结构化）
    search_branch_ms: dict  # 各检索分支耗时（向量/稀疏/图谱），用于定位「哪一路拖慢了检索」
    search_wall_ms: float   # 三路并发检索的墙钟耗时（≈ 最慢的一路）

    # ==================== 融合与重排 ====================
    rrf_chunks: list
    reranked_docs: list

    # ==================== 证据与合规 ====================
    evidence_chain: list    # 结构化证据链
    relations: list         # 归一化关系 [{head, rel, tail, props}]
    metrics: dict           # 经营指标（如 gmv / listing_count），用于规则计算
    findings: list          # 合规规则命中结果
    risk_level: str         # 高 / 中 / 低 / 无

    # ==================== HITL ====================
    needs_human_review: bool
    review_status: str      # processing / pending_approval / approved / rejected / completed / failed
    approval: dict          # 审批信息 {approved, reviewer, comment}

    # ==================== 输出 ====================
    report: str             # 结构化审查报告（Markdown）
    conclusion: str         # 结论
    error: str              # 错误信息


DEFAULT_STATE: ReviewGraphState = {
    "review_id": "",
    "task_id": "",
    "session_id": "",
    "trace_id": "",
    "company": "",
    "review_type": "general",
    "question": "",
    "is_stream": False,
    "intent": "",
    "rewritten_query": "",
    "hyde_doc": "",
    "embedding_chunks": [],
    "sparse_chunks": [],
    "kg_chunks": [],
    "kg_triples": [],
    "kg_paths": [],
    "search_branch_ms": {},
    "search_wall_ms": 0.0,
    "rrf_chunks": [],
    "reranked_docs": [],
    "evidence_chain": [],
    "relations": [],
    "metrics": {},
    "findings": [],
    "risk_level": "",
    "needs_human_review": False,
    "review_status": "processing",
    "approval": {},
    "report": "",
    "conclusion": "",
    "error": "",
}


def create_default_state(**overrides) -> ReviewGraphState:
    """创建默认状态，支持字段覆盖。"""
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> ReviewGraphState:
    """获取默认状态副本（避免污染全局）。"""
    return copy.deepcopy(DEFAULT_STATE)


graph_default_state = DEFAULT_STATE
