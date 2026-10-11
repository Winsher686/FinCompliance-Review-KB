"""电商商品/商家合规审查流程（review_process）。

本单元与 import_process / query_process 同级，是「自包含」的审查服务：
- 输入：商家（主体）+ 审查类型 + 问题描述；
- 处理：LangGraph 状态图（意图识别 → 改写/HyDE → 多路检索 → RRF+重排 →
        证据链 → 合规规则检查 → 人工复核(HITL) → 报告生成）；
- 输出：结构化合规审查报告（结论 / 证据链 / 风险等级 / 建议）+ SSE 进度。

技术栈保持不变：FastAPI / LangGraph / Milvus / Neo4j / BGE-M3 / RRF / BGE-Reranker /
MongoDB / MinIO / SSE。
"""

from knowledge.processor.review_process.state import ReviewGraphState, create_default_state, get_default_state  # noqa: F401

__all__ = ["ReviewGraphState", "create_default_state", "get_default_state"]
