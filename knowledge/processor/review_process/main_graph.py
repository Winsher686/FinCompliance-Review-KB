"""审查流程主图（LangGraph）。

编排 8 个节点，形成电商合规审查流水线：

    intent_recognize（意图识别）
          │
          v
    query_rewrite（改写 + HyDE）
          │
          v
    multi_search（向量 / 稀疏BM25 / 图谱遍历 多路检索）
          │
          v
    rrf_rerank（RRF 融合 + BGE-Reranker 精排）
          │
          v
    evidence_chain（证据链构建）
          │
          v
    compliance_check（合规规则检查）
          │
     ┌────┴─────┐  needs_human_review
     v          v
   hitl      report_generate ──> END
     │
     v
  report_generate ──> END

HITL 说明：使用内存 checkpointer（MemorySaver）持久化图状态，
``hitl_node`` 通过 ``interrupt()`` 暂停，``POST /review/{id}/approve`` 通过
``Command(resume=...)`` 恢复同一 thread 继续执行。
"""

import json

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

from knowledge.processor.review_process.state import ReviewGraphState, create_default_state
from knowledge.processor.review_process.nodes.intent_recognize_node import IntentRecognizeNode
from knowledge.processor.review_process.nodes.query_rewrite_node import QueryRewriteNode
from knowledge.processor.review_process.nodes.multi_search_node import MultiSearchNode
from knowledge.processor.review_process.nodes.rrf_rerank_node import RrfRerankNode
from knowledge.processor.review_process.nodes.evidence_chain_node import EvidenceChainNode
from knowledge.processor.review_process.nodes.compliance_check_node import ComplianceCheckNode
from knowledge.processor.review_process.nodes.hitl_node import HitlNode
from knowledge.processor.review_process.nodes.report_generate_node import ReportGenerateNode

load_dotenv()


def route_after_compliance(state: ReviewGraphState) -> str:
    """合规检查后的路由：需要人工复核则进入 HITL，否则直接生成报告。"""
    return "hitl" if state.get("needs_human_review") else "report_generate"


def create_review_graph():
    """创建并编译审查流程图（带内存 checkpointer，支持 HITL 暂停/恢复）。"""
    workflow = StateGraph(ReviewGraphState)  # type: ignore

    nodes = {
        "intent_recognize": IntentRecognizeNode(),
        "query_rewrite": QueryRewriteNode(),
        "multi_search": MultiSearchNode(),
        "rrf_rerank": RrfRerankNode(),
        "evidence_chain": EvidenceChainNode(),
        "compliance_check": ComplianceCheckNode(),
        "hitl": HitlNode(),
        "report_generate": ReportGenerateNode(),
    }
    for name, node in nodes.items():
        workflow.add_node(name, node)  # type: ignore

    # 入口与主链路
    workflow.set_entry_point("intent_recognize")
    workflow.add_edge("intent_recognize", "query_rewrite")
    workflow.add_edge("query_rewrite", "multi_search")
    workflow.add_edge("multi_search", "rrf_rerank")
    workflow.add_edge("rrf_rerank", "evidence_chain")
    workflow.add_edge("evidence_chain", "compliance_check")

    # 合规检查后条件路由
    workflow.add_conditional_edges(
        "compliance_check",
        route_after_compliance,
        {"hitl": "hitl", "report_generate": "report_generate"},
    )
    workflow.add_edge("hitl", "report_generate")
    workflow.add_edge("report_generate", END)

    # 内存 checkpointer：支撑 interrupt 暂停与 Command(resume) 恢复
    return workflow.compile(checkpointer=MemorySaver())


# 全局图实例（服务层复用；MemorySaver 保存在进程内）
review_app = create_review_graph()


def build_thread_config(review_id: str) -> dict:
    """构造 LangGraph 线程配置（review_id 即 thread_id）。"""
    return {"configurable": {"thread_id": review_id}}


if __name__ == "__main__":
    from knowledge.processor.review_process.base import setup_logging
    setup_logging()

    demo_review_id = "demo-review-001"
    demo_state = create_default_state(
        review_id=demo_review_id,
        task_id="demo-task",
        company="广州XX贸易有限公司（XX旗舰店）",
        review_type="banned_word",
        question="该店铺商品宣传是否使用违禁词/极限词？资质是否齐全？",
        is_stream=False,
        # 离线演示：直接注入关系与经营指标（真实场景由多路检索引擎产出）
        metrics={"gmv": 5000000.0, "listing_count": 120},
        relations=[
            {"head": "广州XX贸易有限公司", "rel": "PROMOTES", "tail": "深海鱼油软胶囊宣传",
             "props": {"banned_words": "国家级,最安全", "discount_rate": "80%"}},
            {"head": "广州XX贸易有限公司", "rel": "HOLDS_CERT", "tail": "食品经营许可证",
             "props": {"cert_type": "食品经营许可证", "status": "缺失"}},
        ],
    )

    config = build_thread_config(demo_review_id)
    snapshot = review_app.invoke(demo_state, config=config)
    print("首次执行完成，是否已中断等待复核 =", bool(snapshot.get("__interrupt__")))
    print("风险等级 =", snapshot.get("risk_level"))
    print("报告预览:\n", (snapshot.get("report") or "")[:300])

    # 若因高风险进入 HITL（图被 interrupt 暂停），则恢复执行
    if snapshot.get("__interrupt__"):
        from langgraph.types import Command
        resumed = review_app.invoke(Command(resume={"approved": True, "reviewer": "demo", "comment": "同意"}), config=config)
        print("\n人工复核通过，review_status =", resumed.get("review_status"))
        print("报告预览:\n", (resumed.get("report") or "")[:300])
