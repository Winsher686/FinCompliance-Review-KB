"""审查业务服务。

职责：
- 提交并异步执行审查流程（LangGraph）；
- 读取审查结果（含证据链、风险等级、报告）；
- 处理人工审批（HITL）：用 Command(resume=...) 恢复被 interrupt 暂停的图。

数据持久化：审查状态由 LangGraph 的内存 checkpointer（MemorySaver）按
thread_id（即 review_id）保存，服务层通过 review_app.get_state 读取。
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from knowledge.processor.review_process.main_graph import review_app, build_thread_config
from knowledge.processor.review_process.state import create_default_state
from knowledge.utils.task_util import (
    update_task_status, set_task_result, get_task_result, get_task_status,
    get_done_task_list, get_running_task_list,
    TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED,
)
from knowledge.utils.sse_util import create_sse_queue, push_sse_event, SSEEvent
from knowledge.utils.observability import ensure_trace_id, get_trace, format_waterfall

logger = logging.getLogger(__name__)


class ReviewService:

    # ==================== ID 生成 ====================
    def generate_review_id(self) -> str:
        return str(uuid.uuid4())

    def generate_task_id(self) -> str:
        return str(uuid.uuid4())

    def generate_session_id(self) -> str:
        return str(uuid.uuid4())

    # ==================== 提交 ====================
    def prepare(self, task_id: str, is_stream: bool) -> None:
        """提交前准备：更新任务状态；流式模式创建 SSE 队列。"""
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        if is_stream:
            create_sse_queue(task_id)

    # ==================== 执行 ====================
    def run_review_graph(self, review_id: str, task_id: str, company: str, review_type: str,
                         question: str, is_stream: bool = False,
                         session_id: str = "",
                         metrics: Optional[Dict[str, float]] = None,
                         relations: Optional[List[Dict[str, Any]]] = None) -> None:
        """执行审查流程（同步阻塞，调用方需放入线程池）。

        说明：本方法在 HITL 触发时不会抛异常——interrupt 会让图暂停，
        invoke 返回的 state 中带 ``__interrupt__`` 标记，据此把任务标记为 pending_approval。
        """
        config = build_thread_config(review_id)
        initial_state = create_default_state(
            review_id=review_id,
            task_id=task_id,
            session_id=session_id or "",
            company=company or "",
            review_type=review_type or "general",
            question=question or "",
            is_stream=is_stream,
            metrics=metrics or {},
            relations=relations or [],
        )
        # 可观测性：trace_id 在入口确定（优先派生自 review_id），随后由 LangGraph 带给所有节点
        trace_id = ensure_trace_id(initial_state)
        initial_state["trace_id"] = trace_id

        try:
            result = review_app.invoke(initial_state, config=config)
            interrupted = bool(result.get("__interrupt__"))
            status = "pending_approval" if interrupted else (result.get("review_status") or "completed")

            set_task_result(task_id, "review_id", review_id)
            set_task_result(task_id, "risk_level", result.get("risk_level", ""))
            set_task_result(task_id, "status", status)
            set_task_result(task_id, "report", result.get("report", ""))
            set_task_result(task_id, "trace_id", trace_id)
            update_task_status(task_id, TASK_STATUS_COMPLETED)

            # 输出节点耗时瀑布（线上排障时可直接 grep trace=xxx 定位到慢节点）
            logger.info("审查完成 trace=%s status=%s\n%s",
                        trace_id, status, format_waterfall(trace_id))

            if is_stream:
                push_sse_event(task_id, SSEEvent.FINAL, {
                    "review_id": review_id,
                    "trace_id": trace_id,
                    "status": status,
                    "risk_level": result.get("risk_level", ""),
                    "conclusion": result.get("conclusion", ""),
                    "report": result.get("report", ""),
                    "needs_human_review": bool(result.get("needs_human_review")),
                    "interrupted": interrupted,
                })
        except Exception as e:
            logger.error(f"审查流程执行失败: {e}", exc_info=True)
            set_task_result(task_id, "error", str(e))
            set_task_result(task_id, "status", "failed")
            update_task_status(task_id, TASK_STATUS_FAILED)
            if is_stream:
                push_sse_event(task_id, SSEEvent.FINAL, {"review_id": review_id, "status": "failed", "error": str(e)})

    # ==================== 读取结果 ====================
    def get_review(self, review_id: str) -> Dict[str, Any]:
        """读取审查结果（含中断状态）。"""
        config = build_thread_config(review_id)
        try:
            snapshot = review_app.get_state(config)
        except Exception as e:
            logger.warning(f"读取审查状态失败({review_id}): {e}")
            snapshot = None

        values = getattr(snapshot, "values", None) or {}
        pending = bool(getattr(snapshot, "next", None))

        if not values:
            return {"review_id": review_id, "status": "not_found"}

        status = values.get("review_status") or "processing"
        if pending:
            status = "pending_approval"

        return {
            "review_id": review_id,
            "trace_id": values.get("trace_id", ""),
            "company": values.get("company", ""),
            "review_type": values.get("review_type", ""),
            "question": values.get("question", ""),
            "intent": values.get("intent", ""),
            "status": status,
            "conclusion": values.get("conclusion", ""),
            "risk_level": values.get("risk_level", ""),
            "findings": values.get("findings") or [],
            "evidence_chain": values.get("evidence_chain") or [],
            "graph_paths": values.get("kg_paths") or [],
            "report": values.get("report", ""),
            "needs_human_review": bool(values.get("needs_human_review")),
            "approval": values.get("approval") or {},
        }

    # ==================== 可观测性 ====================
    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """返回一条链路追踪的完整数据（节点耗时瀑布 + 指标快照）。

        用于线上排障：拿到 trace_id 即可看清每个节点各耗时多少，
        从而判断慢在检索、图谱还是 LLM。
        """
        trace = get_trace(trace_id)
        spans = trace.get("spans") or []
        return {
            "trace_id": trace_id,
            "total_ms": trace.get("total_ms", 0.0),
            "node_count": len(spans),
            "spans": spans,
            "waterfall": format_waterfall(trace_id),
        }

    # ==================== 人工审批 ====================
    def approve(self, review_id: str, approved: bool, reviewer: str = "", comment: str = "") -> Dict[str, Any]:
        """人工审批：恢复被 interrupt 暂停的图执行，并返回最新结果。"""
        config = build_thread_config(review_id)
        try:
            from langgraph.types import Command  # 延迟导入，兼容不同版本
            review_app.invoke(
                Command(resume={"approved": bool(approved), "reviewer": reviewer, "comment": comment}),
                config=config,
            )
        except Exception as e:
            logger.error(f"人工审批恢复执行失败({review_id}): {e}", exc_info=True)
            return {"review_id": review_id, "status": "failed", "error": str(e)}

        result = self.get_review(review_id)
        trace_id = result.get("trace_id") or ""
        if trace_id:
            # 审批恢复后补打瀑布，便于对比「审批前 / 审批后」各节点耗时
            logger.info("人工审批完成 trace=%s\n%s", trace_id, format_waterfall(trace_id))
        return result

    # ==================== 图谱查询 ====================
    def get_entity(self, entity_id: str, name: str = "", limit: int = 50) -> Dict[str, Any]:
        """查询实体画像与其一跳关系。

        Args:
            entity_id: 实体 ID（若上游图谱使用 name 作为标识，则与 name 等价）
            name: 实体名；不传时回退用 entity_id 作为实体名（本项目图谱以 name 为唯一键）
            limit: 关系条数上限
        """
        target = name or entity_id
        try:
            from knowledge.utils.neo4j_util import get_entity_profile, get_entity_relations
            profile = get_entity_profile(target)
            relations = get_entity_relations(target, limit=limit)
        except Exception as e:  # 离线/无驱动时兜底为空结果
            logger.warning(f"查询实体失败({target}): {e}")
            profile, relations = {}, []

        return {
            "entity_id": entity_id,
            "name": (profile or {}).get("name", target),
            "labels": (profile or {}).get("labels", []),
            "props": (profile or {}).get("props", {}),
            "relations": relations,
        }

    def find_path(self, from_name: str, to_name: str, item_name: str = "",
                  max_hops: int = 3, limit: int = 10) -> Dict[str, Any]:
        """查询两个实体之间的多跳路径（如 品牌→授权店铺→商家）。

        跳数会被 clamp_hops 钳制到 [1,5]，避免过深遍历导致性能问题。
        """
        try:
            from knowledge.utils.neo4j_util import find_path_between, clamp_hops
            hops = clamp_hops(max_hops)
            paths = find_path_between(
                from_name, to_name,
                item_name=item_name or None,
                max_hops=hops, limit=limit,
            )
        except Exception as e:  # 离线/无驱动时兜底为空结果
            logger.warning(f"查询实体路径失败({from_name} -> {to_name}): {e}")
            hops, paths = max_hops, []

        return {
            "from_name": from_name,
            "to_name": to_name,
            "max_hops": hops,
            "paths": paths,
        }

    # ==================== 任务状态 ====================
    def get_status(self, task_id: str) -> Dict[str, Any]:
        return {
            "task_id": task_id,
            "status": get_task_status(task_id),
            "review_id": get_task_result(task_id, "review_id", ""),
            "risk_level": get_task_result(task_id, "risk_level", ""),
            "done_list": get_done_task_list(task_id),
            "running_list": get_running_task_list(task_id),
        }
