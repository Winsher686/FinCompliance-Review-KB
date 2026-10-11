"""节点 7：人工复核（HITL）。

当合规检查判定为高风险（或审查类型属于需复核类型）时，本节点调用
LangGraph 的 ``interrupt()`` **暂停图执行**，把审查上下文（风险等级、命中的规则、
证据链摘要）抛给前端/业务方，等待人工审批。

审批通过后，业务方调用 ``POST /review/{review_id}/approve``，
服务端用 ``Command(resume=...)`` 恢复图执行，``interrupt()`` 的返回值即为审批结果，
本节点据此写入 approval 与 review_status，随后进入报告生成节点。

关键实现点：节点基类必须让 GraphInterrupt 穿透（见 base.is_control_flow_exception），
否则中断机制会失效。
"""

from typing import Any, Dict

from knowledge.processor.review_process.base import BaseNode
from knowledge.processor.review_process.state import ReviewGraphState


class HitlNode(BaseNode):
    name = "hitl_node"

    def process(self, state: ReviewGraphState) -> ReviewGraphState:
        # 1. 无需人工复核：直接跳过（由报告节点收尾）
        if not state.get("needs_human_review"):
            return {}

        # 2. 组织待审批载荷
        payload: Dict[str, Any] = {
            "review_id": state.get("review_id", ""),
            "company": state.get("company", ""),
            "risk_level": state.get("risk_level", ""),
            "findings": state.get("findings") or [],
            "evidence_chain": state.get("evidence_chain") or [],
            "message": "该审查结论为高风险，需人工复核确认后方可出具正式报告。",
        }

        # 3. 暂停图执行，等待人工审批（interrupt 的返回值即恢复时传入的审批结果）
        try:
            from langgraph.types import interrupt  # type: ignore
        except Exception as e:  # 兼容不同 langgraph 版本
            self.logger.error(f"当前 langgraph 版本不支持 interrupt，跳过人工复核: {e}")
            return {"approval": {"approved": True, "comment": "auto-approve(no-interrupt)"},
                    "review_status": "approved"}

        decision = interrupt(payload)

        # 4. 恢复后解析审批结果
        approval = decision if isinstance(decision, dict) else {"approved": bool(decision)}
        approved = bool(approval.get("approved", False))
        self.logger.info(f"人工复核结果: approved={approved}, reviewer={approval.get('reviewer', '')}")

        return {
            "approval": approval,
            "review_status": "approved" if approved else "rejected",
        }
