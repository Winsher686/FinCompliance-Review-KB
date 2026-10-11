"""节点 6：合规规则检查。

调用内置规则库（rules/compliance_rules.py），基于证据链中归一化的关系与经营指标
判定是否触发合规风险（违禁词/极限词宣传、资质缺失或过期、违规未整改、问题货源、
类目错放与价格违规、品牌授权链穿透），输出 findings 与整体风险等级。

同时决定是否进入人工复核（HITL）：
- 风险等级命中 HITL_RISK_LEVELS（默认「高」），或
- 审查类型命中 HITL_REVIEW_TYPES（默认含 report/violation/product_compliance）。
"""

from typing import Any, Dict, List

from knowledge.processor.review_process.base import BaseNode
from knowledge.processor.review_process.state import ReviewGraphState
from knowledge.processor.review_process.rules.compliance_rules import (
    build_context, evaluate_rules, overall_risk_level,
)


class ComplianceCheckNode(BaseNode):
    name = "compliance_check_node"

    def process(self, state: ReviewGraphState) -> ReviewGraphState:
        # 1. 装配审查上下文（规则引擎的唯一输入）
        ctx = build_context(
            company=state.get("company", ""),
            relations=state.get("relations") or [],
            graph_paths=state.get("kg_paths") or [],
            documents=state.get("reranked_docs") or [],
            metrics=state.get("metrics") or {},
            disclosed=set(state.get("disclosed") or []),
        )

        # 2. 执行规则
        findings = evaluate_rules(ctx)
        risk_level = overall_risk_level(findings)

        # 3. 是否触发人工复核
        needs_human_review = self._needs_human_review(state, risk_level)

        self.logger.info(
            f"合规检查完成：命中 {len(findings)} 条规则，风险等级={risk_level}，"
            f"需要人工复核={needs_human_review}"
        )
        return {
            "findings": [f.to_dict() for f in findings],
            "risk_level": risk_level,
            "needs_human_review": needs_human_review,
        }

    def _needs_human_review(self, state: ReviewGraphState, risk_level: str) -> bool:
        """判断是否需要人工复核（HITL）。"""
        if not self.config.hitl_enabled:
            return False
        review_type = (state.get("review_type") or "general").strip()
        if risk_level in self.config.hitl_trigger_risk_levels():
            return True
        if review_type in self.config.hitl_trigger_review_types():
            return True
        return False
