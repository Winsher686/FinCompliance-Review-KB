"""节点 8：报告生成。

输出结构化合规审查报告（结论 / 证据链 / 风险等级 / 建议）。
优先调用 LLM 生成，失败时回退到模板拼装，保证离线可跑。
"""

from typing import Any, Dict, List

from knowledge.processor.review_process.base import BaseNode
from knowledge.processor.review_process.state import ReviewGraphState
from knowledge.processor.review_process.prompts.review_prompts import (
    REPORT_SYSTEM_PROMPT, REPORT_USER_TEMPLATE, REVIEW_TYPE_CN,
)


class ReportGenerateNode(BaseNode):
    name = "report_generate_node"

    def process(self, state: ReviewGraphState) -> ReviewGraphState:
        report = self._llm_report(state) or self._template_report(state)
        conclusion = self._template_conclusion(state)
        status = self._final_status(state)

        self.logger.info(f"审查报告生成完成，最终状态: {status}")
        return {"report": report, "conclusion": conclusion, "review_status": status}

    # ==================== 状态 ====================
    @staticmethod
    def _final_status(state: ReviewGraphState) -> str:
        """根据审批结果确定最终状态。"""
        approval = state.get("approval") or {}
        if state.get("needs_human_review"):
            if approval.get("approved") is True:
                return "approved"
            if approval.get("approved") is False:
                return "rejected"
            return "pending_approval"
        return "completed"

    # ==================== LLM 生成 ====================
    def _llm_report(self, state: ReviewGraphState) -> str:
        try:
            from knowledge.utils.llm_client import get_llm_client
            from langchain_core.messages import SystemMessage, HumanMessage

            client = get_llm_client()
            if client is None:
                return ""
            resp = client.invoke([
                SystemMessage(content=REPORT_SYSTEM_PROMPT),
                HumanMessage(content=self._report_user_prompt(state)),
            ])
            return (getattr(resp, "content", "") or "").strip()
        except Exception as e:
            self.logger.debug(f"LLM 报告生成不可用，回退模板: {e}")
            return ""

    def _report_user_prompt(self, state: ReviewGraphState) -> str:
        return REPORT_USER_TEMPLATE.format(
            company=state.get("company", ""),
            review_type=REVIEW_TYPE_CN.get(state.get("review_type", "general"), state.get("review_type", "")),
            question=state.get("question", ""),
            risk_level=state.get("risk_level", ""),
            findings=self._format_findings(state.get("findings") or []),
            evidence=self._format_evidence(state.get("evidence_chain") or []),
            context=self._format_context(state.get("reranked_docs") or [], self.config.max_context_chars),
        )

    # ==================== 模板回退 ====================
    def _template_report(self, state: ReviewGraphState) -> str:
        company = state.get("company", "") or "（未指定主体）"
        review_type = REVIEW_TYPE_CN.get(state.get("review_type", "general"), "综合合规审查")
        risk = state.get("risk_level", "无")

        lines: List[str] = [
            "# 合规审查报告",
            "",
            "## 一、审查对象与范围",
            f"- 审查对象：{company}",
            f"- 审查类型：{review_type}",
            f"- 审查问题：{state.get('question', '') or '（未提供）'}",
            "",
            "## 二、审查结论",
            self._template_conclusion(state),
            "",
            "## 三、证据链",
        ]
        evidence = state.get("evidence_chain") or []
        if evidence:
            for i, item in enumerate(evidence[:20], 1):
                src = item.get("source", "")
                lines.append(f"{i}. [{item.get('type', '')}] {item.get('content', '')}"
                             + (f"（来源：{src}）" if src else ""))
        else:
            lines.append("（未检索到有效证据，现有资料不足以支撑判断。）")

        lines += ["", "## 四、合规风险与规则命中"]
        findings = state.get("findings") or []
        if findings:
            for f in findings:
                lines.append(f"- **[{f.get('severity')}] {f.get('rule_id')} {f.get('rule_name')}**：{f.get('message')}")
                for ev in (f.get("evidence") or [])[:3]:
                    lines.append(f"  - 证据：{ev}")
                if f.get("regulation"):
                    lines.append(f"  - 依据：{f['regulation']}")
        else:
            lines.append("（未命中内置合规规则。）")

        lines += [
            "",
            "## 五、风险等级",
            f"**{risk}**",
            "",
            "## 六、处理建议",
        ]
        if findings:
            for f in findings:
                if f.get("suggestion"):
                    lines.append(f"- {f.get('suggestion')}")
        else:
            lines.append("- 暂无需要特别提示的合规事项；建议持续关注后续公告与监管动态。")

        approval = state.get("approval") or {}
        if state.get("needs_human_review"):
            lines += ["", "## 七、人工复核", 
                      f"- 复核结论：{'通过' if approval.get('approved') else ('驳回' if approval.get('approved') is False else '待复核')}",
                      f"- 复核人：{approval.get('reviewer', '') or '（未填写）'}",
                      f"- 复核意见：{approval.get('comment', '') or '（无）'}"]

        return "\n".join(lines)

    def _template_conclusion(self, state: ReviewGraphState) -> str:
        risk = state.get("risk_level", "无")
        company = state.get("company", "") or "该主体"
        findings = state.get("findings") or []
        if findings:
            top = max(findings, key=lambda f: {"高": 3, "中": 2, "低": 1, "无": 0}.get(f.get("severity", "无"), 0))
            return (f"经审查，{company}存在 {len(findings)} 项合规风险，整体风险等级为「{risk}」；"
                    f"其中以「{top.get('rule_name')}」最为突出：{top.get('message')}")
        return f"经审查，未发现 {company} 命中内置合规规则，整体风险等级为「{risk}」。证据不足部分建议补充资料后复核。"

    # ==================== 格式化 ====================
    @staticmethod
    def _format_findings(findings: List[Dict[str, Any]]) -> str:
        if not findings:
            return "（无）"
        return "\n".join(f"- [{f.get('severity')}] {f.get('rule_id')} {f.get('rule_name')}：{f.get('message')}"
                         for f in findings)

    @staticmethod
    def _format_evidence(evidence: List[Dict[str, Any]]) -> str:
        if not evidence:
            return "（无）"
        return "\n".join(f"- [{e.get('type')}] {e.get('content')}" for e in evidence[:20])

    @staticmethod
    def _format_context(docs: List[Dict[str, Any]], budget: int) -> str:
        out, used = [], 0
        for doc in docs:
            content = (doc.get("content") or "").strip()
            if not content:
                continue
            entry = f"[来源={doc.get('source') or doc.get('title') or ''}] {content}"
            if used + len(entry) > budget:
                break
            out.append(entry)
            used += len(entry) + 1
        return "\n\n".join(out) or "（无）"
