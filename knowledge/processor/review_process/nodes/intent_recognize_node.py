"""节点 1：意图识别。

把用户问题归类到 4 类意图，决定后续走哪条检索/审查策略：
- fact        事实查询（如“该店铺近三年是否受过平台处罚？处罚原因是什么？”）
- path        路径/关系查询（如“该店铺与供应商之间是否存在货源关系？”）
- compliance  合规检查（如“该商品宣传用语是否违反广告法？”）
- report      报告生成（如“生成一份合规审查报告”）

优先调用 LLM（json_mode）；LLM 不可用时回退到关键词规则，保证离线可跑。
"""

import json
import re

from knowledge.processor.review_process.base import BaseNode
from knowledge.processor.review_process.state import ReviewGraphState
from knowledge.processor.review_process.prompts.review_prompts import INTENT_SYSTEM_PROMPT

VALID_INTENTS = {"fact", "path", "compliance", "report"}


class IntentRecognizeNode(BaseNode):
    name = "intent_recognize_node"

    def process(self, state: ReviewGraphState) -> ReviewGraphState:
        question = (state.get("question") or "").strip()
        review_type = (state.get("review_type") or "general").strip()

        # 1. 优先 LLM 识别，失败回退规则
        intent = self._llm_intent(question) or self._rule_intent(question, review_type)
        self.logger.info(f"意图识别结果: {intent}")
        return {"intent": intent}

    def _llm_intent(self, question: str) -> str:
        """调用 LLM 识别意图；任何异常都返回空串以触发规则回退。"""
        if not question:
            return ""
        try:
            from knowledge.utils.llm_client import get_llm_client
            from langchain_core.messages import SystemMessage, HumanMessage

            client = get_llm_client(json_mode=True)
            if client is None:
                return ""
            data = client.invoke_json([
                SystemMessage(content=INTENT_SYSTEM_PROMPT),
                HumanMessage(content=question),
            ])
            if not isinstance(data, dict):
                return ""
            intent = str(data.get("intent", "")).strip().lower()
            return intent if intent in VALID_INTENTS else ""
        except Exception as e:
            self.logger.debug(f"LLM 意图识别不可用，回退规则: {e}")
            return ""

    @staticmethod
    def _rule_intent(question: str, review_type: str = "") -> str:
        """关键词规则兜底（离线可用）。"""
        q = question or ""
        if review_type == "report" or "报告" in q:
            return "report"
        if any(k in q for k in ("路径", "之间的关系", "什么关系", "授权链", "货源", "穿透",
                                "层级", "链条", "间接", "供应商", "关联店铺")):
            return "path"
        if any(k in q for k in ("是否违反", "是否合规", "合规", "违禁", "极限词", "违规",
                                "资质", "类目", "价格", "披露", "规定", "审查", "风险", "整改")):
            return "compliance"
        return "fact"
