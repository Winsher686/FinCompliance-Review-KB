"""节点 2：查询改写 + HyDE 假设性文档生成。

- rewritten_query：把口语化问题改写成面向电商合规知识库的检索查询（补全主体、覆盖合规维度）；
- hyde_doc：先让 LLM 生成一段「假设性合规文档内容」，再用于向量检索，弥补问题与文档的表述差异。

两者均优先 LLM、失败回退模板，保证离线可跑。
"""

from knowledge.processor.review_process.base import BaseNode
from knowledge.processor.review_process.state import ReviewGraphState
from knowledge.processor.review_process.prompts.review_prompts import (
    REWRITE_SYSTEM_PROMPT, HYDE_SYSTEM_PROMPT, HYDE_USER_TEMPLATE,
)
from knowledge.schema.commerce_domain import compliance_dimensions_text

# 合规检索需要覆盖的维度：由领域层单一事实来源生成，禁止在此另写硬编码词表
_COMPLIANCE_DIMENSIONS = compliance_dimensions_text()


class QueryRewriteNode(BaseNode):
    name = "query_rewrite_node"

    def process(self, state: ReviewGraphState) -> ReviewGraphState:
        company = (state.get("company") or "").strip()
        question = (state.get("question") or "").strip()

        rewritten = self._llm_rewrite(company, question) or self._rule_rewrite(company, question)
        hyde_doc = self._llm_hyde(company, question) or self._rule_hyde(company, question)

        self.logger.info(f"改写查询: {rewritten}")
        return {"rewritten_query": rewritten, "hyde_doc": hyde_doc}

    # ---------- 改写 ----------
    def _llm_rewrite(self, company: str, question: str) -> str:
        try:
            from knowledge.utils.llm_client import get_llm_client
            from langchain_core.messages import SystemMessage, HumanMessage

            client = get_llm_client()
            if client is None:
                return ""
            resp = client.invoke([
                SystemMessage(content=REWRITE_SYSTEM_PROMPT),
                HumanMessage(content=f"主体：{company}\n问题：{question}"),
            ])
            return (getattr(resp, "content", "") or "").strip()
        except Exception as e:
            self.logger.debug(f"LLM 改写不可用，回退模板: {e}")
            return ""

    @staticmethod
    def _rule_rewrite(company: str, question: str) -> str:
        return f"{company} {question}（合规维度：{_COMPLIANCE_DIMENSIONS}）".strip()

    # ---------- HyDE ----------
    def _llm_hyde(self, company: str, question: str) -> str:
        try:
            from knowledge.utils.llm_client import get_llm_client
            from langchain_core.messages import SystemMessage, HumanMessage

            client = get_llm_client()
            if client is None:
                return ""
            resp = client.invoke([
                SystemMessage(content=HYDE_SYSTEM_PROMPT),
                HumanMessage(content=HYDE_USER_TEMPLATE.format(company=company, question=question)),
            ])
            return (getattr(resp, "content", "") or "").strip()
        except Exception as e:
            self.logger.debug(f"LLM HyDE 不可用，回退模板: {e}")
            return ""

    @staticmethod
    def _rule_hyde(company: str, question: str) -> str:
        """模板化假设性文档（离线兜底）。"""
        return (
            f"根据平台治理记录，{company}相关情况如下：店铺在售商品类目清晰，"
            f"持有相关经营资质并在有效期内；商品宣传用语符合广告规范，促销价格已明码标价；"
            f"店铺近三年是否存在违规处罚、消费者投诉及问题货源情况，需结合平台处罚单与投诉工单进一步核实。"
            f"针对问题“{question}”，上述治理记录可作为合规判断的依据。"
        )
