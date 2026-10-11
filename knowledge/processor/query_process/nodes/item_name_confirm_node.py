import json
import re
import logging
from json import JSONDecodeError

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.base import BaseNode
from typing import Any,Tuple,Dict,List
from langchain_core.messages import HumanMessage,SystemMessage
from knowledge.utils.bge_m3_embedding_util import generate_hybrid_embeddings, get_beg_m3_embedding_model
from knowledge.utils.milvus_util import get_milvus_client, create_hybrid_search_requests,execute_hybrid_search_query
from knowledge.utils.llm_client import parse_json_response

"""
知识库问答链路的「主体名确认」前置节点。

职责：用户的口语化提问（如"我们那家旗舰店的资质全不全"）通常无法直接用于检索，
需要先把问题对齐到知识库中的标准主体名（商家 / 店铺），再交给下游多路检索。

组成：
- ItemNameExtractor：调 LLM（json_mode）按 ITEM_NAME_EXTRACT_TEMPLATE 提取 item_names
  与 rewritten_query；_clean_parse 负责剥离 ```json 围栏、反序列化并清洗为非空字符串列表，
  异常只记日志不抛出（历史对话目前仅做指代消解输入，抽取环节保留占位）。
- ItemNameAligner：_match_vector 用 BGE-M3 生成 dense+sparse 混合向量，在 Milvus 的
  item_name_collection 上做混合检索（WeightedRanker(0.5, 0.5) + norm_score=True）；
  _item_name_score_align 按分数分档（≥0.7 进 confirmed、≥0.6 进 options，优先采信与 LLM
  抽出名字完全相同的精确匹配，高分段多条时取前 3 条进 options）；_item_name_score_filter
  仅在 confirmed 多于一个时触发，以最高分为基准剔除分差超过 0.15 的误判项。
- ItemNameConfirmNode.process：编排 original_query → 抽名 → 有名则对齐过滤、无名则双空
  → _decide 写 state。三路分支：confirmed 非空 → 写入 rewritten_query 与 item_names，
  继续走下游多路检索；仅有 options → 写入 answer 反问并向用户澄清，在此打断；
  两者皆空 → 写入兜底话术终止链路。

已知取舍：confirmed 与 options 同时有值时走第一支，options 会被静默丢弃，
即多主体场景下被误判进 options 的项不会再获得澄清机会。
"""

class ItemNameExtractor:
    """
    从用户原始问题（结合历史对话）中提取其真正想询问的主体名。

    示例：
      - 单主体："广州XX贸易有限公司今年的资质是否齐全？" → item_names=["广州XX贸易有限公司"]
      - 多主体："广州XX贸易有限公司与YY供应链之间是什么货源关系？" → item_names=["广州XX贸易有限公司", "YY供应链有限公司"]
      - 噪声情况：抽出的名称与库中标准名不完全一致，交由下游对齐分档过滤。
    """
    def extract_item_name(self, original_query: str, chat_history: List[Dict[str, Any]] = None):
        from knowledge.utils.llm_client import get_llm_client
        from knowledge.processor.query_process.prompts.item_name_extract_prompt import ITEM_NAME_EXTRACT_TEMPLATE
        """
        LLM提取用户原始问题的主体名1

        :param original_query: 用户当前轮的问题
        :param chat_history: 本会话的历史消息（MongoDB 记录），用于代词指代消解
        """
        result: Dict[str, Any] = {"item_names": [], "rewritten_query": original_query}

        # 0. 构造历史会话文本（此前这里是硬编码空串，多轮对话无法做指代消解）
        history = self._format_history_text(chat_history)

        #1. 调用llm客户端
        llm_client = get_llm_client(json_mode=True)
        if llm_client is None:
            return result
        # 2. 定义提示词(用户级别的)
        human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(history_text=history if history else "暂无上下文",
                                                         query=original_query)
        system_prompt = "你是一个专业的电商合规助手，擅长理解用户意图并提取关键信息。"

        #3. LLM调用
        llm_response = llm_client.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
        )

        llm_content = llm_response.content.strip()

        # 4. 判断LLM的输出
        if not llm_content:
            return result
        try:
            # 5. 清洗和解析
            parsed_result = self._clean_parse(llm_content)
            result["rewritten_query"] = parsed_result.get("rewritten_query") or original_query
            result["item_names"] = parsed_result.get("item_names")
        except Exception as e:
            logger.error(f"清洗以及解析LLM的输出失败: {str(e)}")

        return result

    @staticmethod
    def _format_history_text(chat_history: List[Dict[str, Any]] = None) -> str:
        """把 MongoDB 历史记录拼成提示词里的【历史会话】文本。

        原实现把 history 硬编码成空串（历史对话提取占位未启用），
        这里补上真正的拼接：role 直接用于指代消解（"它/这个"指哪款产品）。
        """
        if not chat_history:
            return ""
        lines = []
        for msg in chat_history:
            role = msg.get("role", "")
            text = (msg.get("text") or "").strip()
            if text:
                lines.append(f"{role}: {text}")
        return "\n".join(lines)

    def _clean_parse(self, llm_response: str) -> Dict[str, Any]:
        # 1. 反序列化（统一走 llm_client.parse_json_response：去围栏 + 括号配平兜底）
        parsed_llm_result = parse_json_response(llm_response)
        if not isinstance(parsed_llm_result, dict):
            # 解析失败：返回空结果，由上层写入兜底话术，而不是抛异常中断链路
            logger.warning("LLM 主体名提取输出无法解析为 JSON，返回空结果")
            return {"item_names": [], "rewritten_query": ""}

        # 2. 清洗item_names
        raw_item_names = parsed_llm_result.get('item_names')
        if not isinstance(raw_item_names, list):
            clean_item_names = []
        else:
            clean_item_names = [raw_item.strip() for raw_item in raw_item_names
                                if isinstance(raw_item, str) and raw_item.strip()]

        # 3. 清洗rewritten_query
        raw_rewritten_query = parsed_llm_result.get('rewritten_query')
        clean_rewritten_query = "" if not isinstance(raw_rewritten_query, str) else raw_rewritten_query.strip()

        return {"item_names": clean_item_names, "rewritten_query": clean_rewritten_query}

class ItemNameAligner():  # 1 usage
    """
    主要职责：
    1. 查询向量数据库
    2. 评分对齐
    3. 分数差异过滤
    """

    def match_align_filter(self, item_names: List[str]) -> Tuple[List[str], List[str]]:  # 1 usage
        # 1. 查询向量数据库
        search_result: List[Dict[str, Any]] = self._match_vector(item_names)
    
        # 2. 评分对齐
        confirmed, options = self._item_name_score_align(search_result)
    
        # 3. 分数差异过滤
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)
        
        return confirmed,options

    def _item_name_score_align(self, search_results: List[Dict[str, Any]]) -> Tuple[
        List[str], List[str]]:  # 1 usage  new *
        """
        主要职责：根据向量数据库检索到的主体名，放到对应的confirmed或者options

        Args:
            search_result:

        Returns:
            分数阈值的规则：confirm: 0.75   options:0.6
            分数阈值作为放到confirmed或者options的条件。

            返回值：confirmed有，将confirmed中的主体名 传给下游四路检索
            返回值：options有，确认下一步，询问到底在咨询哪一款主体。
            返回值：confirmed没有 options没有，直接告诉没有找到具体的主体名
            返回值：confirmed有 options有，至少确定了一个主体名，没有必要让用户在次确认这个主体。

        注意：
        1. 如果像confirmed列表中添加某一次遍历向量数据库查询到的主体名时，发现confirmed已经有该主体名了。
        2. 如果像confirmed列表中添加某一次遍历向量数据库查询到的主体名时，发现confirmed已经有该主体名了。
        3. 如果confirmed中已经有某一个主体从向量数据库返回的某个对应的item_name，那么下一次从另外一个主体名中根据向量数据库中返回的同一个item_name 既不能加到confirmed（重复） 也不能加入options中
        4. 如果options中已经有某一个主体从向量数据库返回的某个对应的item_name，那么下一次从另外一个主体名中根据向量数据库中返回的同一个item_name 不能加到options中（重复） 但是可以加入confirm中
        """

        # 1. 定义两个容器
        confirmed = []
        options = []

        # 2. 遍历向量数据库查询到的所有LLM提取到主体名相关的相似性结果
        for item_name_search_result in search_results:

            # 2.0 获取LLM提取的主体名
            extracted_name = item_name_search_result.get('extracted_name')

            # 2.1 对某一给主体名下找到相似的item_name的分数值进行降序
            matches = sorted(item_name_search_result.get('matches'), key=lambda x: x['score'], reverse=True)

            # 2.2 获取matches中分数值比能进入到confirmed容器阈值大的对象获取到
            high = [m for m in matches if m.get('score') >= 0.7]

            # 3. 找到最匹配的
            if high:

                # 3.1 准备找最精准的那一个
                extract = next((h for h in high if str(h['item_name']) == extracted_name), None)

                # 场景A:找到了(最准确)
                if extract:
                    picked = extract["item_name"]
                    # 重复的item_name confirmed中只留一份
                    if picked not in confirmed:
                        confirmed.append(picked)
                # 场景B:一般准确
                elif len(high) == 1:
                    picked = high[0]["item_name"]
                    # 重复的item_name confirmed中只留一份
                    if picked not in confirmed:
                        confirmed.append(picked)


                else:

                    # 如果没有找打精确的 & high中还有多个（options合适、confirmed中：选择放）、

                    for h in high[:3]:

                        picked = h.get('item_name')

                        if picked not in options and picked not in confirmed:
                            options.append(picked)

            else:
                # 询问是否能进入到options中
                mid = [m for m in matches if m['score'] >= 0.6 and m.get("item_name") not in options and m.get('item_name') not in confirmed]

                if mid:
                    for m in mid[:3]:
                        picked = m.get('item_name')
                        options.append(picked)
        return confirmed, options



    def _match_vector(self, item_names):  # 1 usage
        """
        职责：根据LLM提取的主体名，查询向量数据库
        Args:
            item_names: LLM提取的主体名

        Returns:
            List[Dict[str, Any]]: 每一个item_name下的查询结果
            Dict[str,Any]:{"extracted_name":"LLM提取出来的主体名字", "matches":[{"item_name":"向量数据库的主体名","score":"结果分数值"}]}
        """
        # 1. 定义最终搜索结果
        search_results = []

        # 2. 获取milvus_client
        milvus_client = get_milvus_client()
        if milvus_client is None:
            return []

        # 3. 获取嵌入模型
        embedding_model = get_beg_m3_embedding_model()
        if embedding_model is None:
            logger.error(f"获取嵌入模型失败")

            return search_results

        # 3. 嵌入item_name获取稠密、稀疏向量
        hybrid_embedding_result = generate_hybrid_embeddings(embedding_model,item_names)

        # 4. 遍历LLM提取的所有主体名
        for index, extract_item_name in enumerate(item_names):
            # 混合向量检索
            # 4.1 创建混合检索的请求
            hybrid_search_requests = create_hybrid_search_requests(
                dense_vector=hybrid_embedding_result['dense'][index],
                sparse_vector=hybrid_embedding_result['sparse'][index],
            )

            # 4.2 执行混合检索的请求
            # （milvus集成bgem3嵌入模型只会对“稠密向量”进行L2的归一化：IP和COSINE【-1, 1】相等 但是不会对稀疏向量进行归一化【权重】）
            # （WeightedRanker: 属性: norm_score: 权重融合排序器: 对稠密向量检索的结果的分数值以及稀疏向量检索到的结果“分数值”进行归一化: 为了统一最后在排序的时候，各个向量维度的结果用权重计算的时候，公平）--- 【0,1】
            hybrid_search_result = execute_hybrid_search_query(milvus_client, collection_name="item_name_collection",
                                        search_requests=hybrid_search_requests,
                                        ranker_weights=(0.5, 0.5), norm_score=True)

            # 4.3 解析混合检索请求的结果对象
            item_name_search_result = {
                "extracted_name": extract_item_name,
                "matches": [
                    {"item_name": h["entity"]["item_name"], "score": h["distance"]}
                    for h in (hybrid_search_result[0] if hybrid_search_result else [])
                ]
            }
            # 4.4 将构建好的查询结果放入到最终搜索结果中
            search_results.append(item_name_search_result)

        return search_results

    def _item_name_score_filter(self, confirmed: List[str], search_results: List[Dict[str, Any]]):  # 1 usage  new *
        """
        item_names: 有三个item_name
        item_name1:0.9 （最相似的（基准））
        item_name2:0.88 （真实比对）
        item_name3:0.66 （可能误判）
        分数差的阈值：0.15
        主要责任：将误判的item_name冲confirmed删除掉。留下真实的item_name
        Args:
            confirmed:
            search_result:

        Returns:

        """
        # 1. 定义字典容器（存储confirmed中item_name在向量数据库中的分数值）
        item_name_score = {}

        for search_result in search_results:
            # 1. 获取matches
            matches = search_result.get('matches')
            for m in matches:
                score = m.get('score')
                item_name = m.get('item_name')
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name) or 0, score)

        # 2. 对item_name_score进行排序
        sorted_item_name_score = sorted(item_name_score.items(), key=lambda x: x[1], reverse=True)

        # 3. 取出分数值最大的（问题询问的比较明确）
        max_item_name_score = sorted_item_name_score[0][1]
        return [name for name, score in item_name_score.items() if max_item_name_score - score <= 0.15]



class ItemNameConfirmNode(BaseNode):
    name = "item_name_confirm_node"

    def __init__(self):
        # 必须调用父类构造：BaseNode.__init__ 负责设置 self.config / self.logger，
        # 漏掉会让 BaseNode.__call__ 的异常分支 self.logger.error 直接 AttributeError，
        # 从而把 process 里真正的报错信息整个盖掉。
        super().__init__()
        self._item_name_extractor = ItemNameExtractor()
        self._item_name_aligner = ItemNameAligner()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 获取用户的原始问题
        original_query = state.get("original_query")
        session_id = state.get("session_id", "")

        # 1.1 从 MongoDB 读取本会话最近的历史对话
        #     （此前本节点完全没接历史，多轮指代消解与 item_names 回填都是缺的）
        chat_history = self._load_chat_history(session_id)

        # 2. 调用LLM提取主体名（本质：是如果直接基于用户的原始问题进行检索，质量很差。而我们实际需要的是明白用户真正想问你的主体是谁。）
        #    传入历史对话 → 支持"它/这个"这类代词的指代消解
        clean_llm_result = self._item_name_extractor.extract_item_name(original_query, chat_history)
        #2.1 获取item_names
        item_names = clean_llm_result.get('item_names')
        #2.2 获取rewitten_query
        rewritten_query = clean_llm_result.get('rewritten_query')

        if item_names:
            # 3. 查询向量数据库&过滤(评分对齐&分数差异过滤)
            confirmed, options = self._item_name_aligner.match_align_filter(item_names)

        else:
            confirmed, options = [], []

        # 4. 决定state的key值（继续、结束）修改state
        self._decide(state, item_names, confirmed, options, rewritten_query)

        # 5. 回填历史记录里为空的 item_names（把确认到的主体名补到之前的问答上）
        if confirmed:
            self._backfill_history_item_names(chat_history, confirmed)

        # 6. 把历史对话写入 state，供下游 answer_output 拼装 ANSWER_PROMPT 的【历史对话】区块
        state["history"] = chat_history

        return state

    def _load_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """读取本会话最近的历史消息；MongoDB 不可用时降级为空列表，不影响主链路。"""
        if not session_id:
            return []
        try:
            from knowledge.utils.mongo_history_util import get_recent_messages
            return get_recent_messages(session_id, limit=10) or []
        except Exception as e:
            self.logger.warning(f"读取历史对话失败，本轮按无历史处理: {e}")
            return []

    def _backfill_history_item_names(self, chat_history: List[Dict[str, Any]], confirmed: List[str]) -> None:
        """把 confirmed 的主体名回填到历史记录中 item_names 为空的那几条上。"""
        ids_to_update = [
            str(msg.get("_id")) for msg in (chat_history or [])
            if msg.get("_id") is not None and not msg.get("item_names")
        ]
        if not ids_to_update:
            return
        try:
            from knowledge.utils.mongo_history_util import update_message_item_names
            update_message_item_names(ids_to_update, confirmed)
        except Exception as e:
            self.logger.warning(f"回填历史 item_names 失败: {e}")

    def _decide(self, state: QueryGraphState, item_names: List[str], confirmed: List[str],
                options: List[str], rewritten_query: str):

        if confirmed:
            state['rewritten_query'] = rewritten_query
            state['item_names'] = confirmed

        elif options:
            # 电商合规场景适配：反问话术面向「主体（商家/店铺）」而非具体物品
            state['answer'] = (f"我不确定您指的是哪家商家/店铺。"
                               f"您是在询问以下主体吗：{'、'.join(options)}？")
        else:
            state['answer'] = "抱歉，我无法识别您询问的具体主体（商家/店铺）名称，请提供更准确的公司名称或统一社会信用代码。"


if __name__ == "__main__":
    test_state: QueryGraphState = {
        "original_query": "我们那家旗舰店，今年的资质和违规整改情况如何",
    }

    print(f"输入: {json.dumps(test_state, ensure_ascii=False, indent=2)}\n")

    node_item_name_confirm = ItemNameConfirmNode()
    result = node_item_name_confirm.process(test_state)
    print(f"确认主体: {result.get('item_names')}")
    print(f"改写查询: {result.get('rewritten_query')}")
    if result.get("answer"):
        print(f"拦截回复: {result.get('answer')}")