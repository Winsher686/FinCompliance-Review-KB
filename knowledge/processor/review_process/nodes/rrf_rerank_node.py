"""节点 4：RRF 融合 + BGE-Reranker 精排。

- RRF（Reciprocal Rank Fusion）：把「向量检索 / 稀疏BM25检索 / 图谱路径」三路结果按排名
  加权融合，规避不同检索路径分数量纲不一致的问题（复用 query_process 的融合思路）；
- 精排：用 BGE-Reranker 对融合后的候选做重排，得到最终送入证据链与报告的高相关文档。
"""

from typing import Any, Dict, List, Tuple

from knowledge.processor.review_process.base import BaseNode
from knowledge.processor.review_process.state import ReviewGraphState


class RrfRerankNode(BaseNode):
    name = "rrf_rerank_node"

    def process(self, state: ReviewGraphState) -> ReviewGraphState:
        # 1. RRF 融合
        rrf_chunks = self._rrf_merge(state)

        # 2. 精排
        reranked_docs = self._rerank(state, rrf_chunks)

        self.logger.info(f"RRF 融合 {len(rrf_chunks)} 条 → 重排后 {len(reranked_docs)} 条")
        return {"rrf_chunks": rrf_chunks, "reranked_docs": reranked_docs}

    # ==================== RRF ====================
    def _rrf_merge(self, state: ReviewGraphState) -> List[Dict[str, Any]]:
        vector_docs = self._normalize(state.get("embedding_chunks") or [])
        sparse_docs = self._normalize(state.get("sparse_chunks") or [])
        kg_docs = self._normalize(state.get("kg_chunks") or [])

        sources: List[Tuple[List[Dict[str, Any]], float]] = [
            (vector_docs, self.config.rrf_vector_weight),
            (sparse_docs, self.config.rrf_sparse_weight),
            (kg_docs, self.config.rrf_kg_weight),
        ]

        scores: Dict[str, float] = {}
        data: Dict[str, Dict[str, Any]] = {}
        for docs, weight in sources:
            for rank, doc in enumerate(docs, 1):
                chunk_id = doc.get("chunk_id")
                if chunk_id is None:
                    continue
                scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (self.config.rrf_k + rank)
                data.setdefault(chunk_id, doc)

        merged = sorted(
            [(data[cid], score) for cid, score in scores.items()],
            key=lambda x: x[1], reverse=True,
        )
        top = merged[: self.config.rrf_max_results]
        # 保留 rrf 分数，便于调试
        return [{**doc, "rrf_score": score} for doc, score in top]

    @staticmethod
    def _normalize(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把各路结果归一为文档字典列表（含 chunk_id/content）。"""
        out: List[Dict[str, Any]] = []
        for doc in docs or []:
            if not isinstance(doc, dict):
                continue
            entity = doc.get("entity")
            if isinstance(entity, dict):
                out.append(entity)
            else:
                out.append(doc)
        return out

    # ==================== 重排 ====================
    def _rerank(self, state: ReviewGraphState, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not docs:
            return []

        query = (state.get("rewritten_query") or state.get("question") or "").strip()
        try:
            from knowledge.utils.bge_rerank_util import get_reranker_model
            model = get_reranker_model()
        except Exception as e:
            self.logger.debug(f"重排序依赖不可用，保留 RRF 顺序: {e}")
            model = None

        if model is None or not query:
            # 退化：使用 RRF 分数作为排序分
            return [{**d, "score": d.get("rrf_score", 0.0)} for d in docs[: self.config.rerank_max_top_k]]

        pairs = [(query, d.get("content", "")) for d in docs]
        try:
            scores = model.compute_score(sentence_pairs=pairs)
        except Exception as e:
            self.logger.error(f"重排序计算失败，保留 RRF 顺序: {e}")
            return [{**d, "score": d.get("rrf_score", 0.0)} for d in docs[: self.config.rerank_max_top_k]]

        if isinstance(scores, (int, float)):
            scores = [scores]
        scored = [{**d, "score": float(s)} for d, s in zip(docs, scores)]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: self.config.rerank_max_top_k]
