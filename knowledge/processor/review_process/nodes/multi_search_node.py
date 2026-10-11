"""节点 3：多路检索（向量 + 稀疏/BM25 + 图谱遍历）。

**三路真并发召回**：三路检索彼此无数据依赖，因此放进同一个线程池同时发起，
墙钟耗时 ≈ max(三路) 而不是 sum(三路)。跨机房部署时向量检索（Milvus）与图谱检索（Neo4j）
的 RTT 是主要成本，串行执行会把这部分延迟直接相加：

    串行：vec(400ms) + sparse(300ms) + kg(900ms) = 1600ms
    并发：max(400, 300, 900) = 900ms 且仍有 1 路冗余

1. 向量检索：BGE-M3 稠密+稀疏混合检索（Milvus hybrid_search）；
2. 稀疏/BM25：仅用稀疏向量做 ANN 检索（对应 BM25 关键词召回）；
3. 图谱遍历：从 Neo4j 抽取实体关系与多跳路径（如 品牌→授权→商家→在售→商品），并把路径转成
   「伪切片」参与后续 RRF 融合，路径本身也作为合规证据返回；
   该路内部同样做批量下推（``get_entity_relations_batch`` / ``find_paths_batch``），
   把「每实体一次往返」的 N+1 压成常数次往返。

容错策略（并发场景下最关键的三条）：
- **单路失败不影响其他路**：任一路抛异常 → 该路降级为 ``None``，其余两路结果照常返回；
- **单路超时**：超过 ``REVIEW_SEARCH_BRANCH_TIMEOUT`` 秒仍未返回 → 该路标记 timeout 并放弃等待，
  不阻塞整条审查流程（慢的那一路线程由线程池后台收尾）；
- **并发不外泄**：分支内异常不外抛，节点对外仍然是「返回三个检索结果字段」的稳定契约。

所有外部依赖（Milvus / Neo4j / BGE-M3）均为「延迟导入 + 异常兜底」，
缺失时返回空结果，保证在无三方服务时仍能跑通整条审查流程（离线可跑）。
"""

import re
import time
from concurrent.futures import (
    Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed,
)
from typing import Any, Callable, Dict, List, Sequence, Tuple

from knowledge.processor.review_process.base import BaseNode
from knowledge.processor.review_process.state import ReviewGraphState
from knowledge.utils.observability import record_metric

# 电商经营主体名匹配（用于从问题中抽取待遍历实体）
_ENTITY_RE = re.compile(
    r"[\u4e00-\u9fa5A-Za-z0-9（）()·]{2,40}?"
    r"(?:股份有限公司|有限责任公司|有限公司|旗舰店|专营店|专卖店|企业店|网店|店铺|商家|公司|"
    r"平台|市场监管|监督管理局|品牌|供应链|贸易)"
)

#: 分支名 → 中文标签（日志与指标 tag 共用，避免两处维护）
_BRANCH_LABELS = {"vector": "向量", "sparse": "稀疏", "kg": "图谱"}

BranchCallable = Callable[..., Any]


class MultiSearchNode(BaseNode):
    name = "multi_search_node"

    def process(self, state: ReviewGraphState) -> ReviewGraphState:
        results, branch_ms, errors = self.parallel_retrieve(self._build_jobs(state))

        embedding_chunks = results.get("vector") or []
        sparse_chunks = results.get("sparse") or []
        kg_chunks, kg_triples, kg_paths = self._unpack_kg(results.get("kg"))

        wall_ms = max(branch_ms.values(), default=0.0)   # 并发墙钟 ≈ 最慢的那一路

        self.logger.info(
            "多路检索完成（并发）："
            f"{self._describe('vector', len(embedding_chunks), branch_ms)} "
            f"{self._describe('sparse', len(sparse_chunks), branch_ms)} "
            f"{self._describe('kg', len(kg_triples), branch_ms)} "
            f"图谱路径={len(kg_paths)} 墙钟≈{wall_ms:.0f}ms"
            + (f" 失败分支={errors}" if errors else "")
        )

        # 打点：分支级延迟进指标，便于看「到底是哪一路拖慢了检索」
        for branch, cost in branch_ms.items():
            record_metric("search_branch_ms", cost, {"branch": branch})
        record_metric("search_wall_ms", wall_ms)

        return {
            "embedding_chunks": embedding_chunks,
            "sparse_chunks": sparse_chunks,
            "kg_chunks": kg_chunks,
            "kg_triples": kg_triples,
            "kg_paths": kg_paths,
            "search_branch_ms": branch_ms,
            "search_wall_ms": round(wall_ms, 2),
        }

    # ==================== 并发调度 ====================
    def _build_jobs(self, state: ReviewGraphState) -> Dict[str, Tuple[BranchCallable, Tuple[Any, ...]]]:
        """把三路检索组装成「分支名 → (可调用, 参数)」的调度任务表。

        单独抽出来是为了：① 与 :meth:`parallel_retrieve` 解耦，便于单测直接构造任务；
        ② 明确「三路检索彼此无数据依赖」这一并发前提（参数只来自 state，没有互相传递）。
        """
        query = (state.get("rewritten_query") or state.get("question") or "").strip()
        company = (state.get("company") or "").strip()
        hyde_doc = (state.get("hyde_doc") or "").strip()
        return {
            # 1. 向量检索（问题 + 假设性文档 拼接后检索）
            "vector": (self._vector_search, (f"{query}\n{hyde_doc}".strip(), company)),
            # 2. 稀疏/BM25 检索（仅稀疏向量）
            "sparse": (self._sparse_search, (query, company)),
            # 3. 图谱遍历（关系 + 多跳路径）
            "kg": (self._kg_search, (query, company)),
        }

    def parallel_retrieve(
        self, jobs: Dict[str, Tuple[BranchCallable, Sequence[Any]]],
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, str]]:
        """并发执行多路检索，返回 ``(结果, 各分支耗时ms, 失败原因)``。

        - 每路在独立线程中执行，异常在分支内被捕获并记录为失败原因；
        - 整体等待上限 ``self.config.search_branch_timeout``，超时分支按失败处理并放弃等待；
        - 无论成功失败，返回值结构恒定，调用方不需要 try/except。
        """
        timeout = float(getattr(self.config, "search_branch_timeout", 20) or 20)
        max_workers = max(1, int(getattr(self.config, "search_max_workers", len(jobs)) or len(jobs)))

        results: Dict[str, Any] = {}
        durations: Dict[str, float] = {}
        errors: Dict[str, str] = {}

        pool = ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(jobs))),
                                  thread_name_prefix="review-retr")
        try:
            futures: Dict[Future, str] = {
                pool.submit(self._call_branch, fn, tuple(args)): key
                for key, (fn, args) in jobs.items()
            }
            # 先到先收集：任一路返回就能立刻落袋，不必等慢的那一路
            try:
                for future in as_completed(futures, timeout=timeout):
                    self._collect(futures[future], future, results, durations, errors)
            except FutureTimeoutError:
                pass

            # 超时未返回的分支：放弃等待（不再 join），标记为 timeout 后继续走流程
            for future, key in futures.items():
                if future.done():
                    continue
                durations[key] = round(timeout * 1000, 2)
                errors[key] = f"timeout(>{timeout}s)"
                self.logger.warning(f"检索分支 {_BRANCH_LABELS.get(key, key)} 超时(>{timeout}s)，已降级为空结果")
                future.cancel()

            # 兜底：恰好卡在超时边界完成、未被 as_completed 吐出的分支，仍然收下它的结果
            for future, key in futures.items():
                if future.done() and key not in results and key not in errors:
                    self._collect(key, future, results, durations, errors)
        finally:
            # wait=False：绝不因为某一路慢而把整条审查流程拖住
            pool.shutdown(wait=False, cancel_futures=True)

        return results, durations, errors

    @staticmethod
    def _call_branch(fn: BranchCallable, args: Tuple[Any, ...]) -> Tuple[Any, float, str]:
        """在线程内执行单个检索分支；异常不抛出，转为错误字符串返回。"""
        began = time.monotonic()
        try:
            value = fn(*args)
            return value, round((time.monotonic() - began) * 1000, 2), ""
        except Exception as e:   # 分支级兜底：任一路失败不影响其他路
            return None, round((time.monotonic() - began) * 1000, 2), f"{type(e).__name__}: {e}"

    def _collect(self, key: str, future: Future, results: Dict[str, Any],
                 durations: Dict[str, float], errors: Dict[str, str]) -> None:
        """收集一个已完成分支的结果。"""
        try:
            value, cost, err = future.result()
        except Exception as e:   # 理论上 _call_branch 已兜住，这里是最后一道保险
            results[key] = None
            errors[key] = f"{type(e).__name__}: {e}"
            return
        durations[key] = cost
        if err:
            results[key] = None
            errors[key] = err
            self.logger.warning(f"检索分支 {_BRANCH_LABELS.get(key, key)} 失败（已降级）：{err}")
        else:
            results[key] = value

    @staticmethod
    def _describe(branch: str, count: int, branch_ms: Dict[str, float]) -> str:
        label = _BRANCH_LABELS.get(branch, branch)
        cost = branch_ms.get(branch)
        return f"{label}={count}({cost:.0f}ms)" if cost is not None else f"{label}={count}"

    @staticmethod
    def _unpack_kg(value: Any) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        """把图谱分支返回值归一为 (kg_chunks, kg_triples, kg_paths)。"""
        if isinstance(value, (tuple, list)) and len(value) == 3:
            return list(value[0] or []), list(value[1] or []), list(value[2] or [])
        return [], [], []

    # ==================== 向量检索 ====================
    def _vector_search(self, text: str, company: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        try:
            from knowledge.utils.bge_m3_embedding_util import (
                get_beg_m3_embedding_model, generate_hybrid_embeddings,
            )
            from knowledge.utils.milvus_util import (
                get_milvus_client, create_hybrid_search_requests, execute_hybrid_search_query,
            )
        except Exception as e:
            self.logger.debug(f"向量检索依赖不可用，跳过: {e}")
            return []

        model = get_beg_m3_embedding_model()
        client = get_milvus_client()
        if not model or not client:
            return []

        emb = generate_hybrid_embeddings(model, [text])
        if not emb:
            return []

        expr = self._item_filter(company)
        reqs = create_hybrid_search_requests(
            dense_vector=emb["dense"][0], sparse_vector=emb["sparse"][0],
            expr=expr, limit=self.config.embedding_search_limit,
        )
        reps = execute_hybrid_search_query(
            client, collection_name=self.config.chunks_collection, search_requests=reqs,
            norm_score=True, limit=self.config.embedding_search_limit,
            output_fields=["chunk_id", "content", "title", "item_name", "doc_type", "publish_date", "source"],
        )
        return self._normalize(reps)

    # ==================== 稀疏/BM25 检索 ====================
    def _sparse_search(self, text: str, company: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        try:
            from knowledge.utils.bge_m3_embedding_util import (
                get_beg_m3_embedding_model, generate_hybrid_embeddings,
            )
            from knowledge.utils.milvus_util import get_milvus_client, execute_single_field_search
        except Exception as e:
            self.logger.debug(f"稀疏检索依赖不可用，跳过: {e}")
            return []

        model = get_beg_m3_embedding_model()
        client = get_milvus_client()
        if not model or not client:
            return []

        emb = generate_hybrid_embeddings(model, [text])
        if not emb:
            return []

        reps = execute_single_field_search(
            milvus_client=client,
            collection_name=self.config.chunks_collection,
            vector=emb["sparse"][0],
            anns_field="sparse_vector",
            metric_type="IP",
            expr=self._item_filter(company),
            limit=self.config.embedding_search_limit,
            output_fields=["chunk_id", "content", "title", "item_name", "doc_type", "publish_date", "source"],
        )
        return self._normalize(reps)

    # ==================== 图谱遍历（批量下推，无 N+1） ====================
    def _kg_search(self, query: str, company: str) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        try:
            from knowledge.utils.neo4j_util import (
                build_entity_pairs, find_paths_batch, get_entity_relations_batch,
            )
        except Exception as e:
            self.logger.debug(f"图谱依赖不可用，跳过: {e}")
            return [], [], []

        entities = self._extract_entities(query, company)

        # 1. 实体一跳关系：一次 UNWIND 批量取回，替代「每个实体开一次 session」
        seed_names = [company] if company else entities[:1]
        relation_map = get_entity_relations_batch(
            seed_names or entities[:1], item_name=company or None, limit_per_entity=self.config.kg_max_relations,
        )
        relations: List[Dict[str, Any]] = []
        for name in (seed_names or entities[:1]):
            relations.extend(relation_map.get(name) or [])
        if not relations and entities:
            fallback = get_entity_relations_batch(entities[:1], limit_per_entity=self.config.kg_max_relations)
            relations = list(fallback.get(entities[0]) or [])

        triples = [f"[{r['head']}] -({r['rel']})-> {r['tail']}" for r in relations]

        # 2. 两两实体间的多跳路径：pairs 一次性提交，一次往返查完（原为 N×(N-1)/2 次往返）
        pairs = build_entity_pairs(entities, max_entities=3)
        paths = find_paths_batch(
            pairs, max_hops=self.config.kg_max_path_hops, limit_per_pair=self.config.kg_max_paths,
        )[: self.config.kg_max_paths]

        # 3. 路径转伪切片，参与 RRF 融合
        kg_chunks = [
            {
                "entity": {
                    "chunk_id": f"graph_path_{i}", "content": p.get("path_text", ""),
                    "title": "图谱路径", "source": "graph",
                },
                "distance": round(1.0 / (i + 1), 6),
            }
            for i, p in enumerate(paths)
        ]
        return kg_chunks, triples, paths

    # ==================== 工具 ====================
    @staticmethod
    def _item_filter(company: str) -> Any:
        """主体（公司）维度过滤表达式；无主体时不过滤。"""
        if not company:
            return None
        safe = company.replace("'", "")
        return f"item_name in ['{safe}']"

    @staticmethod
    def _extract_entities(text: str, company: str) -> List[str]:
        """从文本中抽取待遍历的实体名（主体 + 正则命中的公司/机构）。"""
        names: List[str] = []
        if company:
            names.append(company)
        for match in _ENTITY_RE.finditer(text or ""):
            name = match.group(0).strip()
            if name and name not in names:
                names.append(name)
        return names[:5]

    @staticmethod
    def _normalize(reps: Any) -> List[Dict[str, Any]]:
        """把 Milvus 检索结果统一为 [{"entity": {...}, "distance": ...}]。"""
        if not reps or not reps[0]:
            return []
        out: List[Dict[str, Any]] = []
        for hit in reps[0]:
            entity = hit.get("entity") if isinstance(hit, dict) else getattr(hit, "entity", None)
            distance = hit.get("distance") if isinstance(hit, dict) else getattr(hit, "distance", None)
            if entity:
                out.append({"entity": dict(entity), "distance": distance})
        return out
