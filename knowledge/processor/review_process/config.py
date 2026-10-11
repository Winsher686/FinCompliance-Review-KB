"""审查流程配置管理模块。

集中管理审查流程的配置项，支持环境变量覆盖，全部懒加载。
"""

from dataclasses import dataclass, field
from typing import Optional
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ReviewConfig:
    """审查流程配置。"""

    # ==================== 检索配置 ====================
    embedding_search_limit: int = field(
        default_factory=lambda: int(os.getenv("REVIEW_EMBEDDING_SEARCH_LIMIT", "10"))
    )
    hyde_search_limit: int = field(
        default_factory=lambda: int(os.getenv("REVIEW_HYDE_SEARCH_LIMIT", "5"))
    )
    rerank_max_top_k: int = field(
        default_factory=lambda: int(os.getenv("REVIEW_RERANK_MAX_TOP_K", "10"))
    )
    max_context_chars: int = field(
        default_factory=lambda: int(os.getenv("REVIEW_MAX_CONTEXT_CHARS", "12000"))
    )

    # ==================== 检索并发配置 ====================
    # 单路检索（向量 / 稀疏 / 图谱）各自的超时秒数；超时后该路降级为空结果，不阻塞其他两路
    search_branch_timeout: float = field(
        default_factory=lambda: float(os.getenv("REVIEW_SEARCH_BRANCH_TIMEOUT", "20"))
    )
    # 多路检索线程池大小（当前三路检索，默认 3）
    search_max_workers: int = field(
        default_factory=lambda: int(os.getenv("REVIEW_SEARCH_MAX_WORKERS", "3"))
    )

    # ==================== RRF 配置 ====================
    rrf_k: int = field(default_factory=lambda: int(os.getenv("RRF_K", "60")))
    rrf_max_results: int = field(default_factory=lambda: int(os.getenv("RRF_MAX_RESULTS", "10")))
    rrf_vector_weight: float = field(default_factory=lambda: float(os.getenv("RRF_VECTOR_WEIGHT", "1.0")))
    rrf_sparse_weight: float = field(default_factory=lambda: float(os.getenv("RRF_SPARSE_WEIGHT", "1.0")))
    rrf_kg_weight: float = field(default_factory=lambda: float(os.getenv("RRF_KG_WEIGHT", "0.7")))

    # ==================== 图谱配置 ====================
    kg_max_path_hops: int = field(default_factory=lambda: int(os.getenv("KG_MAX_PATH_HOPS", "3")))
    kg_max_paths: int = field(default_factory=lambda: int(os.getenv("KG_MAX_PATHS", "5")))
    kg_max_relations: int = field(default_factory=lambda: int(os.getenv("KG_MAX_RELATIONS", "50")))

    # ==================== HITL（人工复核）配置 ====================
    hitl_enabled: bool = field(default_factory=lambda: os.getenv("HITL_ENABLED", "1") == "1")
    # 触发人工复核的风险等级（逗号分隔），默认高风险触发
    hitl_risk_levels: str = field(default_factory=lambda: os.getenv("HITL_RISK_LEVELS", "高"))
    # 触发人工复核的审查类型（逗号分隔），如正式报告/处罚认定
    hitl_review_types: str = field(
        default_factory=lambda: os.getenv("HITL_REVIEW_TYPES", "report,violation,product_compliance")
    )

    # ==================== LLM 配置 ====================
    default_model: str = field(default_factory=lambda: os.getenv("MODEL", ""))

    # ==================== Milvus / Neo4j ====================
    milvus_url: str = field(default_factory=lambda: os.getenv("MILVUS_URL", ""))
    chunks_collection: str = field(default_factory=lambda: os.getenv("CHUNKS_COLLECTION", ""))
    neo4j_database: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))

    @classmethod
    def from_env(cls) -> "ReviewConfig":
        return cls()

    def hitl_trigger_risk_levels(self) -> set:
        return {x.strip() for x in (self.hitl_risk_levels or "").split(",") if x.strip()}

    def hitl_trigger_review_types(self) -> set:
        return {x.strip() for x in (self.hitl_review_types or "").split(",") if x.strip()}


_config: Optional[ReviewConfig] = None


def get_config() -> ReviewConfig:
    """获取配置单例。"""
    global _config
    if _config is None:
        _config = ReviewConfig.from_env()
    return _config
