"""审查流程依赖注入。"""

from functools import lru_cache

from knowledge.processor.review_process.services.review_service import ReviewService


@lru_cache
def get_review_service() -> ReviewService:
    """获取审查业务服务单例。"""
    return ReviewService()
