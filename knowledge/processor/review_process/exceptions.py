"""审查流程异常定义。

与 import_process / query_process 保持一致的异常风格，便于统一处理。
"""

from typing import Optional


class ReviewProcessError(Exception):
    """审查流程基础异常。"""

    def __init__(self, message: str, node_name: str = "", cause: Optional[Exception] = None):
        self.message = message
        self.node_name = node_name
        self.cause = cause
        super().__init__(f"[{node_name}] {message}" if node_name else message)


class ValidationError(ReviewProcessError):
    """参数校验失败。"""


class StateFieldError(ReviewProcessError):
    """状态字段缺失或类型错误。"""

    def __init__(self, node_name: str, field_name: str, expected_type=None):
        message = f"状态字段 '{field_name}' 缺失或类型错误"
        if expected_type is not None:
            message += f"（期望 {expected_type}）"
        super().__init__(message=message, node_name=node_name)


class HumanReviewRequired(ReviewProcessError):
    """需要人工复核（由 HITL 节点抛出，用于上层识别挂起状态）。"""

    def __init__(self, review_id: str = "", message: str = "需要人工复核"):
        self.review_id = review_id
        super().__init__(message=message, node_name="hitl_node")
