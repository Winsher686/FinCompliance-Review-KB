"""审查流程节点基类。

与 query_process.base.BaseNode 行为一致，但**额外保证 LangGraph 的控制流异常
（GraphInterrupt / 中断恢复）能够穿透**——这是 HITL 节点用 interrupt() 暂停图的前提：
若基类把 GraphInterrupt 也当成普通异常包装成 ReviewProcessError，图的暂停/恢复机制会失效。
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging

from knowledge.processor.review_process.config import ReviewConfig, get_config
from knowledge.processor.review_process.exceptions import ReviewProcessError
from knowledge.utils.sse_util import push_sse_event
from knowledge.utils.task_util import (
    add_running_task, add_done_task, get_done_task_list, get_running_task_list, get_task_status,
)
from knowledge.utils.observability import (
    ensure_trace_id, node_span, span_extras, trace_prefix,
)

T = TypeVar("T")  # 泛型状态类型


def is_control_flow_exception(exc: BaseException) -> bool:
    """判断是否为 LangGraph 内部控制流异常（中断/恢复/父命令等）。

    LangGraph 会抛出定义在 langgraph 包内的异常来暂停或跳转图执行，
    这类异常必须原样向上抛，不能被节点基类包装。
    """
    module = getattr(type(exc), "__module__", "") or ""
    return module.startswith("langgraph")


class BaseNode(ABC):
    """审查流程节点基类。

    提供统一的日志、任务追踪、SSE 进度上报与错误处理。
    """

    name: str = "base_node"

    def __init__(self, config: Optional[ReviewConfig] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"review.{self.name}")

    def __call__(self, state: T) -> T:
        """节点执行入口（LangGraph 调用）。

        统一做三件事：注入/透传 trace_id、记录节点耗时（可观测性）、维护任务进度。
        """
        trace_id = ensure_trace_id(state)
        task_id = state.get("task_id", "")
        is_stream = state.get("is_stream", False)
        prefix = trace_prefix(trace_id)

        with node_span(trace_id, self.name) as span:
            try:
                self.logger.info(f"{prefix}--- {self.name} 开始 ---")
                if task_id:
                    add_running_task(task_id, self.name)
                    if is_stream:
                        self._push_progress(task_id)

                result = self.process(state)

                self.logger.info(f"{prefix}--- {self.name} 完成 ---")
                span["extras"].update(span_extras(result))
                if task_id:
                    add_done_task(task_id, self.name)
                    if is_stream:
                        self._push_progress(task_id)

                # trace_id 回写进 state（LangGraph 会把返回的 dict 合并进全局状态）
                if isinstance(result, dict) and not result.get("trace_id"):
                    return {**result, "trace_id": trace_id}
                return result

            except Exception as e:
                # LangGraph 控制流异常必须原样抛出（HITL interrupt 依赖）
                if is_control_flow_exception(e):
                    raise
                self.logger.error(f"{prefix}{self.name} 执行失败: {e}")
                raise ReviewProcessError(message=str(e), node_name=self.name, cause=e)

    @staticmethod
    def _push_progress(task_id: str):
        push_sse_event(task_id, "progress", {
            "status": get_task_status(task_id),
            "done_list": get_done_task_list(task_id),
            "running_list": get_running_task_list(task_id),
        })

    @abstractmethod
    def process(self, state: T) -> T:
        """节点核心处理逻辑，子类必须实现。"""
        pass

    def log_step(self, step_name: str, message: str = ""):
        """记录步骤日志。"""
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


def setup_logging(level: int = logging.INFO):
    """配置审查流程日志。"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
