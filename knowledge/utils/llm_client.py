"""LLM 客户端与 LLM 输出解析的统一入口。

设计原则：**把 LLM 当作非确定性网络依赖来对待**——
超时、重试、退避、结构化输出兜底、可观测的调用日志一个都不能少。

本模块分三层：

1. **解析层（纯 Python，无三方依赖）**：``strip_code_fence`` / ``parse_json_response`` /
   ``estimate_tokens``。所有需要解析 LLM JSON 输出的节点都应引用这里，
   避免 ``re.sub("^```json")`` + ``json.loads`` 在全项目复制多份、行为不一致。
2. **客户端层**：``LLMClient`` 封装 ``ChatDeepSeek``，提供 ``timeout`` + 指数退避重试 +
   统一异常归一（把底层 SDK 异常归一为 ``LLMUnavailableError`` / ``LLMTimeoutError``）。
3. **工厂**：``get_llm_client(...)`` 保持历史签名（``model_name`` / ``json_mode``），
   依赖缺失时返回 ``None``，让调用方走模板兜底，保证离线可跑。

相关环境变量（均有默认值，见 ``knowledge/.env.example``）：
    LLM_TIMEOUT          单次请求超时（秒），默认 60
    LLM_MAX_RETRIES      最大重试次数，默认 2（即最多请求 3 次）
    LLM_RETRY_BACKOFF    指数退避基数（秒），默认 0.8
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "LLMError",
    "LLMUnavailableError",
    "LLMTimeoutError",
    "strip_code_fence",
    "extract_json_block",
    "parse_json_response",
    "estimate_tokens",
    "LLMClient",
    "get_llm_client",
]


# ==================== 异常 ====================

class LLMError(Exception):
    """LLM 调用相关异常基类。"""


class LLMUnavailableError(LLMError):
    """LLM 不可用（依赖缺失 / 鉴权失败 / 网络不可达 / 服务端 5xx）。"""


class LLMTimeoutError(LLMError):
    """LLM 调用超时。"""


# ==================== 解析层（纯 Python） ====================

_FENCE_OPEN_RE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*")
_FENCE_CLOSE_RE = re.compile(r"\s*```\s*$")


def strip_code_fence(text: str) -> str:
    """去掉 Markdown 代码围栏（```json ... ``` / ``` ... ```）。

    同时兼容只写了前半段围栏、以及正文里夹杂其他文本的情况：
    先用正则剥离首尾围栏，再退化为「取第一个 { 到最后一个 }」/
    「取第一个 [ 到最后一个 ]」。

    Args:
        text: LLM 的原始输出

    Returns:
        去除围栏后的文本（不含首尾空白）
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    stripped = _FENCE_CLOSE_RE.sub("", _FENCE_OPEN_RE.sub("", raw)).strip()
    if stripped != raw:
        return stripped
    return raw


def extract_json_block(text: str) -> str:
    """从文本中截取最外层的 JSON 对象或数组（容忍前后有解释性文字）。

    采用括号配平扫描，能正确处理字符串内的引号与转义。
    找不到配平的 JSON 时返回空串。
    """
    raw = strip_code_fence(text)
    if not raw:
        return ""

    start = -1
    opener = ""
    for idx, ch in enumerate(raw):
        if ch in "{[":
            start = idx
            opener = ch
            break
    if start < 0:
        return ""

    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(raw)):
        ch = raw[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return raw[start:idx + 1]
    return ""


def parse_json_response(text: str, default: Any = None) -> Any:
    """解析 LLM 输出的 JSON；解析失败返回 ``default`` 而不是抛异常。

    依次尝试：直接 ``json.loads`` → 去围栏 → 括号配平截取。
    这样调用方只需判断返回值是否为 ``None``，无需各自 try/except。

    Args:
        text: LLM 的原始输出
        default: 解析失败时的返回值（默认 ``None``）

    Returns:
        解析出的 dict / list，或 ``default``
    """
    if not text:
        return default
    candidates = [text, strip_code_fence(text), extract_json_block(text)]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            continue
    logger.debug("LLM 输出无法解析为 JSON（前 120 字符）: %s", str(text)[:120])
    return default


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（用于成本观测，不追求与 tokenizer 完全一致）。

    经验口径：中文约 1 token ≈ 1.5 字符，英文约 1 token ≈ 4 字符。
    中英混排时按「CJK 字符按 1 token、其余每 4 字符 1 token」折算。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    others = len(text) - cjk
    return int(cjk + others / 4) + 1


# ==================== 客户端层 ====================

_TRANSIENT_HINTS = ("timeout", "timed out", "connection", "reset", "unavailable",
                    "rate limit", "429", "500", "502", "503", "504", "overloaded")


class LLMClient:
    """带超时与指数退避重试的 LLM 客户端包装。

    对外仅暴露 ``invoke`` / ``invoke_text``，保持与 ``langchain`` 客户端
    相近的调用体感（``client.invoke([SystemMessage(...), HumanMessage(...)])``），
    便于存量节点平滑迁移。
    """

    def __init__(self, client: Any, model_name: str = "",
                 timeout: float = 60.0, max_retries: int = 2,
                 backoff: float = 0.8):
        self._client = client
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.backoff = backoff
        self.last_usage: dict = {}

    # ---------- 内部 ----------
    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        """判断异常是否值得重试（超时 / 连接 / 限流 / 5xx 才重试）。"""
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(hint in text for hint in _TRANSIENT_HINTS)

    # ---------- 对外 ----------
    @staticmethod
    def _normalize(messages: Any) -> Any:
        """兼容两种调用习惯：消息列表 或 纯字符串提示词。

        ``langchain`` 接受 str 也接受 ``list[BaseMessage]``；
        若把 str 误当可迭代对象展开会退化成"逐字符"输入，这里统一兜住。
        """
        if isinstance(messages, str):
            return messages
        return list(messages)

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        """调用模型，失败时按指数退避重试。

        Args:
            messages: ``list[BaseMessage]`` 或纯字符串提示词

        Raises:
            LLMTimeoutError: 最终仍超时
            LLMUnavailableError: 最终仍不可用
        """
        attempts = self.max_retries + 1
        last_exc: Optional[Exception] = None
        payload = self._normalize(messages)

        for attempt in range(1, attempts + 1):
            started = time.monotonic()
            try:
                response = self._client.invoke(payload, **kwargs)
                elapsed = time.monotonic() - started
                self._record_usage(response, elapsed)
                return response
            except Exception as exc:  # 归类后决定是否重试
                elapsed = time.monotonic() - started
                last_exc = exc
                transient = self._is_transient(exc)
                if attempt >= attempts or not transient:
                    logger.warning(
                        "LLM 调用失败（第 %s/%s 次，耗时 %.2fs，可重试=%s）: %s",
                        attempt, attempts, elapsed, transient, exc,
                    )
                    break
                sleep = self.backoff * (2 ** (attempt - 1))
                logger.warning(
                    "LLM 调用异常（第 %s/%s 次，耗时 %.2fs），%.1fs 后重试: %s",
                    attempt, attempts, elapsed, sleep, exc,
                )
                time.sleep(sleep)

        text = f"{last_exc}".lower()
        if "timeout" in text or "timed out" in text:
            raise LLMTimeoutError(f"LLM 调用超时（重试 {self.max_retries} 次后仍失败）: {exc_safe(last_exc)}")
        raise LLMUnavailableError(f"LLM 调用失败（重试 {self.max_retries} 次后仍失败）: {exc_safe(last_exc)}")

    def stream(self, messages: Any, **kwargs: Any):
        """流式调用（SSE 场景使用）。

        流式不做重试（已吐出的 token 无法回滚），只做异常归一；
        调用方负责捕获 ``LLMUnavailableError`` 并走模板兜底。
        """
        payload = self._normalize(messages)
        try:
            return self._client.stream(payload, **kwargs)
        except Exception as exc:
            raise LLMUnavailableError(f"LLM 流式调用失败: {exc_safe(exc)}")

    def invoke_text(self, messages: Any, **kwargs: Any) -> str:
        """调用模型并直接返回纯文本内容（已 strip）。"""
        response = self.invoke(messages, **kwargs)
        return (getattr(response, "content", "") or "").strip()

    def invoke_json(self, messages: Any, default: Any = None, **kwargs: Any) -> Any:
        """调用模型并解析为 JSON，失败返回 ``default``。"""
        return parse_json_response(self.invoke_text(messages, **kwargs), default=default)

    def _record_usage(self, response: Any, elapsed: float) -> None:
        """记录 token 用量与耗时（用于成本/性能观测）。"""
        usage = getattr(response, "usage_metadata", None) or {}
        prompt_tokens = usage.get("input_tokens")
        completion_tokens = usage.get("output_tokens")
        self.last_usage = {
            "model": self.model_name,
            "elapsed": round(elapsed, 4),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens"),
        }
        logger.debug("LLM 调用完成: %s", self.last_usage)


def exc_safe(exc: Optional[Exception]) -> str:
    """把异常安全地转成字符串（避免 None / 空消息）。"""
    if exc is None:
        return "unknown"
    return str(exc) or type(exc).__name__


# ==================== 工厂 ====================

def resolve_deepseek_base_url() -> Optional[str]:
    """解析 DeepSeek / 兼容网关的 API 基址，读不到返回 ``None``（走官方默认）。

    项目里这个值历史上出现过两个名字：``DEEPSEEK_API_BASE``（环境变量样板的写法）
    与 ``DEEPSEEK_BASE_URL``（早期 VLM 调用处的写法）。两个名字都认，是为了避免
    「用户照着样板配了却没生效」这类静默失效——统一在这里收口。
    """
    for key in ("DEEPSEEK_API_BASE", "DEEPSEEK_BASE_URL"):
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return None


def get_llm_client(model_name: str = "deepseek-flash", json_mode: bool = False,
                   timeout: Optional[float] = None,
                   max_retries: Optional[int] = None) -> Optional[LLMClient]:
    """创建带超时/重试的 DeepSeek 客户端。

    Args:
        model_name: 模型名
        json_mode: 是否强制模型输出 JSON。
            注意: DeepSeek 要求开启 ``json_object`` 时提示词里必须出现 "json" 这个单词，
            否则会直接返回 400。本项目的主体名识别要求"纯净输出"，所以默认关闭。
        timeout: 单次请求超时（秒），默认读 ``LLM_TIMEOUT``
        max_retries: 最大重试次数，默认读 ``LLM_MAX_RETRIES``

    Returns:
        ``LLMClient``；当 ``langchain_deepseek`` 未安装时返回 ``None``，
        调用方据此走模板兜底（离线可跑）。
    """
    try:
        from langchain_deepseek import ChatDeepSeek
    except Exception as e:  # pragma: no cover - 取决于运行环境
        logger.debug("langchain_deepseek 不可用，LLM 功能降级为模板兜底: %s", e)
        return None

    try:
        timeout = float(os.getenv("LLM_TIMEOUT", "60") if timeout is None else timeout)
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "2") if max_retries is None else max_retries)
        backoff = float(os.getenv("LLM_RETRY_BACKOFF", "0.8"))
    except (TypeError, ValueError):
        timeout, max_retries, backoff = 60.0, 2, 0.8

    model_kwargs: dict = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    init_kwargs: dict = {
        "model": model_name,
        "model_kwargs": model_kwargs,
        "timeout": timeout,
        "max_retries": 0,     # 重试由 LLMClient 统一控制，避免与 SDK 内建重试叠加
    }
    base_url = resolve_deepseek_base_url()
    if base_url:
        init_kwargs["base_url"] = base_url

    try:
        client = ChatDeepSeek(**init_kwargs)
    except TypeError:
        # 某些版本的 ``ChatDeepSeek`` 字段名/别名不同，显式传 base_url 会报 TypeError：
        # 退回不带 base_url 构造（SDK 自身仍会读 DEEPSEEK_API_BASE 环境变量），
        # 宁可少一层显式覆盖，也不要因为版本差异让整个 LLM 能力不可用。
        init_kwargs.pop("base_url", None)
        try:
            client = ChatDeepSeek(**init_kwargs)
        except Exception as e:  # pragma: no cover - 取决于运行环境
            logger.warning("创建 DeepSeek 客户端失败，LLM 功能降级为模板兜底: %s", e)
            return None
    except Exception as e:  # pragma: no cover - 取决于运行环境
        logger.warning("创建 DeepSeek 客户端失败，LLM 功能降级为模板兜底: %s", e)
        return None

    return LLMClient(client, model_name=model_name, timeout=timeout,
                     max_retries=max_retries, backoff=backoff)
