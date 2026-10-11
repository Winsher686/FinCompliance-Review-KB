"""LLM 客户端与输出解析测试（knowledge/utils/llm_client.py）。

覆盖 L3-1「LLM 客户端工程化」：
- 解析层：代码围栏剥离、括号配平截取、解析失败返回默认值（不抛异常）；
- 客户端层：指数退避重试、非可重试异常立即失败、超时归一为 LLMTimeoutError、
  字符串提示词不被误展开为逐字符、流式调用异常归一、token 估算。

全部离线可跑：用假客户端注入，不需要 langchain_deepseek / 网络。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from knowledge.utils.llm_client import (
    LLMClient, LLMTimeoutError, LLMUnavailableError,
    strip_code_fence, extract_json_block, parse_json_response, estimate_tokens,
)


# ==================== 解析层 ====================

def test_strip_code_fence_variants():
    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fence('```\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fence('  {"a": 1}  ') == '{"a": 1}'
    assert strip_code_fence("") == ""
    assert strip_code_fence(None) == ""


def test_extract_json_block_tolerates_surrounding_text():
    text = '好的，结果如下：\n{"intent": "compliance", "note": "含 { 括号 } 与 \\"引号\\""}\n以上。'
    block = extract_json_block(text)
    assert block.startswith("{") and block.endswith("}")
    assert parse_json_response(text) == {"intent": "compliance", "note": '含 { 括号 } 与 "引号"'}


def test_extract_json_block_array():
    assert extract_json_block('[1, 2, 3] 完毕') == "[1, 2, 3]"
    assert extract_json_block("没有 JSON") == ""


def test_parse_json_response_returns_default_on_garbage():
    assert parse_json_response("这不是 JSON") is None
    assert parse_json_response("这不是 JSON", default={}) == {}
    assert parse_json_response("", default=[]) == []
    assert parse_json_response(None, default="x") == "x"


def test_parse_json_response_accepts_plain_and_fenced():
    assert parse_json_response('{"entities": ["A"]}') == {"entities": ["A"]}
    assert parse_json_response('```json\n{"entities": ["A"]}\n```') == {"entities": ["A"]}


def test_estimate_tokens_monotonic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("商家合规") >= 4
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


# ==================== 客户端层（注入假客户端） ====================

class _FakeResponse:
    def __init__(self, content="ok"):
        self.content = content
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


class _FlakyClient:
    """前 N 次抛可重试异常，之后成功。"""

    def __init__(self, fail_times=1, exc=None):
        self.fail_times = fail_times
        self.calls = 0
        self.last_payload = None
        self._exc = exc or RuntimeError("Connection reset by peer")

    def invoke(self, payload, **kwargs):
        self.calls += 1
        self.last_payload = payload
        if self.calls <= self.fail_times:
            raise self._exc
        return _FakeResponse()


class _BrokenClient:
    def __init__(self, exc):
        self.calls = 0
        self._exc = exc

    def invoke(self, payload, **kwargs):
        self.calls += 1
        raise self._exc

    def stream(self, payload, **kwargs):
        raise self._exc


def _client(inner, **kwargs):
    params = {"max_retries": 2, "backoff": 0.0}
    params.update(kwargs)
    return LLMClient(inner, model_name="fake", **params)


def test_retry_then_success_and_usage_recorded():
    inner = _FlakyClient(fail_times=2)
    client = _client(inner)
    resp = client.invoke([{"role": "user"}])
    assert resp.content == "ok"
    assert inner.calls == 3                       # 首次 + 2 次重试
    assert client.last_usage["total_tokens"] == 15


def test_non_transient_error_fails_fast():
    """鉴权/参数类错误不应重试（否则只是浪费配额与延迟）。"""
    inner = _BrokenClient(RuntimeError("400 invalid api key"))
    client = _client(inner)
    with pytest.raises(LLMUnavailableError):
        client.invoke([{"role": "user"}])
    assert inner.calls == 1


def test_timeout_is_normalized():
    inner = _BrokenClient(RuntimeError("Request timed out"))
    client = _client(inner, max_retries=1)
    with pytest.raises(LLMTimeoutError):
        client.invoke([{"role": "user"}])
    assert inner.calls == 2


def test_string_prompt_is_not_expanded_to_chars():
    """回归防线：str 提示词必须原样透传，不能 list(str) 变成逐字符输入。"""
    inner = _FlakyClient(fail_times=0)
    client = _client(inner)
    client.invoke("请输出 JSON")
    assert inner.last_payload == "请输出 JSON"


def test_invoke_text_and_json_helpers():
    class _JsonClient:
        def invoke(self, payload, **kwargs):
            return _FakeResponse('```json\n{"intent": "fact"}\n```')

    client = _client(_JsonClient())
    assert client.invoke_text("x") == '```json\n{"intent": "fact"}\n```'
    assert client.invoke_json("x") == {"intent": "fact"}
    assert client.invoke_json("x", default={}) == {"intent": "fact"}


def test_stream_error_normalized():
    inner = _BrokenClient(RuntimeError("502 bad gateway"))
    client = _client(inner)
    with pytest.raises(LLMUnavailableError):
        client.stream("x")


def test_get_llm_client_returns_none_without_dependency():
    """依赖缺失时必须返回 None（调用方据此走模板兜底），而不是抛异常。"""
    from knowledge.utils import llm_client as mod
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "langchain_deepseek":
            raise ImportError("no langchain_deepseek")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = _fake_import
    try:
        assert mod.get_llm_client() is None
    finally:
        builtins.__import__ = real_import
