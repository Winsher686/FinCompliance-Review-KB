"""轻量级可观测性：trace_id 透传 + 节点耗时打点 + 关键指标记录。

回答的问题是「线上慢了，怎么定位到是检索还是 LLM」——靠的不是日志的绝对时间戳，
而是**同一条 trace_id 下、每个节点各自的耗时瀑布**。

三个能力：

1. **trace_id 透传**：``ensure_trace_id(state)`` 在流程入口生成（优先复用
   ``review_id`` / ``task_id`` / ``session_id``），写入 state 后由 LangGraph 自动带给下游节点；
   节点基类把它注入日志前缀，于是同一次请求的所有日志可被一条 grep 捞出。
2. **节点耗时打点**：``node_span`` 上下文管理器记录每个节点的进入/退出时间、状态
   （ok / error）与业务产出摘要（如召回条数、命中规则数），汇总成瀑布。
3. **关键指标**：``record_metric`` 记录召回数 / 命中规则数 / 检索延迟等，
   提供 ``snapshot_metrics()`` 做均值 / 分位数统计。

设计取舍：
- **默认纯内存**：不写文件、不连库，保证离线测试零副作用、零依赖。
  需要落库时用环境变量开启（见下），且**任何持久化失败都不得影响主流程**。
- 线程安全：用 ``threading.Lock`` 保护内存结构（多路检索会并发打点）。

环境变量：
    OBSERVABILITY_ENABLED   总开关，默认 1
    OBSERVABILITY_MAX_TRACES  内存中保留的 trace 条数上限，默认 200
    OBSERVABILITY_SINK      持久化方式：off(默认) / jsonl / mongo
    OBSERVABILITY_FILE      jsonl 落盘路径（默认 knowledge/data/observability/trace.jsonl）
    OBSERVABILITY_MONGO_URI / OBSERVABILITY_MONGO_DB   mongo 落库目标（缺省复用 MONGO_*）
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "is_enabled", "new_trace_id", "ensure_trace_id", "trace_prefix",
    "node_span", "span_extras", "record_metric", "get_trace", "format_waterfall",
    "snapshot_metrics", "reset_observability",
]

_LOCK = threading.Lock()
_TRACES: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_METRICS: Dict[str, List[float]] = defaultdict(list)

_NODE_CN_CACHE: Optional[Dict[str, str]] = None


# ==================== 基础开关 ====================

def is_enabled() -> bool:
    """可观测性总开关（关闭时所有打点退化为空操作，O(1) 无副作用）。"""
    return os.getenv("OBSERVABILITY_ENABLED", "1") != "0"


def _max_traces() -> int:
    try:
        return max(1, int(os.getenv("OBSERVABILITY_MAX_TRACES", "200")))
    except (TypeError, ValueError):
        return 200


def new_trace_id(prefix: str = "") -> str:
    """生成短 trace_id（8 位十六进制，便于日志肉眼比对）。"""
    token = uuid.uuid4().hex[:12]
    return f"{prefix}{token}" if prefix else token


def ensure_trace_id(state: Optional[Dict[str, Any]], prefix: str = "") -> str:
    """从 state 中取 trace_id；没有则用业务 ID 派生或新生成。

    优先复用 ``trace_id`` → ``review_id`` → ``task_id`` → ``session_id``，
    这样同一笔审查任务的所有轮次可归到同一条 trace 下。
    """
    if isinstance(state, dict):
        existing = state.get("trace_id")
        if existing:
            return str(existing)
        for key, pre in (("review_id", "rv-"), ("task_id", "tk-"), ("session_id", "ss-")):
            value = state.get(key)
            if value:
                return f"{pre}{value}" if not str(value).startswith(pre) else str(value)
    return new_trace_id(prefix)


def trace_prefix(trace_id: str) -> str:
    """日志前缀。"""
    return f"[trace={trace_id}] " if trace_id else ""


def _node_cn(node_name: str) -> str:
    """节点中文名（复用 task_util 的映射，避免两处维护）。"""
    global _NODE_CN_CACHE
    if _NODE_CN_CACHE is None:
        try:
            from knowledge.utils.task_util import _NODE_NAME_TO_CN
            _NODE_CN_CACHE = dict(_NODE_NAME_TO_CN)
        except Exception:  # pragma: no cover
            _NODE_CN_CACHE = {}
    return _NODE_CN_CACHE.get(node_name, node_name)


# ==================== 打点 ====================

@contextmanager
def node_span(trace_id: str, node_name: str,
              extras: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
    """记录一个节点的耗时与结果。

    用法::

        with node_span(trace_id, self.name) as span:
            result = do_work()
            span["extras"]["docs"] = len(result)

    异常不会在打点逻辑里被吞掉，只把 ``status`` 记为 error 后继续向上抛。
    """
    if not is_enabled() or not trace_id:
        yield {"extras": {}}
        return

    started = time.monotonic()
    payload: Dict[str, Any] = {"extras": dict(extras or {})}
    status = "ok"
    try:
        yield payload
    except BaseException as exc:  # 包含 LangGraph 控制流异常（interrupt）
        status = "interrupted" if type(exc).__module__.startswith("langgraph") else "error"
        payload["error"] = f"{type(exc).__name__}: {exc}"[:300]
        raise
    finally:
        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        _append_span(trace_id, {
            "node": node_name,
            "node_cn": _node_cn(node_name),
            "duration_ms": elapsed_ms,
            "status": status,
            "extras": payload.get("extras") or {},
            "error": payload.get("error"),
            "ts": time.time(),
        })
        record_metric("node_duration_ms", elapsed_ms, {"node": node_name})
        if status != "ok":
            record_metric(f"node_{status}_total", 1, {"node": node_name})


def _append_span(trace_id: str, span: Dict[str, Any]) -> None:
    with _LOCK:
        trace = _TRACES.get(trace_id)
        if trace is None:
            trace = {"trace_id": trace_id, "started_at": span["ts"], "spans": []}
            _TRACES[trace_id] = trace
            while len(_TRACES) > _max_traces():
                _TRACES.popitem(last=False)
        trace["spans"].append(span)
        trace["total_ms"] = round(trace.get("total_ms", 0.0) + span["duration_ms"], 2)
        trace["updated_at"] = span["ts"]
    _persist({"type": "span", "trace_id": trace_id, **span}, span["ts"])


def record_metric(name: str, value: float, tags: Optional[Dict[str, Any]] = None) -> None:
    """记录一个数值型指标（内存聚合 + 可选落盘）。"""
    if not is_enabled() or not name:
        return
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return
    with _LOCK:
        _METRICS[name].append(numeric)
        if len(_METRICS[name]) > 5000:      # 防止长跑内存膨胀
            del _METRICS[name][:1000]
    _persist({"type": "metric", "name": name, "value": numeric, "tags": tags or {}}, time.time())


# 节点返回值 → 打点摘要的字段映射（召回条数 / 命中规则数 / 风险等级 …）
_SPAN_COUNT_FIELDS = (
    ("embedding_chunks", "vec"),
    ("hyde_embedding_chunks", "hyde"),
    ("kg_chunks", "kg"),
    ("kg_triples", "triples"),
    ("kg_paths", "paths"),
    ("sparse_chunks", "sparse"),
    ("rrf_chunks", "rrf"),
    ("reranked_docs", "docs"),
    ("evidence_chain", "evidence"),
    ("findings", "findings"),
    ("chunks", "chunks"),
)

_SPAN_SCALAR_FIELDS = (
    ("intent", "intent"),
    ("risk_level", "risk"),
    ("review_status", "status"),
)

#: 字典型字段 → 压缩成 `vec:420|sparse:180|kg:910` 塞进瀑布旁注（多路检索的分支耗时）
_SPAN_DICT_FIELDS = (
    ("search_branch_ms", "branches"),
)

#: 毫秒型标量字段 → 取整后展示
_SPAN_MS_FIELDS = (
    ("search_wall_ms", "wall_ms"),
)


def _compact_branch_ms(value: Any) -> str:
    """把 {分支: 毫秒} 压成 `vec:420|sparse:180|kg:910`（按耗时降序，肉眼先看到最慢的）。"""
    if not isinstance(value, dict) or not value:
        return ""
    items = []
    for key, cost in value.items():
        try:
            items.append((str(key), float(cost)))
        except (TypeError, ValueError):
            continue
    items.sort(key=lambda kv: kv[1], reverse=True)
    return "|".join(f"{k}:{int(v)}" for k, v in items)


def span_extras(result: Any) -> Dict[str, Any]:
    """从节点返回值里抽取可观测摘要（条数计数 + 关键枚举 + 分支耗时），用于耗时瀑布旁注。"""
    if not isinstance(result, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, label in _SPAN_COUNT_FIELDS:
        value = result.get(key)
        if isinstance(value, (list, tuple)):
            out[label] = len(value)
    for key, label in _SPAN_SCALAR_FIELDS:
        value = result.get(key)
        if value:
            out[label] = value
    for key, label in _SPAN_DICT_FIELDS:
        compact = _compact_branch_ms(result.get(key))
        if compact:
            out[label] = compact
    for key, label in _SPAN_MS_FIELDS:
        value = result.get(key)
        try:
            if value:
                out[label] = int(float(value))
        except (TypeError, ValueError):
            continue
    # 关键指标落库：召回条数与命中规则数是排障时最常看的两项
    if "docs" in out:
        record_metric("retrieval_docs", out["docs"])
    if "findings" in out:
        record_metric("rule_findings", out["findings"])
    return out


# ==================== 读取 / 展示 ====================

def get_trace(trace_id: str) -> Dict[str, Any]:
    """取出一条 trace 的完整结构（含节点耗时瀑布）。"""
    with _LOCK:
        trace = _TRACES.get(trace_id)
        if trace is None:
            return {"trace_id": trace_id, "spans": [], "total_ms": 0.0}
        return json.loads(json.dumps(trace))   # 深拷贝，避免调用方改动内存结构


def format_waterfall(trace_id: str, width: int = 28) -> str:
    """把一条 trace 渲染成人类可读的耗时瀑布（用于日志 / 排障 / 报告附录）。"""
    trace = get_trace(trace_id)
    spans = trace.get("spans") or []
    if not spans:
        return f"[trace={trace_id}] 无打点数据"

    total = trace.get("total_ms") or sum(s["duration_ms"] for s in spans) or 1.0
    peak = max(s["duration_ms"] for s in spans) or 1.0
    lines = [f"[trace={trace_id}] 节点耗时瀑布（总计 {total:.0f} ms）"]
    for idx, span in enumerate(spans, 1):
        ratio = span["duration_ms"] / peak
        bar = "█" * max(1, int(ratio * width))
        extras = span.get("extras") or {}
        extra_text = ("  " + " ".join(f"{k}={v}" for k, v in extras.items())) if extras else ""
        flag = "" if span["status"] == "ok" else f" ✅{span['status']}"
        lines.append(
            f"  {idx:>2}. {span['node_cn']:<12} {bar:<{width}} "
            f"{span['duration_ms']:>8.1f} ms ({span['duration_ms'] / total:>5.1%}){extra_text}{flag}"
        )
    return "\n".join(lines)


def snapshot_metrics() -> Dict[str, Dict[str, float]]:
    """指标快照：每个指标给出 count / mean / p50 / p95 / max。"""
    with _LOCK:
        data = {name: list(values) for name, values in _METRICS.items()}
    out: Dict[str, Dict[str, float]] = {}
    for name, values in data.items():
        if not values:
            continue
        ordered = sorted(values)
        def _pct(p: float) -> float:
            idx = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
            return round(ordered[idx], 3)
        out[name] = {
            "count": len(ordered),
            "mean": round(statistics.fmean(ordered), 3),
            "p50": _pct(0.50),
            "p95": _pct(0.95),
            "max": round(ordered[-1], 3),
        }
    return out


def reset_observability() -> None:
    """清空内存中的 trace 与指标（测试用）。"""
    with _LOCK:
        _TRACES.clear()
        _METRICS.clear()


# ==================== 可选持久化（失败一律降级为日志） ====================

_MONGO_SINK_WARNED = False


def _persist(event: Dict[str, Any], ts: float) -> None:
    """按需把打点事件落盘/落库；任何异常都只记日志，不影响主流程。"""
    sink = (os.getenv("OBSERVABILITY_SINK", "off") or "off").strip().lower()
    if sink in ("", "off", "none"):
        return
    try:
        if sink == "jsonl":
            _persist_jsonl(event, ts)
        elif sink == "mongo":
            _persist_mongo(event, ts)
        else:
            logger.debug("未知 OBSERVABILITY_SINK=%s，已忽略", sink)
    except Exception as e:  # pragma: no cover - 取决于运行环境
        logger.debug("可观测性持久化失败（已忽略）: %s", e)


def _default_jsonl_path() -> str:
    path = os.getenv("OBSERVABILITY_FILE", "").strip()
    if path:
        return path
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "observability"))
    return os.path.join(root, "trace.jsonl")


def _persist_jsonl(event: Dict[str, Any], ts: float) -> None:
    path = _default_jsonl_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = {"ts": ts, **event}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _persist_mongo(event: Dict[str, Any], ts: float) -> None:
    global _MONGO_SINK_WARNED
    uri = os.getenv("OBSERVABILITY_MONGO_URI") or os.getenv("MONGO_URL", "")
    db_name = os.getenv("OBSERVABILITY_MONGO_DB") or os.getenv("MONGO_DB_NAME", "")
    if not uri or not db_name:
        if not _MONGO_SINK_WARNED:
            logger.warning("OBSERVABILITY_SINK=mongo 但未配置 MONGO_URL / MONGO_DB_NAME，已跳过落库")
            _MONGO_SINK_WARNED = True
        return
    from pymongo import MongoClient
    client = MongoClient(uri, serverSelectionTimeoutMS=1500)
    client[db_name]["observability_events"].insert_one({"ts": ts, **event})
