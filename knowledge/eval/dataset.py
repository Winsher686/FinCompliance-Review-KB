"""评测集加载、相对日期解析与指标计算（纯 Python，离线可跑）。

**为什么需要相对日期**：评测集里会出现「处罚日期」「资质有效期」这类日期。
若在 JSON 里写死绝对日期，数据集会随时间腐烂——今天是「近三年内的处罚」，
两年后就变成「五年前的旧案」，R3 不再命中，评测分数会莫名下降。
所以约定在数据里写**相对日期 token**，读取时再落到绝对日期：

    @today       今天          @-30d     30 天前
    @-2y         两年前        @+365d    一年后

`load_samples(today=...)` 支持注入基准日，因此「日期边界」这类用例可以稳定复现。

**指标定义**（与 `run_eval.py` 输出一致）：

- ``intent_accuracy``    意图识别准确率（离线用关键词兜底；在线由 LLM 决定）
- ``rule_hit_f1``        规则命中集合的 micro-F1（对每条样本取交集/并集后汇总）
- ``risk_level_accuracy`` 整体风险等级完全一致的比例
- ``hitl_accuracy``      HITL 触发判定与期望一致的比例
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Set

from knowledge.processor.review_process.config import get_config
from knowledge.processor.review_process.nodes.intent_recognize_node import IntentRecognizeNode
from knowledge.processor.review_process.rules.compliance_rules import (
    build_context, evaluate_rules, overall_risk_level,
)

#: 相对日期 token：@today / @-30d / @+365d / @-2y
_TOKEN_RE = re.compile(r"^@(?:(today)|([+-]?\d+)([dy]))$")

_DEFAULT_QA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qa_pairs.jsonl")

ALL_RULES = ("R1", "R2", "R3", "R4", "R5", "R6")


# ==================== 路径与加载 ====================

def dataset_path() -> str:
    """评测集路径（可用 ``EVAL_QA_FILE`` 覆盖，便于换数据集跑同一套指标）。"""
    return os.getenv("EVAL_QA_FILE", "").strip() or _DEFAULT_QA_FILE


def resolve_date_token(value: Any, today: Optional[date] = None) -> Any:
    """把单个相对日期 token 解析为 ``YYYY-MM-DD``；非 token 原样返回。"""
    if not isinstance(value, str):
        return value
    match = _TOKEN_RE.match(value.strip())
    if not match:
        return value
    base = today or date.today()
    if match.group(1):                      # @today
        return base.isoformat()
    amount = int(match.group(2))
    unit = match.group(3)
    delta = timedelta(days=amount * 365) if unit == "y" else timedelta(days=amount)
    return (base + delta).isoformat()


def resolve_dates(obj: Any, today: Optional[date] = None) -> Any:
    """递归解析嵌套结构里的相对日期 token（dict / list / str）。"""
    if isinstance(obj, dict):
        return {k: resolve_dates(v, today) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_dates(v, today) for v in obj]
    return resolve_date_token(obj, today)


def load_samples(path: Optional[str] = None, today: Optional[date] = None) -> List[Dict[str, Any]]:
    """读取评测集（JSONL），并把相对日期 token 解析为绝对日期。"""
    file_path = path or dataset_path()
    samples: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(resolve_dates(json.loads(line), today))
    return samples


# ==================== 单条样本评估 ====================

def expected_hitl(review_type: str, risk_level: str) -> bool:
    """按 `compliance_check_node._needs_human_review` 的口径计算期望 HITL 结果。"""
    cfg = get_config()
    if not cfg.hitl_enabled:
        return False
    if risk_level in cfg.hitl_trigger_risk_levels():
        return True
    return (review_type or "general").strip() in cfg.hitl_trigger_review_types()


def predict_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """对单条样本跑「确定性部分」：规则命中 + 风险等级 + HITL + 关键词兜底意图。"""
    ctx = build_context(
        company=sample.get("company", ""),
        relations=sample.get("relations") or [],
        graph_paths=sample.get("graph_paths") or [],
        documents=sample.get("documents") or [],
        metrics=sample.get("metrics") or {},
    )
    findings = evaluate_rules(ctx)
    risk = overall_risk_level(findings)
    review_type = sample.get("review_type") or "general"
    return {
        "rules": {f.rule_id for f in findings},
        "risk_level": risk,
        "needs_human_review": expected_hitl(review_type, risk),
        "intent": IntentRecognizeNode._rule_intent(sample.get("question", ""), review_type),
        "findings": [f.to_dict() for f in findings],
    }


# ==================== 指标 ====================

@dataclass
class EvalReport:
    """评测结果汇总。"""

    total: int = 0
    intent_accuracy: float = 0.0
    rule_hit_precision: float = 0.0
    rule_hit_recall: float = 0.0
    rule_hit_f1: float = 0.0
    risk_level_accuracy: float = 0.0
    hitl_accuracy: float = 0.0
    rule_recall_by_id: Dict[str, float] = field(default_factory=dict)
    #: 意图混淆：期望意图 → {兜底预测意图: 次数}（只统计不一致的格子）
    intent_confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    #: 期望有规则的样本中「一条都没命中」的（召回漏检）
    missed_samples: List[str] = field(default_factory=list)
    #: 期望无规则但命中了规则的（误报）
    false_positive_samples: List[str] = field(default_factory=list)
    mismatches: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "intent_accuracy": round(self.intent_accuracy, 4),
            "rule_hit_precision": round(self.rule_hit_precision, 4),
            "rule_hit_recall": round(self.rule_hit_recall, 4),
            "rule_hit_f1": round(self.rule_hit_f1, 4),
            "risk_level_accuracy": round(self.risk_level_accuracy, 4),
            "hitl_accuracy": round(self.hitl_accuracy, 4),
            "rule_recall_by_id": {k: round(v, 4) for k, v in self.rule_recall_by_id.items()},
            "intent_confusion": self.intent_confusion,
            "missed_samples": self.missed_samples,
            "false_positive_samples": self.false_positive_samples,
            "mismatches": self.mismatches,
        }

    def to_markdown(self) -> str:
        rule_bits = " ".join(f"{k} {v:.0%}" for k, v in sorted(self.rule_recall_by_id.items()))
        confusion_bits = "；".join(
            f"{exp}→{pred} {cnt} 例"
            for exp, preds in sorted(self.intent_confusion.items())
            for pred, cnt in sorted(preds.items(), key=lambda kv: -kv[1])
        )
        lines = [
            f"- 样本数：**{self.total}**",
            f"- `intent_accuracy`（关键词兜底）：**{self.intent_accuracy:.1%}**",
            f"- `rule_hit_f1`（micro）：**{self.rule_hit_f1:.1%}**"
            f"（P {self.rule_hit_precision:.1%} / R {self.rule_hit_recall:.1%}）",
            f"- `risk_level_accuracy`：**{self.risk_level_accuracy:.1%}**",
            f"- `hitl_accuracy`：**{self.hitl_accuracy:.1%}**",
            f"- 分规则召回：{rule_bits}",
        ]
        if confusion_bits:
            lines.append(f"- 意图兜底主要偏差：{confusion_bits}")
        return "\n".join(lines)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_dataset(samples: Optional[List[Dict[str, Any]]] = None,
                     today: Optional[date] = None) -> EvalReport:
    """对全量样本计算四项指标（可离线运行）。"""
    samples = samples if samples is not None else load_samples(today=today)
    report = EvalReport(total=len(samples))
    if not samples:
        return report

    tp = fp = fn = 0
    intent_ok = risk_ok = hitl_ok = 0
    expected_rule_total: Dict[str, int] = {r: 0 for r in ALL_RULES}
    hit_expected: Dict[str, int] = {r: 0 for r in ALL_RULES}

    for sample in samples:
        expect = sample.get("expect") or {}
        pred = predict_sample(sample)
        sid = sample.get("id", "?")

        exp_rules: Set[str] = set(expect.get("rules") or [])
        hit_rules: Set[str] = pred["rules"]
        tp += len(exp_rules & hit_rules)
        fp += len(hit_rules - exp_rules)
        fn += len(exp_rules - hit_rules)

        for rule in exp_rules:
            expected_rule_total[rule] = expected_rule_total.get(rule, 0) + 1
            if rule in hit_rules:
                hit_expected[rule] = hit_expected.get(rule, 0) + 1

        if hit_rules != exp_rules:
            report.mismatches.append(
                f"{sid} 规则命中不符：实际={sorted(hit_rules)} 期望={sorted(exp_rules)}"
            )
        if exp_rules and not hit_rules:
            report.missed_samples.append(sid)
        if not exp_rules and hit_rules:
            report.false_positive_samples.append(sid)

        pred_intent = pred["intent"]
        exp_intent = expect.get("intent")
        if pred_intent == exp_intent:
            intent_ok += 1
        else:
            bucket = report.intent_confusion.setdefault(str(exp_intent), {})
            bucket[pred_intent] = bucket.get(pred_intent, 0) + 1
        if pred["risk_level"] == expect.get("risk_level"):
            risk_ok += 1
        else:
            report.mismatches.append(
                f"{sid} 风险等级不符：实际={pred['risk_level']} 期望={expect.get('risk_level')}"
            )
        if pred["needs_human_review"] == bool(expect.get("hitl")):
            hitl_ok += 1
        else:
            report.mismatches.append(
                f"{sid} HITL 判定不符：实际={pred['needs_human_review']} 期望={expect.get('hitl')}"
            )

    report.intent_accuracy = _ratio(intent_ok, len(samples))
    report.risk_level_accuracy = _ratio(risk_ok, len(samples))
    report.hitl_accuracy = _ratio(hitl_ok, len(samples))
    report.rule_hit_precision = _ratio(tp, tp + fp)
    report.rule_hit_recall = _ratio(tp, tp + fn)
    denom = report.rule_hit_precision + report.rule_hit_recall
    report.rule_hit_f1 = (2 * report.rule_hit_precision * report.rule_hit_recall / denom) if denom else 0.0
    report.rule_recall_by_id = {
        rule: _ratio(hit_expected.get(rule, 0), total)
        for rule, total in expected_rule_total.items() if total
    }
    return report


def dataset_meta(samples: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """由样本重算 evaluation.json 里的统计块（保证元信息与数据永不漂移）。"""
    samples = samples if samples is not None else load_samples()
    by_review_type: Dict[str, int] = {}
    by_intent: Dict[str, int] = {}
    expected_rules: Dict[str, int] = {}
    expected_risk: Dict[str, int] = {}
    hitl_ids: List[str] = []
    for s in samples:
        expect = s.get("expect") or {}
        by_review_type[s["review_type"]] = by_review_type.get(s["review_type"], 0) + 1
        by_intent[expect.get("intent", "")] = by_intent.get(expect.get("intent", ""), 0) + 1
        for rule in expect.get("rules") or []:
            expected_rules[rule] = expected_rules.get(rule, 0) + 1
        expected_risk[expect.get("risk_level", "")] = expected_risk.get(expect.get("risk_level", ""), 0) + 1
        if expect.get("hitl"):
            hitl_ids.append(s["id"])
    return {
        "total": len(samples),
        "by_review_type": by_review_type,
        "by_intent": by_intent,
        "expected_rules": expected_rules,
        "expected_risk_levels": expected_risk,
        "expected_hitl_ids": hitl_ids,
    }
