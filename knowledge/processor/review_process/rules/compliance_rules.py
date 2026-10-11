"""电商商品/商家合规规则库（内置规则引擎）。

设计目标：
1. 纯 Python、无三方依赖，可离线单元测试（不依赖 Milvus/Neo4j/LLM）。
2. 规则与数据解耦：规则只读取「结构化审查上下文 ReviewContext」，
   上下文由图谱关系、图谱路径、文档片段、经营指标装配而成。
3. 阈值全部来自环境变量，便于按平台治理口径调整。

内置规则（可扩展）：
    R1 违禁词/极限词宣传   商品宣传命中违禁词 / 极限词
    R2 资质缺失或过期      受管控类目所需资质缺失或已过期
    R3 违规未整改          近 N 年存在违规处罚但未见整改
    R4 问题货源            上游供应商为被投诉主体
    R5 类目错放与价格违规  类目错放，或促销折扣率异常（虚构原价嫌疑）
    R6 品牌授权链穿透      存在多层级品牌授权链（信息提示）
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Set

# 严重度排序（数值越大越严重）
SEVERITY_ORDER: Dict[str, int] = {"高": 3, "中": 2, "低": 1, "无": 0}


# ==================== 数值 / 日期解析 ====================

_UNIT = {"亿": 1e8, "万": 1e4, "千": 1e3, "元": 1.0}

_TRUE_WORDS = {"1", "true", "yes", "y", "是", "真", "有"}


def parse_amount(value: Any) -> Optional[float]:
    """把「1.2亿元 / 5,000万 / 3000000」解析为以元为单位的浮点数。

    解析失败返回 None（规则会跳过该条，不影响其它证据）。
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("，", "")
    if not text:
        return None
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*(亿|万|千|元)?", text)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2) or "元"
    return number * _UNIT.get(unit, 1.0)


def _parse_ratio(value: Any) -> Optional[float]:
    """把「70%」「0.7」「70」统一解析为 0~1 的比例，失败返回 None。"""
    if value is None:
        return None
    text = str(value).strip().replace(" ", "")
    if not text:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    number = float(match.group(1))
    if "%" in text or number > 1:
        number = number / 100.0
    return number


def _parse_date(value: Any) -> Optional[date]:
    """解析「2025-12-31」「2025/12/31」「2025年12月31日」为 date，失败返回 None。"""
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"(\d{4})\s*[-/年.]\s*(\d{1,2})(?:\s*[-/月.]\s*(\d{1,2}))?", text)
    if not match:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    day = int(match.group(3)) if match.group(3) else 1
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _is_true(value: Any) -> bool:
    """宽松识别真值（用于 is_fake_price / is_misplaced 这类标记字段）。"""
    return str(value).strip().lower() in _TRUE_WORDS


# ==================== 数据结构 ====================

@dataclass
class ReviewContext:
    """审查上下文：规则引擎的唯一输入。"""

    company: str = ""                                                # 主体（商家/店铺）名
    relations: List[Dict[str, Any]] = field(default_factory=list)    # [{head,rel,tail,props}]
    graph_paths: List[Dict[str, Any]] = field(default_factory=list)  # 图谱多跳路径
    documents: List[Dict[str, Any]] = field(default_factory=list)    # 命中的文档片段
    metrics: Dict[str, float] = field(default_factory=dict)          # {gmv: ..., listing_count: ...}
    disclosed: Set[str] = field(default_factory=set)                 # 已整改/已处理事项关键字

    def relations_of(self, rel_type: str) -> List[Dict[str, Any]]:
        """获取指定类型的全部关系。"""
        return [r for r in self.relations
                if (r.get("rel") or r.get("type")) == rel_type]

    def has_disclosure(self, *keywords: str) -> bool:
        """判断是否已就某事项整改/处理（关键字命中即视为已处理）。"""
        if not keywords:
            return False
        joined = " ".join(self.disclosed)
        return any(kw and kw in joined for kw in keywords)


@dataclass
class ComplianceFinding:
    """单条合规判定结果。"""

    rule_id: str
    rule_name: str
    severity: str            # 高/中/低
    triggered: bool
    message: str
    evidence: List[str] = field(default_factory=list)
    suggestion: str = ""
    regulation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "severity": self.severity,
            "triggered": self.triggered,
            "message": self.message,
            "evidence": self.evidence,
            "suggestion": self.suggestion,
            "regulation": self.regulation,
        }


# ==================== 规则基类 ====================

class ComplianceRule:
    """合规规则基类：子类实现 evaluate。"""

    rule_id: str = "R0"
    rule_name: str = "基础规则"
    severity: str = "低"
    regulation: str = ""

    def evaluate(self, ctx: ReviewContext) -> Optional[ComplianceFinding]:
        """返回命中的 finding；未命中返回 None。"""
        raise NotImplementedError


# ==================== 内置规则 ====================

class BannedWordPromotionRule(ComplianceRule):
    """R1：商品宣传命中违禁词 / 极限词。"""

    rule_id = "R1"
    rule_name = "违禁词/极限词宣传"
    severity = "高"
    regulation = "《广告法》《反不正当竞争法》及平台商品宣传规范"

    def evaluate(self, ctx: ReviewContext) -> Optional[ComplianceFinding]:
        threshold = int(os.getenv("COMPLIANCE_BANNED_WORD_HIT", "1"))
        hits: List[str] = []
        words: List[str] = []
        for rel in ctx.relations_of("PROMOTES"):
            props = rel.get("props") or {}
            raw = str(props.get("banned_words") or "").strip()
            if not raw:
                continue
            for word in re.split(r"[,，、;；\s]+", raw):
                word = word.strip()
                if word and word not in words:
                    words.append(word)
            hits.append(f"{rel.get('head')} -[PROMOTES]-> {rel.get('tail')} 宣传用语={raw}")
        if len(words) < max(threshold, 1):
            return None
        return ComplianceFinding(
            rule_id=self.rule_id, rule_name=self.rule_name, severity=self.severity,
            triggered=True,
            message=(f"商品宣传中共命中 {len(words)} 个违禁词/极限词（{'、'.join(words)}），"
                     f"达到阈值 {max(threshold, 1)}，涉嫌违反广告宣传规范。"),
            evidence=hits,
            suggestion="下架或整改相关宣传文案，删除违禁词/极限词并留痕复查。",
            regulation=self.regulation,
        )


class QualificationRule(ComplianceRule):
    """R2：受管控类目所需资质缺失或已过期。"""

    rule_id = "R2"
    rule_name = "资质缺失或过期"
    severity = "中"
    regulation = "《电子商务法》及平台类目资质准入规范"

    def evaluate(self, ctx: ReviewContext) -> Optional[ComplianceFinding]:
        ahead_days = int(os.getenv("COMPLIANCE_CERT_EXPIRE_DAYS", "0"))
        deadline = date.today() + timedelta(days=ahead_days)
        problems: List[str] = []

        for rel in ctx.relations_of("HOLDS_CERT"):
            props = rel.get("props") or {}
            status = str(props.get("status") or "").strip()
            cert = rel.get("tail") or props.get("cert_type") or "所需资质"
            if status in ("缺失", "missing", "无", "未取得"):
                problems.append(f"{rel.get('head')} 缺失资质：{cert}（状态={status}）")
                continue
            expire = _parse_date(props.get("expire_date"))
            if expire is not None and expire < deadline:
                problems.append(
                    f"{rel.get('head')} 资质已过期：{cert}（有效期至 {props.get('expire_date')}）"
                )

        if not problems:
            return None
        return ComplianceFinding(
            rule_id=self.rule_id, rule_name=self.rule_name, severity=self.severity,
            triggered=True,
            message=f"存在 {len(problems)} 项资质缺失或过期情形，不符合类目准入要求。",
            evidence=problems,
            suggestion="要求商家补齐/更新资质，未补齐前限制对应类目商品销售。",
            regulation=self.regulation,
        )


class ViolationRectifyRule(ComplianceRule):
    """R3：近 N 年存在违规处罚但未见整改。"""

    rule_id = "R3"
    rule_name = "违规未整改"
    severity = "高"
    regulation = "平台违规治理规则与《电子商务法》相关要求"

    def evaluate(self, ctx: ReviewContext) -> Optional[ComplianceFinding]:
        years = int(os.getenv("COMPLIANCE_PENALTY_YEARS", "3"))
        sanctions = ctx.relations_of("SANCTIONED_BY")
        if not sanctions:
            return None
        if ctx.has_disclosure("已整改", "整改完成", "已处理"):
            return None

        cutoff = date.today() - timedelta(days=365 * years)
        recent: List[Dict[str, Any]] = []
        for rel in sanctions:
            status = str((rel.get("props") or {}).get("status") or "").strip()
            if status in ("已整改", "整改完成", "已处理"):
                continue
            happened = _parse_date((rel.get("props") or {}).get("date"))
            if happened is not None and happened < cutoff:
                continue
            recent.append(rel)

        if not recent:
            return None
        total = sum(
            parse_amount((rel.get("props") or {}).get("amount")) or 0.0 for rel in recent
        )
        return ComplianceFinding(
            rule_id=self.rule_id, rule_name=self.rule_name, severity=self.severity,
            triggered=True,
            message=(f"近 {years} 年存在 {len(recent)} 条违规处罚记录且未见整改，"
                     f"累计处罚金额 {total:,.0f} 元，存在重复违规风险。"),
            evidence=[f"{r.get('head')} -[SANCTIONED_BY]-> {r.get('tail')} "
                      f"事由={((r.get('props') or {}).get('reason'))} "
                      f"金额={((r.get('props') or {}).get('amount'))} "
                      f"状态={((r.get('props') or {}).get('status')) or '未标注'}"
                      for r in recent],
            suggestion="要求商家提交整改报告并复核；必要时降权、限流或暂停经营。",
            regulation=self.regulation,
        )


class SupplierComplaintRule(ComplianceRule):
    """R4：上游供应商中存在被投诉主体。"""

    rule_id = "R4"
    rule_name = "问题货源"
    severity = "中"
    regulation = "平台供应链质量管理与消费者权益保护要求"

    def evaluate(self, ctx: ReviewContext) -> Optional[ComplianceFinding]:
        complained = {(r.get("head") or "").strip()
                      for r in ctx.relations_of("COMPLAINED_IN") if r.get("head")}
        if not complained:
            return None
        hits = []
        for rel in ctx.relations_of("SUPPLIES"):
            for party in (rel.get("head"), rel.get("tail")):
                if party and party.strip() in complained:
                    hits.append(f"{rel.get('head')} -[SUPPLIES]-> {rel.get('tail')}"
                                f"（{party} 存在投诉记录）")
        if not hits:
            return None
        return ComplianceFinding(
            rule_id=self.rule_id, rule_name=self.rule_name, severity=self.severity,
            triggered=True,
            message="上游货源主体中存在被投诉主体，可能影响商品质量与售后稳定性。",
            evidence=hits,
            suggestion="复核该供应商的投诉与质检记录，必要时更换货源或加严抽检。",
            regulation=self.regulation,
        )


class PriceAndCategoryRule(ComplianceRule):
    """R5：类目错放，或促销折扣率异常（虚构原价嫌疑）。"""

    rule_id = "R5"
    rule_name = "类目错放与价格违规"
    severity = "中"
    regulation = "《价格法》《明码标价和禁止价格欺诈规定》及平台类目规范"

    def evaluate(self, ctx: ReviewContext) -> Optional[ComplianceFinding]:
        ratio_line = float(os.getenv("COMPLIANCE_FAKE_PRICE_RATIO", "0.5"))
        problems: List[str] = []

        for rel in ctx.relations_of("BELONGS_TO"):
            props = rel.get("props") or {}
            if _is_true(props.get("is_misplaced")):
                problems.append(
                    f"{rel.get('head')} 类目错放：{rel.get('tail')}"
                    f"（类目代码={props.get('category_code')}）"
                )

        for rel in ctx.relations_of("PROMOTES"):
            props = rel.get("props") or {}
            ratio = _parse_ratio(props.get("discount_rate"))
            if _is_true(props.get("is_fake_price")) or (ratio is not None and ratio >= ratio_line):
                shown = props.get("discount_rate")
                problems.append(
                    f"{rel.get('head')} -[PROMOTES]-> {rel.get('tail')} "
                    f"折扣率={shown}（阈值 {ratio_line:.0%}），涉嫌虚构原价"
                )

        if not problems:
            return None
        return ComplianceFinding(
            rule_id=self.rule_id, rule_name=self.rule_name, severity=self.severity,
            triggered=True,
            message="存在类目错放或促销价格异常情形，涉嫌违反平台类目与价格规范。",
            evidence=problems,
            suggestion="纠正商品类目并核验原价依据；无依据的折扣宣传应下架整改。",
            regulation=self.regulation,
        )


class BrandAuthorizationRule(ComplianceRule):
    """R6：存在多层级品牌授权链（授权穿透），作为信息提示。"""

    rule_id = "R6"
    rule_name = "品牌授权链穿透"
    severity = "低"
    regulation = "平台品牌授权与知识产权治理规范"

    def evaluate(self, ctx: ReviewContext) -> Optional[ComplianceFinding]:
        max_depth = int(os.getenv("COMPLIANCE_PENETRATION_DEPTH", "2"))
        chain_types = {"AUTHORIZED_BY", "SHARED_BRAND"}

        # 统计授权链的「最大穿透深度」（图谱路径 hop 数为主，关系列表数量为兜底）
        chain_evidence: List[str] = []
        max_reached = 0

        # 1) 图谱路径：hops 即穿透深度；仅统计完全由授权类关系构成的链
        for path in ctx.graph_paths:
            nodes = path.get("nodes") or []
            if len(nodes) < 2:
                continue
            rels = path.get("rels") or []
            if rels and not all(r in chain_types for r in rels):
                continue
            depth = int(path.get("hops") or (len(nodes) - 1))
            chain_evidence.append(" → ".join(str(n) for n in nodes))
            max_reached = max(max_reached, depth)

        # 2) 无图谱路径时回退：以授权型关系的条数近似穿透深度
        if not chain_evidence:
            for rel in ctx.relations:
                rel_type = rel.get("rel") or rel.get("type")
                if rel_type in chain_types:
                    chain_evidence.append(f"{rel.get('head')} -[{rel_type}]-> {rel.get('tail')}")
            max_reached = len(chain_evidence)

        # 未达到最小穿透深度则不提示
        if max_reached < max_depth:
            return None
        return ComplianceFinding(
            rule_id=self.rule_id, rule_name=self.rule_name, severity=self.severity,
            triggered=True,
            message=(f"存在多层级品牌授权链（深度 ≥ {max_depth}），"
                     "可穿透核验授权真实性（结构性提示）。"),
            evidence=chain_evidence[:5],
            suggestion="核验各级品牌授权书与授权有效期，确认授权链完整可追溯。",
            regulation=self.regulation,
        )


# 注册顺序即为执行顺序
BUILTIN_RULES: List[ComplianceRule] = [
    BannedWordPromotionRule(),
    QualificationRule(),
    ViolationRectifyRule(),
    SupplierComplaintRule(),
    PriceAndCategoryRule(),
    BrandAuthorizationRule(),
]


# ==================== 对外入口 ====================

def evaluate_rules(ctx: ReviewContext, rules: Optional[List[ComplianceRule]] = None) -> List[ComplianceFinding]:
    """执行全部规则，返回命中的 findings。"""
    findings: List[ComplianceFinding] = []
    for rule in (rules or BUILTIN_RULES):
        try:
            finding = rule.evaluate(ctx)
            if finding and finding.triggered:
                findings.append(finding)
        except Exception:  # 单条规则异常不影响其它规则
            continue
    return findings


def overall_risk_level(findings: List[ComplianceFinding]) -> str:
    """由命中的 findings 汇总整体风险等级。"""
    if not findings:
        return "无"
    return max(findings, key=lambda f: SEVERITY_ORDER.get(f.severity, 0)).severity


def build_context(company: str = "", relations: Optional[List[Dict[str, Any]]] = None,
                  graph_paths: Optional[List[Dict[str, Any]]] = None,
                  documents: Optional[List[Dict[str, Any]]] = None,
                  metrics: Optional[Dict[str, float]] = None,
                  disclosed: Optional[Set[str]] = None) -> ReviewContext:
    """便捷构造 ReviewContext。"""
    return ReviewContext(
        company=company or "",
        relations=list(relations or []),
        graph_paths=list(graph_paths or []),
        documents=list(documents or []),
        metrics=dict(metrics or {}),
        disclosed=set(disclosed or set()),
    )


if __name__ == "__main__":
    demo_ctx = build_context(
        company="广州XX贸易有限公司",
        metrics={"gmv": 5000000.0, "listing_count": 120},
        relations=[
            {"head": "广州XX贸易有限公司", "rel": "PROMOTES", "tail": "深海鱼油软胶囊宣传",
             "props": {"banned_words": "国家级,最安全", "discount_rate": "80%"}},
            {"head": "广州XX贸易有限公司", "rel": "HOLDS_CERT", "tail": "食品经营许可证",
             "props": {"cert_type": "食品经营许可证", "status": "缺失"}},
            {"head": "广州XX贸易有限公司", "rel": "SANCTIONED_BY", "tail": "平台治理中心",
             "props": {"reason": "虚假宣传", "amount": "5000元", "status": "未整改"}},
        ],
    )
    for f in evaluate_rules(demo_ctx):
        print(f.severity, f.rule_id, f.rule_name, "-", f.message)
    print("整体风险等级：", overall_risk_level(evaluate_rules(demo_ctx)))
