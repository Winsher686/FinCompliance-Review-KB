"""审查流程请求/响应模型（Pydantic）。"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ==================== 提交审查 ====================

class ReviewSubmitRequest(BaseModel):
    company: str = Field(..., description="审查主体（商家/店铺）名称")
    review_type: str = Field("general", description="审查类型：banned_word/qualification/violation/supply_chain/price_category/brand_authorization/product_compliance/general/report")
    question: str = Field("", description="用户问题描述")
    session_id: Optional[str] = Field(None, description="会话ID，不传则自动生成")
    is_stream: bool = Field(False, description="是否流式（SSE）返回")
    # 可选：直接注入经营指标 / 关系（离线评估或上游系统已装配数据时使用）
    metrics: Optional[Dict[str, float]] = Field(None, description="经营指标，如 gmv / listing_count")
    relations: Optional[List[Dict[str, Any]]] = Field(None, description="已装配的关系列表")


class ReviewSubmitResponse(BaseModel):
    message: str = Field(..., description="响应消息")
    review_id: str = Field(..., description="审查ID")
    task_id: str = Field(..., description="任务ID")
    session_id: str = Field(..., description="会话ID")
    status: str = Field("processing", description="当前状态")


# ==================== 审查结果 ====================

class EvidenceItem(BaseModel):
    type: str = Field("", description="证据类型：graph_path/document/relation")
    content: str = Field("", description="证据内容")
    source: str = Field("", description="证据来源")
    chunk_id: Optional[Any] = Field(None, description="切片ID（文档证据）")
    hops: Optional[int] = Field(None, description="路径跳数（图谱证据）")


class FindingItem(BaseModel):
    rule_id: str = Field("", description="规则编号，如 R1")
    rule_name: str = Field("", description="规则名称")
    severity: str = Field("", description="严重度：高/中/低")
    message: str = Field("", description="命中描述")
    evidence: List[str] = Field(default_factory=list, description="证据")
    suggestion: str = Field("", description="处理建议")
    regulation: str = Field("", description="治理依据")


class ReviewResultResponse(BaseModel):
    review_id: str
    trace_id: str = Field("", description="链路追踪 ID（用于 GET /trace/{trace_id} 查看节点耗时瀑布）")
    company: str = ""
    review_type: str = ""
    question: str = ""
    intent: str = ""
    status: str = Field("", description="processing/pending_approval/approved/rejected/completed/failed/not_found")
    conclusion: str = ""
    risk_level: str = Field("", description="高/中/低/无")
    findings: List[FindingItem] = Field(default_factory=list)
    evidence_chain: List[EvidenceItem] = Field(default_factory=list)
    graph_paths: List[Dict[str, Any]] = Field(default_factory=list)
    report: str = ""
    needs_human_review: bool = False
    approval: Dict[str, Any] = Field(default_factory=dict)


# ==================== 链路追踪 / 指标（可观测性） ====================

class TraceSpan(BaseModel):
    node: str = Field("", description="节点内部名")
    node_cn: str = Field("", description="节点中文名")
    duration_ms: float = Field(0.0, description="该节点耗时（毫秒）")
    status: str = Field("ok", description="ok / error / interrupted")
    extras: Dict[str, Any] = Field(default_factory=dict, description="业务摘要，如召回条数/命中规则数")
    error: Optional[str] = Field(None, description="异常摘要（status != ok 时）")


class TraceResponse(BaseModel):
    trace_id: str = ""
    total_ms: float = Field(0.0, description="所有节点耗时之和（毫秒）")
    node_count: int = 0
    spans: List[TraceSpan] = Field(default_factory=list)
    waterfall: str = Field("", description="人类可读的耗时瀑布文本")


class MetricsResponse(BaseModel):
    metrics: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="每个指标的 count / mean / p50 / p95 / max")



# ==================== 人工审批 ====================

class ApproveRequest(BaseModel):
    approved: bool = Field(True, description="是否通过")
    reviewer: str = Field("", description="复核人")
    comment: str = Field("", description="复核意见")


# ==================== 图谱查询 ====================

class GraphEntityResponse(BaseModel):
    entity_id: str = ""
    name: str = ""
    labels: List[str] = Field(default_factory=list)
    props: Dict[str, Any] = Field(default_factory=dict)
    relations: List[Dict[str, Any]] = Field(default_factory=list)


class GraphPathResponse(BaseModel):
    from_name: str = ""
    to_name: str = ""
    max_hops: int = 3
    paths: List[Dict[str, Any]] = Field(default_factory=list)


# ==================== 任务状态 ====================

class ReviewTaskStatusResponse(BaseModel):
    task_id: str = ""
    status: str = ""
    review_id: str = ""
    risk_level: str = ""
    done_list: List[str] = Field(default_factory=list)
    running_list: List[str] = Field(default_factory=list)
