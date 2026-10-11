"""电商商品/商家合规领域模型定义（领域层单一事实来源）。

本模块只存放「数据结构」与「词表」，不依赖任何三方库、不产生副作用，
供导入侧（实体/关系抽取清洗）、查询侧（实体抽取、图谱遍历）、
审查侧（合规规则）与 API 层共同引用，避免多处硬编码导致口径不一致。

设计说明：
- 实体标签按电商平台治理场景定义（Merchant / Product / Qualification / Violation / …），见 ENTITY_LABELS。
- 关系带属性（如商品价格、类目代码、资质有效期、处罚金额），定义在 RELATION_TYPES 的 attrs 中。
- 字段名 item_name 沿用知识库既有命名，其语义为「主体名（商家 / 店铺）」，
  作为知识库的隔离与过滤维度。
"""

from typing import Any, Dict, List, Optional, Set

# 实体名长度上限：商家全称（地区 + 字号 + 行业 + 组织形式）通常较长，故放宽到 40。
MAX_ENTITY_NAME_LENGTH: int = 40

# 兼容旧代码引用（导入侧原本定义在 kg_graph_node.py）
DEFAULT_RELATION_TYPE: str = "RELATED_TO"

# ==================== 实体标签（Neo4j Label / Milvus entity label） ====================
# key   : 英文标签，写入 Neo4j 作为动态 Label，也是 LLM 抽取时要求输出的 label
# cn    : 中文名，用于日志/报告/前端展示
# desc  : 给 LLM 看的语义说明
# attrs : 该实体允许携带的属性字段（用于抽取清洗白名单）
ENTITY_LABELS: Dict[str, Dict[str, Any]] = {
    "Merchant": {
        "cn": "商家/店铺",
        "desc": "入驻平台的商家、店铺等经营主体",
        "attrs": ["uscc", "shop_id", "reg_address", "business_scope"],
    },
    "Product": {
        "cn": "商品",
        "desc": "在售商品（含商品标题、SKU）",
        "attrs": ["sku", "price", "brand"],
    },
    "Person": {
        "cn": "自然人",
        "desc": "自然人（含店主、法定代表人、经办人）",
        "attrs": ["id_no_masked"],
    },
    "Brand": {
        "cn": "品牌",
        "desc": "商品品牌（自有品牌或授权品牌）",
        "attrs": ["brand_owner", "trademark_no"],
    },
    "Authority": {
        "cn": "监管/平台治理机构",
        "desc": "市场监管部门、平台治理机构等",
        "attrs": ["level"],
    },
    "Violation": {
        "cn": "违规记录",
        "desc": "违规处罚、违规整改等事实",
        "attrs": ["amount", "date", "doc_no", "reason", "status"],
    },
    "Complaint": {
        "cn": "投诉纠纷",
        "desc": "消费者投诉、知识产权纠纷等",
        "attrs": ["case_no", "cause", "amount", "handle_result"],
    },
    "Promotion": {
        "cn": "促销宣传",
        "desc": "促销活动与商品宣传文案（含价格与宣传用语）",
        "attrs": ["amount", "date", "discount_rate", "banned_words", "is_fake_price"],
    },
    "Qualification": {
        "cn": "资质证照",
        "desc": "经营/行业资质证照（如食品经营许可证、化妆品特证）",
        "attrs": ["cert_no", "cert_type", "expire_date", "status"],
    },
    "Supplier": {
        "cn": "供应商",
        "desc": "上游供应商 / 货源方",
        "attrs": ["role", "product"],
    },
    "Category": {
        "cn": "商品类目",
        "desc": "平台类目（一级/二级类目、类目代码）",
        "attrs": ["category", "code", "is_regulated"],
    },
}

# ==================== 关系类型 ====================
# key   : 英文关系类型，写入 Neo4j 作为关系类型
# cn    : 中文名
# from  : 头实体标签集合
# to    : 尾实体标签集合
# attrs : 关系允许携带的属性
RELATION_TYPES: Dict[str, Dict[str, Any]] = {
    "SELLS": {
        "cn": "销售", "from": {"Merchant"}, "to": {"Product"},
        "attrs": ["price", "listing_date"], "desc": "商家在售商品",
    },
    "OPERATES": {
        "cn": "经营", "from": {"Person"}, "to": {"Merchant"},
        "attrs": ["role", "since"], "desc": "自然人经营 / 负责该店铺",
    },
    "AUTHORIZED_BY": {
        "cn": "品牌授权", "from": {"Brand"}, "to": {"Merchant"},
        "attrs": ["auth_type", "expire_date", "level"], "desc": "商家获得品牌授权",
    },
    "BELONGS_TO": {
        "cn": "归属类目", "from": {"Product"}, "to": {"Category"},
        "attrs": ["category_code", "is_misplaced"], "desc": "商品归属类目",
    },
    "PROMOTES": {
        "cn": "促销宣传", "from": {"Merchant"}, "to": {"Promotion"},
        "attrs": ["amount", "date", "discount_rate", "banned_words", "is_fake_price"],
        "desc": "商家发布促销 / 宣传文案",
    },
    "HOLDS_CERT": {
        "cn": "持有资质", "from": {"Merchant"}, "to": {"Qualification"},
        "attrs": ["cert_no", "cert_type", "expire_date", "status"], "desc": "商家持有资质证照",
    },
    "SANCTIONED_BY": {
        "cn": "受到处罚", "from": {"Merchant", "Product"}, "to": {"Authority"},
        "attrs": ["amount", "date", "doc_no", "reason", "status"], "desc": "受到平台 / 监管处罚",
    },
    "COMPLAINED_IN": {
        "cn": "涉及投诉", "from": {"Merchant", "Product", "Supplier"}, "to": {"Complaint"},
        "attrs": ["case_no", "cause", "amount"], "desc": "涉及投诉 / 纠纷案件",
    },
    "SUPPLIES": {
        "cn": "供应货源", "from": {"Supplier", "Merchant"}, "to": {"Merchant"},
        "attrs": ["product", "amount", "direction"], "desc": "供应链上下游关系",
    },
    "SAME_OWNER": {
        "cn": "同一经营者", "from": {"Merchant"}, "to": {"Merchant"},
        "attrs": [], "desc": "由同一实际经营者控制",
    },
    "SHARED_BRAND": {
        "cn": "共用品牌授权", "from": {"Merchant", "Brand"}, "to": {"Merchant"},
        "attrs": ["brand", "ratio"], "desc": "共用同一品牌授权（疑似关联店铺）",
    },
    # 内部结构关系：实体被某个文档切片提及（查询侧依赖它反查 chunk，勿删）
    "MENTIONED_IN": {
        "cn": "提及于", "from": set(ENTITY_LABELS.keys()), "to": {"Chunk"},
        "attrs": [], "desc": "实体被文档切片提及（系统内部关系）",
    },
    # 兜底关系：无法判断类型时使用
    "RELATED_TO": {
        "cn": "相关", "from": set(ENTITY_LABELS.keys()), "to": set(ENTITY_LABELS.keys()),
        "attrs": [], "desc": "通用相关关系（兜底）",
    },
}

# 可直接写入 Neo4j 的关系类型白名单（含内部关系）
ALLOWED_RELATION_TYPES: Set[str] = set(RELATION_TYPES.keys())

# 可直接写入 Neo4j 的实体标签白名单
ALLOWED_ENTITY_LABELS: Set[str] = set(ENTITY_LABELS.keys())

# ==================== 文档类型 ====================
DOC_TYPES: Dict[str, str] = {
    "violation_decision": "违规处罚决定",
    "merchant_profile": "商家资质档案",
    "product_page": "商品详情页",
    "platform_notice": "平台治理公告",
    "promotion_record": "促销活动记录",
    "customer_complaint": "投诉纠纷记录",
    "inspection_report": "抽检/质检报告",
    "other": "其他",
}

# 依据文件名/正文关键词推断文档类型（用于导入侧元数据）
_DOC_TYPE_KEYWORDS: List[tuple] = [
    ("violation_decision", ("违规处罚", "处罚决定书", "违规处理", "违规通知", "罚单", "扣分")),
    ("merchant_profile", ("商家资质", "店铺资质", "营业执照", "入驻资料", "资质档案")),
    ("product_page", ("商品详情", "商品页", "商品信息", "SKU", "sku")),
    ("platform_notice", ("平台公告", "治理规则", "规则公告", "平台通知", "违规公示")),
    ("promotion_record", ("促销", "活动报名", "宣传文案", "直播话术", "满减")),
    ("inspection_report", ("抽检", "质检报告", "检测报告", "质量检验")),
    ("customer_complaint", ("投诉", "举报", "纠纷", "售后")),
]

# 切片元数据字段（写入 Milvus 标量字段 / 供 Prompt 引用）
CHUNK_METADATA_FIELDS: List[str] = [
    "company",       # 主体（商家/店铺）名，等价于 item_name 的语义
    "doc_type",      # 文档类型
    "publish_date",  # 发布日期
    "source",        # 来源
    "chapter",       # 章节路径
    "clause_no",     # 条款号（如 第三条）
]

# ==================== 合规审查维度 ====================
# 审查/检索需要覆盖的合规维度。作为单一事实来源，供改写 Prompt 与规则兜底引用，
# 禁止在节点内另写一份硬编码词表。
COMPLIANCE_DIMENSIONS: List[str] = [
    "商品宣传用语（违禁词/极限词）",
    "类目与价格（错放类目、虚构原价）",
    "资质证照（缺失/过期）",
    "违规处罚与整改",
    "投诉纠纷",
    "货源供应链",
    "品牌授权链",
]


def compliance_dimensions_text() -> str:
    """生成合规维度的一行文本（供查询改写回退模板使用）。"""
    return "、".join(COMPLIANCE_DIMENSIONS)


# ==================== 辅助函数 ====================

def is_valid_entity_label(label: str) -> bool:
    """判断实体标签是否在白名单内。"""
    return (label or "").strip() in ALLOWED_ENTITY_LABELS


def is_valid_relation_type(relation_type: str) -> bool:
    """判断关系类型是否在白名单内。"""
    return (relation_type or "").strip() in ALLOWED_RELATION_TYPES


def normalize_relation_type(relation_type: str) -> str:
    """关系类型归一：非法或缺失时回退到 RELATED_TO。"""
    relation_type = (relation_type or "").strip().upper()
    return relation_type if relation_type in ALLOWED_RELATION_TYPES else DEFAULT_RELATION_TYPE


def entity_label_cn(label: str) -> str:
    """实体标签英文 → 中文展示名。"""
    return ENTITY_LABELS.get((label or "").strip(), {}).get("cn", label or "")


def relation_type_cn(relation_type: str) -> str:
    """关系类型英文 → 中文展示名。"""
    return RELATION_TYPES.get((relation_type or "").strip(), {}).get("cn", relation_type or "")


def relation_attrs(relation_type: str) -> List[str]:
    """获取某关系类型允许的属性字段。"""
    return list(RELATION_TYPES.get((relation_type or "").strip(), {}).get("attrs", []))


def entity_attrs(label: str) -> List[str]:
    """获取某实体标签允许的属性字段。"""
    return list(ENTITY_LABELS.get((label or "").strip(), {}).get("attrs", []))


def infer_doc_type(file_title: str = "", content: str = "") -> str:
    """根据文件名与正文关键词推断文档类型，默认 other。"""
    text = f"{file_title or ''}\n{(content or '')[:2000]}"
    for doc_type, keywords in _DOC_TYPE_KEYWORDS:
        if any(kw in text for kw in keywords):
            return doc_type
    return "other"


def doc_type_cn(doc_type: str) -> str:
    """文档类型英文 → 中文。"""
    return DOC_TYPES.get((doc_type or "").strip(), DOC_TYPES["other"])


def build_entity_prompt_block() -> str:
    """生成抽取 Prompt 中「允许的实体类型」区块。"""
    lines = []
    for label, meta in ENTITY_LABELS.items():
        lines.append(f"- {label}（{meta['cn']}）：{meta['desc']}")
    return "\n".join(lines)


def build_relation_prompt_block() -> str:
    """生成抽取 Prompt 中「允许的关系类型」区块。"""
    lines = []
    for rel, meta in RELATION_TYPES.items():
        if rel in ("MENTIONED_IN", "RELATED_TO"):
            continue
        from_cn = "/".join(entity_label_cn(x) for x in meta["from"])
        to_cn = "/".join(entity_label_cn(x) for x in meta["to"])
        attrs = ("，属性：" + ",".join(meta["attrs"])) if meta["attrs"] else ""
        lines.append(f"- {rel}（{meta['cn']}）：{from_cn} → {to_cn}。{meta['desc']}{attrs}")
    return "\n".join(lines)


def build_attribute_prompt_block() -> str:
    """生成抽取 Prompt 中「属性字段」说明区块。"""
    lines = []
    for rel, meta in RELATION_TYPES.items():
        if not meta["attrs"]:
            continue
        lines.append(f"- {rel}（{meta['cn']}）: {', '.join(meta['attrs'])}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("实体标签：", list(ENTITY_LABELS.keys()))
    print("关系类型：", list(RELATION_TYPES.keys()))
    print("推断文档类型：", infer_doc_type("某商家违规处罚决定书.pdf"))
    print(build_relation_prompt_block())
