"""电商领域模型测试（knowledge/schema/commerce_domain.py）。

校验实体/关系白名单、关系属性、文档类型推断与 Prompt 区块生成，
确保导入侧与查询侧引用同一份口径。
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from knowledge.schema import commerce_domain as cd


def test_entity_labels_contain_commerce_core():
    for label in ["Merchant", "Product", "Person", "Brand", "Authority",
                  "Violation", "Complaint", "Promotion", "Qualification",
                  "Supplier", "Category"]:
        assert label in cd.ENTITY_LABELS, f"缺少核心实体标签: {label}"
    assert cd.is_valid_entity_label("Merchant") is True
    assert cd.is_valid_entity_label("Device") is False
    # 非电商领域的标签不应进入白名单，避免场景混用
    assert cd.is_valid_entity_label("Company") is False


def test_relation_types_and_attrs():
    for rel in ["SELLS", "OPERATES", "AUTHORIZED_BY", "BELONGS_TO", "PROMOTES",
                "HOLDS_CERT", "SANCTIONED_BY", "COMPLAINED_IN", "SUPPLIES",
                "SAME_OWNER", "SHARED_BRAND"]:
        assert rel in cd.RELATION_TYPES, f"缺少核心关系: {rel}"
    # 价格/资质/处罚等属性必须定义，供抽取清洗白名单与规则计算使用
    assert "discount_rate" in cd.relation_attrs("PROMOTES")
    assert "banned_words" in cd.relation_attrs("PROMOTES")
    assert "expire_date" in cd.relation_attrs("HOLDS_CERT")
    assert "reason" in cd.relation_attrs("SANCTIONED_BY")


def test_normalize_relation_type_fallback():
    assert cd.normalize_relation_type("sells") == "SELLS"
    assert cd.normalize_relation_type("NOT_A_REL") == cd.DEFAULT_RELATION_TYPE
    assert cd.normalize_relation_type("") == cd.DEFAULT_RELATION_TYPE


def test_infer_doc_type():
    assert cd.infer_doc_type("广州XX_商品详情页.html") == "product_page"
    assert cd.infer_doc_type("", "本中心决定给予违规处罚") == "violation_decision"
    assert cd.infer_doc_type("某商家资质档案.docx") == "merchant_profile"
    assert cd.infer_doc_type("未知文件.txt", "正文无关键词") == "other"


def test_prompt_blocks_non_empty():
    entity_block = cd.build_entity_prompt_block()
    relation_block = cd.build_relation_prompt_block()
    attr_block = cd.build_attribute_prompt_block()
    assert "Merchant" in entity_block
    assert "PROMOTES" in relation_block
    # 内部关系不应出现在给 LLM 的关系说明中
    assert "MENTIONED_IN" not in relation_block
    assert "PROMOTES" in attr_block


def test_compliance_dimensions_single_source():
    """合规维度由领域层统一提供，供改写节点引用（防硬编码漂移）。"""
    text = cd.compliance_dimensions_text()
    assert "违禁词" in text
    assert "资质" in text
    assert len(cd.COMPLIANCE_DIMENSIONS) >= 5


def test_query_rewrite_dimensions_derive_from_schema():
    """回归防线：改写节点的维度串必须来自 schema，不得硬编码（L2-4）。"""
    from knowledge.processor.review_process.nodes.query_rewrite_node import _COMPLIANCE_DIMENSIONS
    assert _COMPLIANCE_DIMENSIONS == cd.compliance_dimensions_text()
    # 合规维度词表不得在节点内硬编码，必须逐项来自领域层
    for dim in cd.COMPLIANCE_DIMENSIONS:
        assert dim in _COMPLIANCE_DIMENSIONS


def test_name_length_constant():
    assert cd.MAX_ENTITY_NAME_LENGTH >= 30  # 商家/店铺全称较长，上限需足够宽松


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    print("领域模型测试全部通过")
