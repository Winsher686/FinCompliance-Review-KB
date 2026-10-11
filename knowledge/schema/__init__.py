"""电商商品/商家合规领域模型（领域层单一事实来源）。

对外统一从本包导出，业务代码使用::

    from knowledge.schema.commerce_domain import ENTITY_LABELS, RELATION_TYPES
"""

from knowledge.schema import commerce_domain  # noqa: F401
from knowledge.schema.commerce_domain import (  # noqa: F401
    MAX_ENTITY_NAME_LENGTH,
    DEFAULT_RELATION_TYPE,
    ENTITY_LABELS,
    RELATION_TYPES,
    ALLOWED_ENTITY_LABELS,
    ALLOWED_RELATION_TYPES,
    DOC_TYPES,
    CHUNK_METADATA_FIELDS,
    COMPLIANCE_DIMENSIONS,
    compliance_dimensions_text,
    is_valid_entity_label,
    is_valid_relation_type,
    normalize_relation_type,
    entity_label_cn,
    relation_type_cn,
    relation_attrs,
    entity_attrs,
    infer_doc_type,
    doc_type_cn,
    build_entity_prompt_block,
    build_relation_prompt_block,
    build_attribute_prompt_block,
)

__all__ = [
    "MAX_ENTITY_NAME_LENGTH",
    "DEFAULT_RELATION_TYPE",
    "ENTITY_LABELS",
    "RELATION_TYPES",
    "ALLOWED_ENTITY_LABELS",
    "ALLOWED_RELATION_TYPES",
    "DOC_TYPES",
    "CHUNK_METADATA_FIELDS",
    "COMPLIANCE_DIMENSIONS",
    "compliance_dimensions_text",
    "is_valid_entity_label",
    "is_valid_relation_type",
    "normalize_relation_type",
    "entity_label_cn",
    "relation_type_cn",
    "relation_attrs",
    "entity_attrs",
    "infer_doc_type",
    "doc_type_cn",
    "build_entity_prompt_block",
    "build_relation_prompt_block",
    "build_attribute_prompt_block",
]
