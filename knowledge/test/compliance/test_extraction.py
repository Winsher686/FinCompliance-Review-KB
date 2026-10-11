"""实体/关系抽取清洗测试（knowledge/processor/import_process/nodes/kg_graph_node.py）。

校验：非法标签过滤、超长实体名截断、关系类型归一、属性白名单清洗、同名同标签去重。

注意：`kg_graph_node` 依赖 pymilvus / langchain 等重型三方库。
在精简环境（未安装这些依赖）下，本测试会自动跳过而非报错；
在完整环境（knowledge/.venv，见 start_import_8000.bat）中会完整执行。
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

_IMPORT_OK = True
_IMPORT_ERR = ""
try:
    from knowledge.processor.import_process.nodes.kg_graph_node import KnowLedgeGraphNode
except Exception as e:  # pragma: no cover - 取决于环境是否安装重型依赖
    _IMPORT_OK = False
    _IMPORT_ERR = str(e)

_MERCHANT = "广州XX贸易有限公司"
_BRAND = "森田"


def _require():
    """重型依赖缺失时跳过用例（pytest 下 skip，脚本下打印 SKIP）。"""
    if _IMPORT_OK:
        return True
    msg = f"跳过：缺少重型依赖（{_IMPORT_ERR}）。请在 knowledge/.venv 中运行。"
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    print(f"[SKIP] {msg}")
    return False


def test_clean_entities_filters_labels_and_truncates():
    if not _require():
        return
    node = KnowLedgeGraphNode()
    raw = [
        {"name": _MERCHANT, "label": "Merchant", "attributes": {"uscc": "91440101XXX", "bogus": "x"}},
        {"name": "张三", "label": "Person"},
        {"name": "非法实体", "label": "Device"},          # 非电商标签，应剔除
        {"name": "", "label": "Merchant"},                 # 空名，应剔除
        {"name": "A" * 100, "label": "Merchant"},          # 超长，应截断
        {"name": "张三", "label": "Person"},               # 重复，应去重
    ]
    cleaned = node._clean_entities(raw)
    names = [e["name"] for e in cleaned]
    assert _MERCHANT in names
    assert "非法实体" not in names
    assert "" not in names
    assert any(len(n) <= 40 for n in names)
    # 属性白名单：uscc 保留，bogus 丢弃
    merchant = next(e for e in cleaned if e["name"] == _MERCHANT)
    assert "props" in merchant and merchant["props"].get("uscc") == "91440101XXX"
    assert "bogus" not in merchant.get("props", {})
    # 去重：Person 张三 只保留一个
    assert names.count("张三") == 1


def test_clean_relations_normalizes_and_attrs():
    if not _require():
        return
    node = KnowLedgeGraphNode()
    names = {_MERCHANT, _BRAND}
    raw = [
        {"head": _BRAND, "tail": _MERCHANT, "type": "AUTHORIZED_BY",
         "props": {"level": "一级", "junk": "x"}},
        {"head": _BRAND, "tail": _MERCHANT, "type": "NOT_A_TYPE"},   # 归一为 RELATED_TO
        {"head": "不存在的主体", "tail": _MERCHANT, "type": "AUTHORIZED_BY"},  # head 不在实体集，剔除
        {"head": "", "tail": _MERCHANT, "type": "AUTHORIZED_BY"},    # 空 head，剔除
    ]
    cleaned = node._clean_relations(names, raw)
    assert len(cleaned) == 2
    authorized = next(r for r in cleaned if r["type"] == "AUTHORIZED_BY")
    assert authorized["props"].get("level") == "一级"
    assert "junk" not in authorized["props"]
    assert any(r["type"] == "RELATED_TO" for r in cleaned)


def test_parse_and_clean_full_chain():
    if not _require():
        return
    node = KnowLedgeGraphNode()
    llm_output = "```json\n" + json.dumps({
        "entities": [
            {"name": _MERCHANT, "label": "Merchant", "description": "平台入驻商家"},
            {"name": _BRAND, "label": "Brand", "attributes": {"brand_owner": "森田株式会社"}},
        ],
        "relations": [
            {"head": _BRAND, "tail": _MERCHANT, "type": "AUTHORIZED_BY",
             "props": {"level": "一级"}},
        ],
    }, ensure_ascii=False) + "\n```"
    result = node._parse_and_clean(llm_output)
    assert len(result["entities"]) == 2
    assert len(result["relations"]) == 1
    assert result["relations"][0]["type"] == "AUTHORIZED_BY"


if __name__ == "__main__":
    ran = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
            ran += 1
    print(f"抽取清洗测试完成（{ran} 项）")
