"""阶段一（L1 工程底线）修复的回归测试。

覆盖：
- L1-3 CORS 白名单 / API Key 中间件的公开路径判定；
- L1-4 上传文件名净化（防路径遍历 + 类型白名单）；
- L1-5 导入不再 drop_collection 清空历史，改按主体删除；
- L1-7 实体清洗不再被"恒 False"的判空绕过。

设计：纯逻辑用例始终运行；依赖三方库（pymilvus / langchain_deepseek）的用例
在精简环境下自动跳过，保证 `pytest knowledge/test/compliance -q` 无三方库时也能跑通。
"""

import pytest

from knowledge.utils.auth_util import get_cors_origins, _is_public
from knowledge.utils.file_util import sanitize_filename

# pymilvus / langchain 均为可选依赖：缺失时相关用例跳过而非报错
try:
    from knowledge.processor.import_process.nodes.import_milvus_node import ImportMilvusNode
    _HAS_PYMILVUS = True
except Exception:  # pragma: no cover - 取决于运行环境
    ImportMilvusNode = None
    _HAS_PYMILVUS = False

try:
    from knowledge.processor.query_process.nodes.kg_search_node import _clean_parse_llm_content
    _HAS_KG_NODE = True
except Exception:  # pragma: no cover - 取决于运行环境
    _clean_parse_llm_content = None
    _HAS_KG_NODE = False

requires_pymilvus = pytest.mark.skipif(not _HAS_PYMILVUS, reason="需要 pymilvus（精简环境跳过）")
requires_kg_node = pytest.mark.skipif(not _HAS_KG_NODE, reason="需要 langchain 依赖（精简环境跳过）")

_DOC_EXT = {".pdf", ".md", ".markdown", ".html", ".htm", ".docx", ".doc"}


# ==================== L1-3 CORS / 鉴权 ====================

def test_cors_default_origins_never_wildcard():
    """CORS 白名单绝不能是通配符（`*` + allow_credentials 属不安全配置）。"""
    origins = get_cors_origins()
    assert origins, "CORS 白名单不应为空"
    assert "*" not in origins


def test_api_key_middleware_public_paths():
    """健康检查 / 文档 / 静态页放行，业务接口需鉴权。"""
    assert _is_public("/health") is True
    assert _is_public("/") is True
    assert _is_public("/review.html") is True
    assert _is_public("/docs") is True
    assert _is_public("/review") is False
    assert _is_public("/upload") is False
    assert _is_public("/query") is False


# ==================== L1-4 文件名净化 ====================

def test_safe_filename_strips_path_traversal():
    """`../../etc/passwd.md` 只能落到基名，不能穿越目录。"""
    assert sanitize_filename("../../etc/passwd.md", _DOC_EXT) == "passwd.md"
    assert sanitize_filename("..\\..\\windows\\note.html", _DOC_EXT) == "note.html"
    assert "/" not in sanitize_filename("a/b/c/文 件.pdf", _DOC_EXT)


def test_safe_filename_rejects_disallowed_extension():
    """白名单外的扩展名应被拒绝。"""
    with pytest.raises(ValueError):
        sanitize_filename("evil.exe", _DOC_EXT)
    with pytest.raises(ValueError):
        sanitize_filename("noext", _DOC_EXT)


def test_safe_filename_rejects_empty_name():
    with pytest.raises(ValueError):
        sanitize_filename("", _DOC_EXT)
    with pytest.raises(ValueError):
        sanitize_filename("../", _DOC_EXT)


def test_safe_filename_without_whitelist_only_sanitizes():
    """不传白名单时只做路径净化，不校验类型。"""
    assert sanitize_filename("../../x/y/any.bin") == "any.bin"


# ==================== L1-5 导入不再清空历史 ====================

class _FakeMilvusSchema:
    def add_field(self, **kwargs):
        return self


class _FakeMilvusIndex:
    def add_index(self, **kwargs):
        return self


class _StubMilvusClient:
    """记录调用的 Milvus 客户端替身。"""

    def __init__(self, collection_exists: bool):
        self._exists = collection_exists
        self.dropped = []
        self.deleted = []
        self.created = []

    def has_collection(self, collection_name):
        return self._exists

    def drop_collection(self, collection_name):
        self.dropped.append(collection_name)

    def delete(self, collection_name, filter=None):
        self.deleted.append((collection_name, filter))

    def create_schema(self, **kwargs):
        return _FakeMilvusSchema()

    def prepare_index_params(self, collection_name):
        return _FakeMilvusIndex()

    def create_collection(self, collection_name, schema=None, index_params=None):
        self.created.append(collection_name)


@requires_pymilvus
def test_import_milvus_never_drops_existing_collection():
    """集合已存在时：绝不 drop_collection，且按 item_name 删除本次主体。"""
    node = ImportMilvusNode()
    client = _StubMilvusClient(collection_exists=True)

    node._ensure_has_collection(client, "chunks_collection", 1024, delete_item_names=["某商家A"])

    assert client.dropped == [], "集合已存在时不允许 drop_collection（否则会清空其它主体数据）"
    assert len(client.deleted) == 1
    _, flt = client.deleted[0]
    assert "item_name in" in flt
    assert "某商家A" in flt


@requires_pymilvus
def test_import_milvus_skips_cleanup_without_item_name():
    """未识别到主体时不应删除任何数据（宁可保留，也不误删）。"""
    node = ImportMilvusNode()
    client = _StubMilvusClient(collection_exists=True)

    node._ensure_has_collection(client, "chunks_collection", 1024, delete_item_names=[])

    assert client.dropped == []
    assert client.deleted == []


@requires_pymilvus
def test_import_milvus_creates_collection_when_absent():
    """集合不存在时创建，且同样不 drop。"""
    node = ImportMilvusNode()
    client = _StubMilvusClient(collection_exists=False)

    node._ensure_has_collection(client, "chunks_collection", 1024, delete_item_names=["某商家A"])

    assert client.created == ["chunks_collection"]
    assert client.dropped == []


# ==================== L1-7 实体清洗判空 ====================

@requires_kg_node
def test_clean_parse_llm_content_skips_empty_and_non_str():
    """空串 / 非字符串元素应被跳过（旧代码判空写错会放进空串）。"""
    result = _clean_parse_llm_content('{"entities": ["A公司", "", 123, null, "A公司", "B公司"]}')
    assert result == ["A公司", "B公司"]
    assert "" not in result


@requires_kg_node
def test_clean_parse_llm_content_handles_code_fence():
    raw = '```json\n{"entities": ["某科技有限公司"]}\n```'
    assert _clean_parse_llm_content(raw) == ["某科技有限公司"]
