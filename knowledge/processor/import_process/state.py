"""

导入流程状态类型定义

定义完整的状态结构和辅助函数（电商合规场景）

"""

from typing import TypedDict, List, Dict, Any, Optional

import copy


class ImportGraphState(TypedDict, total=False):


    """

    导入流程图状态

    包含整个导入流程中传递的所有数据

    """

    # ==================== 任务标识 ====================

    task_id: str  # 任务 ID，用于任务追踪

    # ==================== 控制标志 ====================

    is_md_read_enabled: bool  # 是否启用 MD 读取

    is_pdf_read_enabled: bool  # 是否启用 PDF 读取

    is_doc_convert_enabled: bool  # 是否启用 HTML/Word → Markdown 转换（业务文档常见格式）

    # ==================== 路径信息 ====================

    import_file_path: str  # 导入文件路径

    file_dir: str  # 导入(出)文件目录

    pdf_path: str  # PDF 文件路径

    md_path: str  # 转换后Markdown 文件路径

    # ==================== 文件信息 ====================

    file_title: str  # 文件标题（不含扩展名）

    item_name: str  # 主体（商家/店铺）名称，作为知识库隔离与过滤维度

    # ==================== 文档元数据（电商合规扩展） ====================

    doc_type: str  # 文档类型（product_page / violation_decision / ...，见 commerce_domain.DOC_TYPES）

    publish_date: str  # 发布日期（YYYY-MM-DD 或原文表述）

    source: str  # 来源（平台治理 / 市场监管 / 上传文件）

    # ==================== 处理中间数据 ====================

    md_content: str  # Markdown 文档内容

    chunks: List  # 文档切片列表

    # ==================== 默认状态 ====================


GRAPH_DEFAULT_STATE: ImportGraphState = {

    "task_id": "",

    "is_pdf_read_enabled": False,

    "is_md_read_enabled": False,

    "is_doc_convert_enabled": False,

    "file_dir": "",

    "import_file_path": "",

    "pdf_path": "",

    "md_path": "",

    "file_title": "",

    "md_content": "",

    "chunks": [],

    "item_name": "",

    "doc_type": "",

    "publish_date": "",

    "source": "",

}


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖

    Args:
        **overrides: 要覆盖的字段

    Returns:
        新的状态实例

    Examples:
        >>> state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """
    获取默认状态副本

    Returns:
        状态副本（避免全局污染）
    """
    return copy.deepcopy(GRAPH_DEFAULT_STATE)
