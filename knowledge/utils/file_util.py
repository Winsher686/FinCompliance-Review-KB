"""文件相关工具。

上传文件名净化单独放在这里，而不是写在业务服务类里，原因有二：
1. 它是安全原语（防目录穿越 / 非法类型），值得独立单测；
2. 这里**不依赖任何三方库**（不 import pymilvus / minio 等），保证纯逻辑可离线测试。
"""

import os
from typing import Iterable, Optional


def sanitize_filename(filename: str, allowed_extensions: Optional[Iterable[str]] = None) -> str:
    """净化上传文件名：只取基名，并按白名单校验扩展名。

    Args:
        filename: 原始上传文件名，可能包含 ``../`` 等路径片段。
        allowed_extensions: 允许的扩展名集合（含点，如 ``{".pdf", ".md"}``）；
            为 ``None`` 时只做路径净化、不校验类型。

    Returns:
        安全的文件名（不含任何目录分隔符）。

    Raises:
        ValueError: 文件名为空 / 非法，或扩展名不在白名单内。
    """
    # 1. 统一分隔符后取基名，杜绝 ../../ 形式的目录穿越
    name = os.path.basename((filename or "").replace("\\", "/")).strip()
    if not name or name in (".", ".."):
        raise ValueError("非法的文件名")

    # 2. 扩展名白名单校验
    if allowed_extensions is not None:
        ext = os.path.splitext(name)[1].lower()
        allowed = {str(item).lower() for item in allowed_extensions}
        if ext not in allowed:
            raise ValueError(f"不支持的文件类型: {ext or '(无扩展名)'}")

    return name
