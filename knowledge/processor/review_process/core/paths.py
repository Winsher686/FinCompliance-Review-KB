"""审查流程路径配置（与其他流程保持一致的相对目录约定）。"""

import os

# 本流程包根目录： .../knowledge/processor/review_process
REVIEW_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 本地文件存储基础目录
LOCAL_BASE_DIR = os.path.join(REVIEW_ROOT, "temp_data")

# 前端页面静态资源目录
FRONT_PAGE_DIR = os.path.join(REVIEW_ROOT, "front")


def get_local_base_dir() -> str:
    """获取本地文件存储基础目录。"""
    return LOCAL_BASE_DIR


def get_front_page_dir() -> str:
    """获取前端静态页面目录。"""
    return FRONT_PAGE_DIR
