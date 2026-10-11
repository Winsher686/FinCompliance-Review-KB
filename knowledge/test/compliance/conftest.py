"""pytest 公共夹具：把项目根目录加入 sys.path，使 `knowledge.*` 可导入。

这样无论是 `pytest knowledge/test/compliance` 还是直接 `python knowledge/test/compliance/xxx.py`
都能正确解析包路径，无需安装或设置 PYTHONPATH。
"""

import os
import sys

# 本文件位于 <root>/knowledge/test/compliance/conftest.py，向上 4 层即项目根
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
