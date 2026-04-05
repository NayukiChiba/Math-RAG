"""测试全局配置：确保 src 目录位于 sys.path 中。"""

import os
import sys

# 将 src 目录加入 Python 路径，使得 import config / import retrieval 等正常工作
_SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
