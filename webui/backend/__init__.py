"""FastAPI 后端入口包。

在导入 app 之前先将 `src/` 与项目根加入 sys.path，
使得后续 app 及 routes 中的 `core.*` / `research.*` 等导入可正常解析。
"""

from __future__ import annotations

import os
import sys

_backendDir = os.path.dirname(os.path.abspath(__file__))
_webuiDir = os.path.dirname(_backendDir)
_projectRoot = os.path.dirname(_webuiDir)
_srcDir = os.path.join(_projectRoot, "src")
for _path in (_projectRoot, _srcDir):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from webui.backend.app import createApp  # noqa: E402  # sys.path 设置后再导入

__all__ = ["createApp"]
