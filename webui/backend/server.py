"""uvicorn 启动包装：供 main.py 调用。"""

from __future__ import annotations

import os
import sys

import uvicorn


def _ensureProjectPaths() -> None:
    """确保 `src/` 与项目根目录都在 sys.path，使得 FastAPI 路由中能 import core/research/reports_generation。"""
    backendDir = os.path.dirname(os.path.abspath(__file__))
    webuiDir = os.path.dirname(backendDir)
    projectRoot = os.path.dirname(webuiDir)
    srcDir = os.path.join(projectRoot, "src")

    for path in (projectRoot, srcDir):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def runServer(host: str = "127.0.0.1", port: int = 7860) -> None:
    """启动 FastAPI 服务器。"""
    _ensureProjectPaths()

    print(f" 启动 Math-RAG Web 服务：http://{host}:{port}")
    print("   - API 文档：/docs")
    print("   - 健康检查：/api/health")

    uvicorn.run(
        "webui.backend.app:createApp",
        host=host,
        port=port,
        factory=True,
        reload=False,
        log_level="info",
    )
