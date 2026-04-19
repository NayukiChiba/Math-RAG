"""FastAPI 应用入口。"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from webui.backend.routes import (
    configRoutes,
    dataRoutes,
    indexRoutes,
    ingestRoutes,
    ragRoutes,
    reportsRoutes,
    researchRoutes,
    statsRoutes,
    tasksRoutes,
)


def createApp() -> FastAPI:
    """构建 FastAPI 应用。

    开发模式下仅暴露 /api 和 /ws；生产模式下额外挂载前端静态文件。
    """
    app = FastAPI(
        title="Math-RAG Web API",
        description="Math-RAG 产品线与研究线统一 Web API",
        version="0.1.0",
    )

    # 开发模式允许 Vite dev server (5173) 跨域访问
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    # 路由注册
    app.include_router(ragRoutes.router, prefix="/api/rag", tags=["rag"])
    app.include_router(ingestRoutes.router, prefix="/api/ingest", tags=["ingest"])
    app.include_router(indexRoutes.router, prefix="/api/index", tags=["index"])
    app.include_router(researchRoutes.router, prefix="/api/research", tags=["research"])
    app.include_router(reportsRoutes.router, prefix="/api", tags=["reports"])
    app.include_router(dataRoutes.router, prefix="/api/data", tags=["data"])
    app.include_router(configRoutes.router, prefix="/api/config", tags=["config"])
    app.include_router(statsRoutes.router, prefix="/api/stats", tags=["stats"])
    app.include_router(tasksRoutes.router, prefix="/api/tasks", tags=["tasks"])

    # WebSocket 路由（直接挂在 app 上，不走 prefix）
    app.include_router(ragRoutes.wsRouter)
    app.include_router(tasksRoutes.wsRouter)

    # 生产模式：挂载前端构建产物
    staticDir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "frontend",
        "dist",
    )
    if os.path.isdir(staticDir):
        _mountFrontend(app, staticDir)

    return app


def _mountFrontend(app: FastAPI, staticDir: str) -> None:
    """挂载 Vite 构建产物，并对任意前端路由回退到 index.html。"""
    assetsDir = os.path.join(staticDir, "assets")
    if os.path.isdir(assetsDir):
        app.mount("/assets", StaticFiles(directory=assetsDir), name="assets")

    @app.get("/{fullPath:path}")
    def spaFallback(fullPath: str):
        # API/WS 不走这里（它们在 include_router 里已注册）
        indexHtml = os.path.join(staticDir, "index.html")
        targetFile = os.path.join(staticDir, fullPath)
        if fullPath and os.path.isfile(targetFile):
            return FileResponse(targetFile)
        return FileResponse(indexHtml)
