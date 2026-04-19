"""报告/日志/图表浏览路由。

仅允许访问 outputs/log、outputs/reports、outputs/figures 三个目录下的文件。
"""

from __future__ import annotations

import mimetypes
import os
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from webui.backend.schemas import FigureInfo, ReportRunInfo

router = APIRouter()


def _safeJoin(rootDir: str, relPath: str) -> str:
    """将 relPath 限制在 rootDir 内，防越权。"""
    rootAbs = os.path.abspath(rootDir)
    targetAbs = os.path.abspath(os.path.join(rootAbs, relPath))
    if os.path.commonpath([rootAbs, targetAbs]) != rootAbs:
        raise HTTPException(status_code=403, detail="禁止访问目标路径")
    return targetAbs


def _mtime(path: str) -> str:
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts, tz=UTC).isoformat()
    except Exception:
        return ""


# ── 运行日志与报告 ──────────────────────────────────────────────────


@router.get("/reports", response_model=list[ReportRunInfo])
def listReportRuns() -> list[ReportRunInfo]:
    """列出所有 outputs/log/<run_id> 目录。"""
    from core import config

    logBase = config.LOG_BASE_DIR
    if not os.path.isdir(logBase):
        return []

    items: list[ReportRunInfo] = []
    for name in sorted(os.listdir(logBase), reverse=True):
        runDir = os.path.join(logBase, name)
        if not os.path.isdir(runDir):
            continue
        jsonDir = os.path.join(runDir, "json")
        finalReport = os.path.join(jsonDir, "final_report.md")
        comparison = os.path.join(jsonDir, "comparison_results.json")
        fullEval = os.path.join(jsonDir, "full_eval")

        items.append(
            ReportRunInfo(
                runId=name,
                path=runDir,
                hasFinalReport=os.path.isfile(finalReport),
                hasComparison=os.path.isfile(comparison),
                hasFullEval=os.path.isdir(fullEval),
                createdAt=_mtime(runDir),
            )
        )
    return items


@router.get("/reports/{runId}/tree")
def listRunFiles(runId: str) -> dict:
    """列出指定运行目录下所有文件（带相对路径）。"""
    from core import config

    runDir = _safeJoin(config.LOG_BASE_DIR, runId)
    if not os.path.isdir(runDir):
        raise HTTPException(status_code=404, detail=f"未找到运行: {runId}")

    files: list[dict] = []
    for dirpath, _, filenames in os.walk(runDir):
        for fname in filenames:
            fullPath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fullPath, runDir).replace(os.sep, "/")
            try:
                size = os.path.getsize(fullPath)
            except Exception:
                size = 0
            files.append(
                {
                    "relPath": rel,
                    "sizeBytes": size,
                    "modifiedAt": _mtime(fullPath),
                }
            )
    files.sort(key=lambda x: x["relPath"])
    return {"runId": runId, "path": runDir, "files": files}


@router.get("/reports/{runId}/file")
def getRunFile(runId: str, path: str):
    """下载/读取运行目录内的文件。"""
    from core import config

    runDir = _safeJoin(config.LOG_BASE_DIR, runId)
    target = _safeJoin(runDir, path)
    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail=f"未找到文件: {path}")

    mime, _ = mimetypes.guess_type(target)
    return FileResponse(target, media_type=mime or "application/octet-stream")


# ── 发布的报告（outputs/reports） ───────────────────────────────────


@router.get("/reports-published/tree")
def listPublishedFiles() -> dict:
    """列出 outputs/reports/ 下全部文件。"""
    from core import config

    baseDir = config.REPORTS_PUBLISH_DIR
    if not os.path.isdir(baseDir):
        return {"path": baseDir, "files": []}

    files: list[dict] = []
    for dirpath, _, filenames in os.walk(baseDir):
        for fname in filenames:
            fullPath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fullPath, baseDir).replace(os.sep, "/")
            try:
                size = os.path.getsize(fullPath)
            except Exception:
                size = 0
            files.append(
                {
                    "relPath": rel,
                    "sizeBytes": size,
                    "modifiedAt": _mtime(fullPath),
                }
            )
    files.sort(key=lambda x: x["relPath"])
    return {"path": baseDir, "files": files}


@router.get("/reports-published/file")
def getPublishedFile(path: str):
    from core import config

    target = _safeJoin(config.REPORTS_PUBLISH_DIR, path)
    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail=f"未找到文件: {path}")
    mime, _ = mimetypes.guess_type(target)
    return FileResponse(target, media_type=mime or "application/octet-stream")


# ── 图表 ────────────────────────────────────────────────────────────


@router.get("/figures", response_model=list[FigureInfo])
def listFigures() -> list[FigureInfo]:
    """列出 outputs/figures/ 下的所有图片。"""
    from core import config

    baseDir = config.FIGURES_DIR
    if not os.path.isdir(baseDir):
        return []

    items: list[FigureInfo] = []
    for dirpath, _, filenames in os.walk(baseDir):
        for fname in filenames:
            fullPath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fullPath, baseDir).replace(os.sep, "/")
            try:
                size = os.path.getsize(fullPath)
            except Exception:
                size = 0
            items.append(
                FigureInfo(
                    relPath=rel,
                    sizeBytes=size,
                    modifiedAt=_mtime(fullPath),
                )
            )
    items.sort(key=lambda x: x.relPath)
    return items


@router.get("/figures/file")
def getFigureFile(path: str):
    from core import config

    target = _safeJoin(config.FIGURES_DIR, path)
    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail=f"未找到图表: {path}")
    mime, _ = mimetypes.guess_type(target)
    return FileResponse(target, media_type=mime or "application/octet-stream")
