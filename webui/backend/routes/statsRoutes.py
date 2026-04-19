"""统计数据路由。

从 data/stats/ 读取已生成的术语统计 JSON；若不存在则返回空对象。
"""

from __future__ import annotations

import json
import os

from fastapi import APIRouter

router = APIRouter()


def _statsDir() -> str:
    from core import config

    return config.STATS_DIR


@router.get("")
def getStats() -> dict:
    """读取术语统计数据（若 stats 命令已运行过）。"""
    statsDir = _statsDir()
    if not os.path.isdir(statsDir):
        return {"available": False, "statsDir": statsDir, "data": {}}

    result: dict = {"available": True, "statsDir": statsDir, "files": {}}
    for name in os.listdir(statsDir):
        path = os.path.join(statsDir, name)
        if not os.path.isfile(path):
            continue
        if name.endswith(".json"):
            try:
                with open(path, encoding="utf-8") as f:
                    result["files"][name] = json.load(f)
            except Exception as e:
                result["files"][name] = {"_error": str(e)}
    return result


@router.get("/figures")
def listStatsFigures() -> list[dict]:
    """列出 outputs/figures/ 下与统计相关的 PNG。"""
    from core import config
    from core.config import getReportsGenerationConfig

    reportsCfg = getReportsGenerationConfig()
    vizFilenames = reportsCfg.get("viz_filenames", {})
    figuresDir = os.path.join(
        config.FIGURES_DIR, reportsCfg.get("viz_output_subdir", "visualizations")
    )

    items: list[dict] = []
    if os.path.isdir(figuresDir):
        for label, fname in vizFilenames.items():
            fullPath = os.path.join(figuresDir, fname)
            if os.path.isfile(fullPath):
                items.append(
                    {
                        "label": label,
                        "filename": fname,
                        "relPath": os.path.join(
                            reportsCfg.get("viz_output_subdir", "visualizations"),
                            fname,
                        ).replace(os.sep, "/"),
                    }
                )
    return items
