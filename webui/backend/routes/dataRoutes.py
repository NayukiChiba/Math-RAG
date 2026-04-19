"""数据目录查询路由。"""

from __future__ import annotations

import os
from datetime import UTC, datetime

from fastapi import APIRouter

from webui.backend.schemas import ProcessedInfo, RawPdfInfo

router = APIRouter()


def _mtime(path: str) -> str:
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts, tz=UTC).isoformat()
    except Exception:
        return ""


@router.get("/raw", response_model=list[RawPdfInfo])
def listRawPdfs() -> list[RawPdfInfo]:
    """列出 data/raw 下的 PDF 文件。"""
    from core import config

    rawDir = config.RAW_DIR
    if not os.path.isdir(rawDir):
        return []

    items: list[RawPdfInfo] = []
    for name in sorted(os.listdir(rawDir)):
        path = os.path.join(rawDir, name)
        if not os.path.isfile(path):
            continue
        if not name.lower().endswith(".pdf"):
            continue
        try:
            size = os.path.getsize(path)
        except Exception:
            size = 0
        items.append(RawPdfInfo(name=name, sizeBytes=size, modifiedAt=_mtime(path)))
    return items


@router.get("/processed", response_model=ProcessedInfo)
def listProcessed() -> ProcessedInfo:
    """列出 data/processed 各子目录的可用书籍。"""
    from core import config

    def _listSubdirs(root: str) -> list[str]:
        if not os.path.isdir(root):
            return []
        return sorted(
            name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))
        )

    return ProcessedInfo(
        ocrBooks=_listSubdirs(config.OCR_DIR),
        termsBooks=_listSubdirs(config.TERMS_DIR),
        chunkBooks=_listSubdirs(config.CHUNK_DIR),
    )
