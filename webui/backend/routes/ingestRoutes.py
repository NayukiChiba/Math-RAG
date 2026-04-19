"""PDF 入库路由。

- POST /api/ingest/upload   上传 PDF 到 data/raw/
- POST /api/ingest          启动入库流水线（OCR → 术语抽取 → 生成 → 索引）
"""

from __future__ import annotations

import argparse
import os
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from webui.backend.schemas import IngestRequest, TaskRef
from webui.backend.taskManager import getTaskManager

router = APIRouter()


@router.post("/upload")
async def uploadPdf(file: UploadFile = File(...)) -> dict:
    """将 PDF 上传到 data/raw/（同名存在时拒绝覆盖）。"""
    from core import config

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    rawDir = config.RAW_DIR
    os.makedirs(rawDir, exist_ok=True)

    target = os.path.join(rawDir, file.filename)
    if os.path.exists(target):
        raise HTTPException(status_code=409, detail=f"同名文件已存在: {file.filename}")

    with open(target, "wb") as f:
        shutil.copyfileobj(file.file, f)

    sizeBytes = os.path.getsize(target)
    return {"name": file.filename, "sizeBytes": sizeBytes, "path": target}


@router.post("", response_model=TaskRef)
async def startIngest(req: IngestRequest) -> TaskRef:
    """启动 PDF 入库任务。等价于 `python main.py cli ingest ...`。"""
    from core import config
    from core.cli.handlers import handle_ingest

    retrievalCfg = config.getRetrievalConfig()
    vectorModel = req.vectorModel or retrievalCfg.get(
        "default_vector_model", "BAAI/bge-base-zh-v1.5"
    )

    args = argparse.Namespace(
        pdf=req.pdf,
        ocr_start_page=req.ocrStartPage,
        extract_start_page=req.extractStartPage,
        generate_start_page=req.generateStartPage,
        skip_generation=req.skipGeneration,
        skip_index=req.skipIndex,
        rebuild_index=req.rebuildIndex,
        skip_bm25=req.skipBm25,
        skip_bm25plus=req.skipBm25plus,
        skip_vector=req.skipVector,
        vector_model=vectorModel,
        batch_size=req.batchSize,
    )

    def run() -> dict:
        handle_ingest(args)
        return {"ok": True}

    taskId = await getTaskManager().submit(
        command="cli.ingest",
        target=run,
        args=req.model_dump(),
    )
    return TaskRef(taskId=taskId)
