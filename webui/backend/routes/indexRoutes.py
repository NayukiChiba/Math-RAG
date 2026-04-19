"""索引状态查询与构建任务路由。

复用 core.cli.handlers 中的 build_indexes/ensure_corpus 实现，保持 CLI 行为一致。
"""

from __future__ import annotations

import os

from fastapi import APIRouter

from webui.backend.schemas import BuildIndexRequest, IndexStatus, TaskRef
from webui.backend.taskManager import getTaskManager

router = APIRouter()


def _retrievalDir() -> str:
    from core import config

    return os.path.join(config.PROCESSED_DIR, "retrieval")


@router.get("/status", response_model=IndexStatus)
def getIndexStatus() -> IndexStatus:
    retrievalDir = _retrievalDir()

    corpusFile = os.path.join(retrievalDir, "corpus.jsonl")
    bm25File = os.path.join(retrievalDir, "bm25_index.pkl")
    bm25plusFile = os.path.join(retrievalDir, "bm25plus_index.pkl")
    vectorFile = os.path.join(retrievalDir, "vector_index.faiss")

    docCount: int | None = None
    if os.path.isfile(corpusFile):
        try:
            with open(corpusFile, encoding="utf-8") as f:
                docCount = sum(1 for line in f if line.strip())
        except Exception:
            docCount = None

    return IndexStatus(
        corpusExists=os.path.isfile(corpusFile),
        corpusDocCount=docCount,
        bm25Exists=os.path.isfile(bm25File),
        bm25plusExists=os.path.isfile(bm25plusFile),
        vectorExists=os.path.isfile(vectorFile),
        retrievalDir=retrievalDir,
    )


@router.post("/build", response_model=TaskRef)
async def buildIndex(req: BuildIndexRequest) -> TaskRef:
    from core import config
    from core.cli.handlers import build_indexes

    retrievalCfg = config.getRetrievalConfig()
    vectorModel = req.vectorModel or retrievalCfg.get(
        "default_vector_model", "BAAI/bge-base-zh-v1.5"
    )

    def run() -> dict:
        build_indexes(
            rebuild=req.rebuild,
            skip_bm25=req.skipBm25,
            skip_bm25plus=req.skipBm25plus,
            skip_vector=req.skipVector,
            vector_model=vectorModel,
            batch_size=req.batchSize,
        )
        return {"ok": True}

    taskId = await getTaskManager().submit(
        command="cli.build-index",
        target=run,
        args=req.model_dump(),
    )
    return TaskRef(taskId=taskId)
