"""RAG 问答路由：

- POST /api/rag/query  非流式单次问答（备用，不适合长回答）
- POST /api/rag/batch  批量问答（长任务，走 TaskManager）
- WS   /ws/rag         流式问答（推荐路径）
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from webui.backend.schemas import (
    RagBatchRequest,
    RagQueryRequest,
    RagQueryResponse,
    TaskRef,
)
from webui.backend.taskManager import getTaskManager

router = APIRouter()
wsRouter = APIRouter()

# 全局 RagPipeline 单例（懒加载，所有请求共享）
_ragPipeline = None
_ragLock = asyncio.Lock()


async def _getRagPipeline():
    """异步获取单例 RagPipeline。首次调用会初始化检索器/生成器。"""
    global _ragPipeline
    async with _ragLock:
        if _ragPipeline is None:
            from core import config
            from core.answerGeneration.ragPipeline import RagPipeline

            retrievalCfg = config.getRetrievalConfig()
            _ragPipeline = RagPipeline(
                strategy="hybrid",
                topK=5,
                modelName=retrievalCfg.get(
                    "default_vector_model", "BAAI/bge-base-zh-v1.5"
                ),
                hybridAlpha=float(retrievalCfg.get("bm25_default_weight", 0.7)),
                hybridBeta=float(retrievalCfg.get("vector_default_weight", 0.3)),
            )
    return _ragPipeline


# ── REST：非流式单次问答 ─────────────────────────────────────────────


@router.post("/query", response_model=RagQueryResponse)
async def ragQuery(req: RagQueryRequest) -> RagQueryResponse:
    rag = await _getRagPipeline()

    def runQuery() -> dict[str, Any]:
        if req.useRag:
            return rag.query(
                queryText=req.query,
                temperature=req.temperature,
                topP=req.topP,
                maxNewTokens=req.maxNewTokens,
            )

        # 非 RAG：直接调用 generator
        from core.answerGeneration.generatorFactory import createGenerator

        generator = createGenerator()
        messages = [
            {
                "role": "system",
                "content": "你是一位专业的数学教学助手。请直接回答用户的数学问题。",
            },
            {"role": "user", "content": req.query},
        ]
        startTime = time.time()
        answer = generator.generateFromMessages(
            messages=messages,
            temperature=req.temperature,
            topP=req.topP,
            maxNewTokens=req.maxNewTokens,
        )
        elapsed = int((time.time() - startTime) * 1000)
        return {
            "query": req.query,
            "answer": answer,
            "retrieved_terms": [],
            "sources": [],
            "latency": {
                "retrieval_ms": 0,
                "generation_ms": elapsed,
                "total_ms": elapsed,
            },
        }

    result = await asyncio.to_thread(runQuery)
    return RagQueryResponse(
        query=result.get("query", req.query),
        answer=result.get("answer", ""),
        retrievedTerms=result.get("retrieved_terms", []),
        sources=result.get("sources", []),
        latency=result.get("latency", {}),
    )


# ── REST：批量问答（长任务） ────────────────────────────────────────


@router.post("/batch", response_model=TaskRef)
async def ragBatch(req: RagBatchRequest) -> TaskRef:
    rag = await _getRagPipeline()

    def runBatch() -> dict[str, Any]:
        results = rag.batchQuery(
            req.queries,
            temperature=req.temperature,
            topP=req.topP,
            maxNewTokens=req.maxNewTokens,
            showProgress=True,
        )
        return {"count": len(results), "results": results}

    taskId = await getTaskManager().submit(
        command="rag.batch",
        target=runBatch,
        args={"queryCount": len(req.queries)},
    )
    return TaskRef(taskId=taskId)


# ── WebSocket：流式问答 ──────────────────────────────────────────────


@wsRouter.websocket("/ws/rag")
async def wsRag(websocket: WebSocket) -> None:
    """流式 RAG 问答协议：

    客户端 → 服务器（一次）：
        {"query": str, "useRag": bool, "temperature": float?, "topP": float?,
         "maxNewTokens": int?}

    服务器 → 客户端（多次）：
        {"type": "status", "message": str}
        {"type": "retrieval", "retrievedTerms": [...], "sources": [...]}
        {"type": "token", "delta": str}
        {"type": "done", "latency": {...}}
        {"type": "error", "error": str}
    """
    await websocket.accept()
    try:
        payload = await websocket.receive_text()
        data = json.loads(payload)
    except Exception as e:
        await websocket.send_json({"type": "error", "error": f"请求解析失败: {e}"})
        await websocket.close()
        return

    query = str(data.get("query", "")).strip()
    useRag = bool(data.get("useRag", True))
    temperature = data.get("temperature")
    topP = data.get("topP")
    maxNewTokens = data.get("maxNewTokens")

    if not query:
        await websocket.send_json({"type": "error", "error": "查询为空"})
        await websocket.close()
        return

    try:
        await _runStream(
            websocket,
            query=query,
            useRag=useRag,
            temperature=temperature,
            topP=topP,
            maxNewTokens=maxNewTokens,
        )
    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


async def _runStream(
    websocket: WebSocket,
    *,
    query: str,
    useRag: bool,
    temperature: float | None,
    topP: float | None,
    maxNewTokens: int | None,
) -> None:
    totalStart = time.time()

    if not useRag:
        # 非 RAG 模式：直接流式调用 generator
        await websocket.send_json({"type": "status", "message": "直接生成..."})
        await _streamPlainAnswer(
            websocket,
            query,
            temperature=temperature,
            topP=topP,
            maxNewTokens=maxNewTokens,
            totalStart=totalStart,
        )
        return

    rag = await _getRagPipeline()

    # 非数学领域短路：与 RagPipeline.query 行为保持一致
    if not rag._isMathDomainQuery(query):
        await websocket.send_json(
            {"type": "retrieval", "retrievedTerms": [], "sources": []}
        )
        await websocket.send_json({"type": "token", "delta": "我不知道。"})
        await websocket.send_json(
            {
                "type": "done",
                "latency": {
                    "retrieval_ms": 0,
                    "generation_ms": 0,
                    "total_ms": int((time.time() - totalStart) * 1000),
                },
            }
        )
        return

    # 检索
    await websocket.send_json({"type": "status", "message": "检索中..."})
    retrievalStart = time.time()
    retrievalResults: list[dict[str, Any]]

    def doRetrieve() -> list[dict[str, Any]]:
        raw = rag._retrieve(query)
        return rag._enrichResults(raw)

    try:
        retrievalResults = await asyncio.to_thread(doRetrieve)
    except Exception as e:
        retrievalResults = []
        await websocket.send_json({"type": "status", "message": f"检索失败: {e}"})

    retrievalMs = int((time.time() - retrievalStart) * 1000)

    sources = [
        {"source": r.get("source", ""), "page": r.get("page")}
        for r in retrievalResults
        if r.get("source")
    ]
    retrievedTerms = [
        {
            "rank": r.get("rank"),
            "term": r.get("term", ""),
            "subject": r.get("subject", ""),
            "source": r.get("source", ""),
            "page": r.get("page"),
            "score": r.get("score", 0.0),
            "text": r.get("text", ""),
        }
        for r in retrievalResults
    ]
    await websocket.send_json(
        {"type": "retrieval", "retrievedTerms": retrievedTerms, "sources": sources}
    )

    # 拒答判断
    if rag._shouldRefuseOutOfScope(query, retrievalResults):
        await websocket.send_json({"type": "token", "delta": "我不知道。"})
        await websocket.send_json(
            {
                "type": "done",
                "latency": {
                    "retrieval_ms": retrievalMs,
                    "generation_ms": 0,
                    "total_ms": int((time.time() - totalStart) * 1000),
                },
            }
        )
        return

    # 生成：调用 RagPipeline.queryStream 异步生成器
    await websocket.send_json({"type": "status", "message": "生成中..."})
    generationStart = time.time()

    try:
        async for delta in rag.queryStream(
            queryText=query,
            retrievalResults=retrievalResults,
            temperature=temperature,
            topP=topP,
            maxNewTokens=maxNewTokens,
        ):
            if delta:
                await websocket.send_json({"type": "token", "delta": delta})
    except Exception as e:
        await websocket.send_json({"type": "error", "error": f"生成失败: {e}"})
        return

    generationMs = int((time.time() - generationStart) * 1000)
    await websocket.send_json(
        {
            "type": "done",
            "latency": {
                "retrieval_ms": retrievalMs,
                "generation_ms": generationMs,
                "total_ms": int((time.time() - totalStart) * 1000),
            },
        }
    )


async def _streamPlainAnswer(
    websocket: WebSocket,
    query: str,
    *,
    temperature: float | None,
    topP: float | None,
    maxNewTokens: int | None,
    totalStart: float,
) -> None:
    """无检索纯模型流式问答。"""
    from core.answerGeneration.generatorFactory import createGenerator

    generator = createGenerator()
    messages = [
        {
            "role": "system",
            "content": (
                "你是一位专业的数学教学助手，专注于大学数学课程。"
                "数学公式使用 LaTeX（行内 $...$，行间 $$...$$）。"
                "仅在你确实无法确定答案时才回答“我不知道。”；不回答与数学无关的话题。"
            ),
        },
        {"role": "user", "content": query},
    ]

    generationStart = time.time()

    # 优先使用流式接口
    streamFn = getattr(generator, "generateStreamFromMessages", None)
    if streamFn is not None:
        import asyncio

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel: object = object()

        def pump() -> None:
            try:
                for chunk in streamFn(
                    messages=messages,
                    temperature=temperature,
                    topP=topP,
                    maxNewTokens=maxNewTokens,
                ):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop).result()

        asyncio.create_task(asyncio.to_thread(pump))

        while True:
            item = await queue.get()
            if item is sentinel:
                break
            if item:
                await websocket.send_json({"type": "token", "delta": item})
    else:
        # 回退：一次性生成
        def runSync() -> str:
            return generator.generateFromMessages(
                messages=messages,
                temperature=temperature,
                topP=topP,
                maxNewTokens=maxNewTokens,
            )

        answer = await asyncio.to_thread(runSync)
        await websocket.send_json({"type": "token", "delta": answer})

    await websocket.send_json(
        {
            "type": "done",
            "latency": {
                "retrieval_ms": 0,
                "generation_ms": int((time.time() - generationStart) * 1000),
                "total_ms": int((time.time() - totalStart) * 1000),
            },
        }
    )
