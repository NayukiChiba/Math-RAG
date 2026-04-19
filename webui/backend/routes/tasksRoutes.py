"""任务管理路由：REST 查询 + WebSocket 日志订阅。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from webui.backend.eventBus import getEventBus
from webui.backend.schemas import TaskInfo
from webui.backend.taskManager import getTaskManager

router = APIRouter()
wsRouter = APIRouter()


@router.get("", response_model=list[TaskInfo])
async def listTasks() -> list[TaskInfo]:
    return await getTaskManager().listTasks()


@router.get("/{taskId}", response_model=TaskInfo)
async def getTask(taskId: str) -> TaskInfo:
    info = await getTaskManager().getTask(taskId)
    if info is None:
        raise HTTPException(status_code=404, detail=f"未找到任务: {taskId}")
    return info


@router.delete("/{taskId}")
async def cancelTask(taskId: str) -> dict:
    ok = await getTaskManager().cancel(taskId)
    if not ok:
        raise HTTPException(status_code=404, detail=f"无法取消: {taskId}")
    return {"cancelled": True}


@wsRouter.websocket("/ws/tasks/{taskId}")
async def wsTaskStream(websocket: WebSocket, taskId: str) -> None:
    """订阅任务事件流。"""
    await websocket.accept()

    bus = getEventBus()
    queue = await bus.subscribe(taskId)

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except TimeoutError:
                # 发送心跳，检测客户端存活
                await websocket.send_json({"type": "ping"})
                continue

            await websocket.send_json(event)

            if event.get("type") == "done":
                # 最终事件发送完毕，主动关闭
                break
    except WebSocketDisconnect:
        pass
    finally:
        await bus.unsubscribe(taskId, queue)
        try:
            await websocket.close()
        except Exception:
            pass
