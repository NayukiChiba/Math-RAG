"""长任务管理器：串行化的后台任务队列。

设计约束：
- 由于 stdout 是进程级共享，同一时刻只允许一个任务运行，后续任务排队等待。
- 任务状态存内存 dict；重启后丢失（用户希望简单，不做持久化）。
- 每个任务在独立线程中执行目标函数（通过 asyncio.to_thread），避免阻塞事件循环。

事件协议（通过 EventBus 广播）：
- {"type": "log", "stream": "stdout|stderr", "line": str}
- {"type": "status", "status": "pending|running|succeeded|failed|cancelled", "message": str?}
- {"type": "progress", "progress": float, "message": str?}
- {"type": "done", "status": ..., "result": Any?, "error": str?}
"""

from __future__ import annotations

import asyncio
import traceback
import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from webui.backend.eventBus import getEventBus
from webui.backend.logCapture import CaptureContext
from webui.backend.schemas import TaskInfo, TaskStatus


def _now() -> str:
    return datetime.now(UTC).isoformat()


class _TaskRecord:
    """单个任务的内存记录。"""

    def __init__(self, taskId: str, command: str, args: dict[str, Any] | None) -> None:
        self.taskId = taskId
        self.command = command
        self.args = args
        self.status: TaskStatus = "pending"
        self.createdAt = _now()
        self.startedAt: str | None = None
        self.finishedAt: str | None = None
        self.progress: float | None = None
        self.message: str | None = None
        self.errorMessage: str | None = None
        self.logTail: deque[str] = deque(maxlen=200)
        self.result: Any = None
        self.asyncTask: asyncio.Task | None = None

    def toInfo(self) -> TaskInfo:
        return TaskInfo(
            taskId=self.taskId,
            command=self.command,
            status=self.status,
            createdAt=self.createdAt,
            startedAt=self.startedAt,
            finishedAt=self.finishedAt,
            progress=self.progress,
            message=self.message,
            args=self.args,
            errorMessage=self.errorMessage,
            logTail=list(self.logTail),
        )


class TaskManager:
    """串行任务管理器。"""

    def __init__(self) -> None:
        self._tasks: dict[str, _TaskRecord] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workerTask: asyncio.Task | None = None
        self._currentTaskId: str | None = None
        self._lock = asyncio.Lock()
        self._eventBus = getEventBus()

    def ensureWorker(self) -> None:
        """首次使用时启动后台工作协程。"""
        if self._workerTask is None or self._workerTask.done():
            loop = asyncio.get_event_loop()
            self._workerTask = loop.create_task(self._worker())

    async def submit(
        self,
        command: str,
        target: Callable[..., Any],
        args: dict[str, Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> str:
        """提交一个长任务。

        Args:
            command: 任务命令名（用于 UI 展示，如 "cli.ingest"）
            target: 真正执行的可调用对象（同步函数），将在线程池中执行
            args: 记录用的参数字典（可序列化）
            kwargs: 传给 target 的关键字参数（不记录进任务信息，可含非序列化对象）

        Returns:
            taskId
        """
        taskId = str(uuid.uuid4())
        record = _TaskRecord(taskId, command, args)
        record._target = target  # type: ignore[attr-defined]
        record._kwargs = kwargs or {}  # type: ignore[attr-defined]

        async with self._lock:
            self._tasks[taskId] = record

        self.ensureWorker()
        await self._queue.put(taskId)
        await self._eventBus.publish(
            taskId,
            {"type": "status", "status": "pending", "message": "已加入队列"},
        )
        return taskId

    async def listTasks(self) -> list[TaskInfo]:
        async with self._lock:
            items = [r.toInfo() for r in self._tasks.values()]
        items.sort(key=lambda x: x.createdAt, reverse=True)
        return items

    async def getTask(self, taskId: str) -> TaskInfo | None:
        async with self._lock:
            record = self._tasks.get(taskId)
            if record is None:
                return None
            return record.toInfo()

    async def cancel(self, taskId: str) -> bool:
        async with self._lock:
            record = self._tasks.get(taskId)
            if record is None:
                return False

            if record.status == "pending":
                record.status = "cancelled"
                record.finishedAt = _now()
                record.message = "已取消（未启动）"
                await self._eventBus.publish(
                    taskId,
                    {"type": "done", "status": "cancelled", "error": None},
                )
                return True

            if record.status == "running" and record.asyncTask is not None:
                record.asyncTask.cancel()
                return True

        return False

    async def updateProgress(
        self, taskId: str, progress: float, message: str | None = None
    ) -> None:
        """由目标函数主动报告进度（可选）。"""
        async with self._lock:
            record = self._tasks.get(taskId)
            if record is None:
                return
            record.progress = progress
            record.message = message
        await self._eventBus.publish(
            taskId,
            {"type": "progress", "progress": progress, "message": message},
        )

    # ── 内部 ─────────────────────────────────────────────────────────

    async def _worker(self) -> None:
        """串行消费任务队列。"""
        while True:
            taskId = await self._queue.get()
            async with self._lock:
                record = self._tasks.get(taskId)
                if record is None or record.status == "cancelled":
                    continue

            await self._runOne(record)

    async def _runOne(self, record: _TaskRecord) -> None:
        taskId = record.taskId
        loop = asyncio.get_event_loop()

        async with self._lock:
            record.status = "running"
            record.startedAt = _now()
            self._currentTaskId = taskId

        await self._eventBus.publish(taskId, {"type": "status", "status": "running"})

        async def logPublisher(tid: str, event: dict[str, Any]) -> None:
            async with self._lock:
                rec = self._tasks.get(tid)
                if rec is not None and event.get("type") == "log":
                    line = event.get("line", "")
                    if line:
                        rec.logTail.append(line)
            await self._eventBus.publish(tid, event)

        def runInThread() -> Any:
            with CaptureContext(taskId, loop, logPublisher):
                target = record._target  # type: ignore[attr-defined]
                kwargs = record._kwargs  # type: ignore[attr-defined]
                return target(**kwargs)

        record.asyncTask = loop.create_task(asyncio.to_thread(runInThread))

        try:
            result = await record.asyncTask
            async with self._lock:
                record.status = "succeeded"
                record.finishedAt = _now()
                record.progress = 1.0
                record.result = result
            await self._eventBus.publish(
                taskId,
                {"type": "done", "status": "succeeded", "result": _safeResult(result)},
            )
        except asyncio.CancelledError:
            async with self._lock:
                record.status = "cancelled"
                record.finishedAt = _now()
                record.message = "已取消"
            await self._eventBus.publish(
                taskId, {"type": "done", "status": "cancelled"}
            )
        except SystemExit as e:
            async with self._lock:
                if int(getattr(e, "code", 0) or 0) == 0:
                    record.status = "succeeded"
                else:
                    record.status = "failed"
                    record.errorMessage = f"SystemExit: {e.code}"
                record.finishedAt = _now()
            await self._eventBus.publish(
                taskId,
                {
                    "type": "done",
                    "status": record.status,
                    "error": record.errorMessage,
                },
            )
        except Exception as e:
            tb = traceback.format_exc()
            async with self._lock:
                record.status = "failed"
                record.finishedAt = _now()
                record.errorMessage = f"{type(e).__name__}: {e}\n{tb}"
            await self._eventBus.publish(
                taskId,
                {"type": "done", "status": "failed", "error": record.errorMessage},
            )
        finally:
            async with self._lock:
                self._currentTaskId = None


def _safeResult(value: Any) -> Any:
    """尽量保留可 JSON 序列化的结果，否则转字符串。"""
    try:
        import json

        json.dumps(value)
        return value
    except Exception:
        return str(value)


_taskManager: TaskManager | None = None


def getTaskManager() -> TaskManager:
    global _taskManager
    if _taskManager is None:
        _taskManager = TaskManager()
    return _taskManager
