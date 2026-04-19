"""任务事件总线。

为每个任务维护一个 asyncio.Queue，WebSocket 端点订阅时读取该队列。
任务运行时向队列推送事件（日志、进度、完成/失败）。
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any


class EventBus:
    """简单的任务事件总线：每个 taskId 对应多个订阅者队列。"""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def publish(self, taskId: str, event: dict[str, Any]) -> None:
        """向该任务的所有订阅者广播事件，并写入历史缓存。"""
        async with self._lock:
            # 有限历史：最后 500 条日志事件
            history = self._history[taskId]
            history.append(event)
            if len(history) > 500:
                del history[: len(history) - 500]

            queues = list(self._subscribers[taskId])

        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # 队列满则丢弃最旧事件，保留最新（保护订阅者不阻塞发布）
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    pass

    async def subscribe(self, taskId: str) -> asyncio.Queue:
        """订阅指定任务。返回一个已填充历史事件的队列。"""
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        async with self._lock:
            # 回放历史事件
            for event in self._history.get(taskId, []):
                queue.put_nowait(event)
            self._subscribers[taskId].add(queue)
        return queue

    async def unsubscribe(self, taskId: str, queue: asyncio.Queue) -> None:
        """取消订阅。"""
        async with self._lock:
            self._subscribers[taskId].discard(queue)

    async def clearTask(self, taskId: str) -> None:
        """清空任务历史与订阅者（任务彻底删除时调用）。"""
        async with self._lock:
            self._history.pop(taskId, None)
            self._subscribers.pop(taskId, None)


_eventBus = EventBus()


def getEventBus() -> EventBus:
    """获取全局事件总线单例。"""
    return _eventBus
