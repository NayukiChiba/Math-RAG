"""stdout/stderr 捕获到事件总线。

每个长任务运行在独立线程中；使用 contextvars 可能不够可靠
（部分库直接操作 sys.stdout），因此采用"临时替换 sys.stdout/sys.stderr
+ 每行转发到事件总线"的方式，仅在任务线程内生效。

注意：由于 sys.stdout 是进程级共享的，当同时执行多个任务时
本实现会让所有任务的输出混到一起。为了保证 MVP 可用，我们用
串行任务队列（TaskManager 内部维护），同一时刻仅运行一个任务。
"""

from __future__ import annotations

import asyncio
import io
import sys
from typing import Any


class StreamToEventBus(io.TextIOBase):
    """行缓冲的文本流，每遇到换行就推送一条日志事件。"""

    def __init__(
        self,
        taskId: str,
        stream: str,
        eventLoop: asyncio.AbstractEventLoop,
        publisher,
    ):
        self._taskId = taskId
        self._stream = stream  # "stdout" or "stderr"
        self._loop = eventLoop
        self._publisher = publisher  # async 函数 publisher(taskId, event)
        self._buffer = ""
        # 保留原始流，便于同步回显到服务器控制台
        self._mirror = sys.__stdout__ if stream == "stdout" else sys.__stderr__

    def writable(self) -> bool:
        return True

    def write(self, data: str) -> int:
        if not data:
            return 0
        try:
            # 同时回显到真实终端，保留 CLI 可读性
            if self._mirror is not None:
                try:
                    self._mirror.write(data)
                    self._mirror.flush()
                except Exception:
                    pass
        except Exception:
            pass

        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emitLine(line)
        return len(data)

    def flush(self) -> None:
        if self._buffer:
            self._emitLine(self._buffer)
            self._buffer = ""

    def _emitLine(self, line: str) -> None:
        event = {
            "type": "log",
            "stream": self._stream,
            "line": line,
        }
        # 从工作线程将协程调度回主事件循环
        try:
            asyncio.run_coroutine_threadsafe(
                self._publisher(self._taskId, event), self._loop
            )
        except Exception:
            pass


class CaptureContext:
    """上下文管理器：临时替换 sys.stdout/sys.stderr。"""

    def __init__(
        self,
        taskId: str,
        eventLoop: asyncio.AbstractEventLoop,
        publisher,
    ):
        self._taskId = taskId
        self._loop = eventLoop
        self._publisher = publisher
        self._originalStdout: Any = None
        self._originalStderr: Any = None
        self._proxyStdout: StreamToEventBus | None = None
        self._proxyStderr: StreamToEventBus | None = None

    def __enter__(self) -> CaptureContext:
        self._originalStdout = sys.stdout
        self._originalStderr = sys.stderr
        self._proxyStdout = StreamToEventBus(
            self._taskId, "stdout", self._loop, self._publisher
        )
        self._proxyStderr = StreamToEventBus(
            self._taskId, "stderr", self._loop, self._publisher
        )
        sys.stdout = self._proxyStdout
        sys.stderr = self._proxyStderr
        return self

    def __exit__(self, excType, excVal, excTb) -> None:
        try:
            if self._proxyStdout is not None:
                self._proxyStdout.flush()
            if self._proxyStderr is not None:
                self._proxyStderr.flush()
        finally:
            sys.stdout = self._originalStdout
            sys.stderr = self._originalStderr
