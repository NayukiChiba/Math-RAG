"""统一输出路径管理。"""

import os
import time


class OutputManager:
    """控制一次运行内的输出目录与文件路径。"""

    def __init__(self, base_dir: str):
        self.base_dir = os.path.abspath(base_dir)
        self._run_dir = None
        self._json_dir = None
        self._text_dir = None

    def set_base_dir(self, base_dir: str) -> None:
        """更新基础目录并重置缓存。"""
        new_base_dir = os.path.abspath(base_dir)
        if new_base_dir != self.base_dir:
            self.base_dir = new_base_dir
            self.reset()

    def reset(self) -> None:
        """重置当前运行缓存目录。"""
        self._run_dir = None
        self._json_dir = None
        self._text_dir = None

    def get_run_dir(self) -> str:
        """获取运行目录：<base_dir>/YYYYMMDD_HHMMSS。"""
        if self._run_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self._run_dir = os.path.join(self.base_dir, timestamp)
            os.makedirs(self._run_dir, exist_ok=True)
        return self._run_dir

    def get_json_dir(self) -> str:
        """获取 JSON 输出目录：<run_dir>/json。"""
        if self._json_dir is None:
            self._json_dir = os.path.join(self.get_run_dir(), "json")
            os.makedirs(self._json_dir, exist_ok=True)
        return self._json_dir

    def get_text_dir(self) -> str:
        """获取文本日志目录：<run_dir>/text。"""
        if self._text_dir is None:
            self._text_dir = os.path.join(self.get_run_dir(), "text")
            os.makedirs(self._text_dir, exist_ok=True)
        return self._text_dir

    def normalize_json_path(self, path: str | None, default_name: str) -> str:
        """将任意路径归一化到 JSON 目录。"""
        filename = os.path.basename(path or "").strip() or default_name
        return os.path.join(self.get_json_dir(), filename)

    def normalize_text_path(self, path: str | None, default_name: str) -> str:
        """将任意路径归一化到 text 目录。"""
        filename = os.path.basename(path or "").strip() or default_name
        return os.path.join(self.get_text_dir(), filename)


_DEFAULT_OUTPUT_MANAGER: OutputManager | None = None


def getOutputManager(base_dir: str) -> OutputManager:
    """获取可复用的全局 OutputManager 实例。"""
    global _DEFAULT_OUTPUT_MANAGER
    if _DEFAULT_OUTPUT_MANAGER is None:
        _DEFAULT_OUTPUT_MANAGER = OutputManager(base_dir)
    else:
        _DEFAULT_OUTPUT_MANAGER.set_base_dir(base_dir)
    return _DEFAULT_OUTPUT_MANAGER
