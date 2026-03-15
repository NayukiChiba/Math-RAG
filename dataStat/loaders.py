"""数据统计读取工具。"""

from typing import Any

from utils import getFileLoader

_LOADER = getFileLoader()


def loadJsonFile(filepath: str) -> dict[str, Any] | None:
    """加载 JSON 文件。"""
    try:
        return _LOADER.json(filepath)
    except Exception as e:
        print(f"❌ 加载文件失败: {filepath}, 错误: {e}")
        return None
