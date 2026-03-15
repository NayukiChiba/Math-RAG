"""语料文件 I/O：加载查询文件和 JSON 术语文件。"""

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils import getFileLoader

_LOADER = getFileLoader()


def loadQueriesFile(filepath: str) -> list[dict[str, Any]]:
    """加载评测查询文件。"""
    if not os.path.exists(filepath):
        return []
    return _LOADER.jsonl(filepath)


def loadJsonFile(filepath: str) -> dict[str, Any] | None:
    """
    加载 JSON 文件。

    Args:
        filepath: JSON 文件路径

    Returns:
        解析后的字典，失败返回 None
    """
    try:
        return _LOADER.json(filepath)
    except FileNotFoundError:
        print(f"❌ 文件不存在: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {filepath}, 错误: {e}")
        return None
    except Exception as e:
        print(f"❌ 加载文件失败: {filepath}, 错误: {e}")
        return None
