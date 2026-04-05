"""评测模块通用 I/O 工具。

本模块统一封装 JSON/JSONL 的读取与写入行为，主要价值是：
1. 读写逻辑一致：不同评测脚本使用相同的数据访问约定。
2. 错误处理统一：读取失败时给出清晰日志，并返回安全默认值。
3. 降低耦合：上层业务无需直接依赖具体 Loader 细节。
"""

from __future__ import annotations

import json
from typing import Any

from core.utils import getFileLoader

_LOADER = getFileLoader()


def loadJsonFile(filepath: str) -> dict[str, Any] | None:
    """读取单个 JSON 文件。

    参数：
    - filepath: JSON 文件路径。

    返回：
    - 成功时返回字典。
    - 失败时返回 None，并打印错误信息。

    说明：
    - 这里刻意不抛异常给上层，便于上层按“容错模式”继续处理。
    """
    try:
        return _LOADER.json(filepath)
    except Exception as exc:
        print(f" 加载文件失败: {filepath}, 错误: {exc}")
        return None


def loadJsonlFile(filepath: str) -> list[dict[str, Any]]:
    """读取 JSONL 文件并返回记录列表。

    参数：
    - filepath: JSONL 文件路径。

    返回：
    - 成功时返回列表（每行为一个字典）。
    - 文件不存在或解析失败时返回空列表。

    说明：
    - 返回空列表而非抛异常，可让批处理流程更稳健。
    """
    try:
        rows = _LOADER.jsonl(filepath)
        return rows if isinstance(rows, list) else []
    except FileNotFoundError:
        return []
    except Exception as exc:
        print(f" 加载文件失败: {filepath}, 错误: {exc}")
        return []


def saveJsonFile(payload: dict[str, Any], filepath: str) -> None:
    """保存字典为 JSON 文件。

    写入策略：
    - 使用 UTF-8 编码。
    - `ensure_ascii=False` 保留中文原文，便于人工阅读。
    - `indent=2` 提供可读缩进。
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def saveJsonlFile(rows: list[dict[str, Any]], filepath: str) -> None:
    """保存记录列表为 JSONL 文件。

    规则：
    - 每一条记录占一行。
    - 每行是一个独立 JSON 对象。
    - 保留中文字符，避免转义为 unicode 编码。
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
