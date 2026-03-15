"""Data loading helpers for retrieval modelEvaluation."""

from __future__ import annotations

import os
from typing import Any

import config
from modelEvaluation.common.ioUtils import loadJsonlFile


def _resolveQueryFilePath(filepath: str) -> str:
    """解析查询文件路径。

    兼容以下输入：
    1. 绝对路径。
    2. 相对当前工作目录的路径。
    3. 仅文件名（自动在 data/evaluation 下查找）。
    """
    if os.path.isabs(filepath) and os.path.exists(filepath):
        return filepath

    if os.path.exists(filepath):
        return os.path.abspath(filepath)

    candidateInEvalDir = os.path.join(config.EVALUATION_DIR, os.path.basename(filepath))
    if os.path.exists(candidateInEvalDir):
        return os.path.abspath(candidateInEvalDir)

    return filepath


def loadQueries(filepath: str) -> list[dict[str, Any]]:
    try:
        resolvedPath = _resolveQueryFilePath(filepath)
        if resolvedPath != filepath:
            print(f"📂 查询文件自动解析为: {resolvedPath}")

        raw_queries = loadJsonlFile(resolvedPath)
        if not raw_queries:
            print(f"⚠️  查询文件为空或不存在: {resolvedPath}")

        queries: list[dict[str, Any]] = []
        for i, query in enumerate(raw_queries, 1):
            if not all(k in query for k in ["query", "relevant_terms", "subject"]):
                print(f"⚠️  第 {i} 行缺少必需字段，跳过")
                continue
            if (
                not isinstance(query["relevant_terms"], list)
                or not query["relevant_terms"]
            ):
                print(f"⚠️  第 {i} 行 relevant_terms 格式错误，跳过")
                continue
            queries.append(query)
        print(f"✅ 加载了 {len(queries)} 条查询")
        return queries
    except Exception as exc:
        print(f"❌ 加载查询集失败: {exc}")
        return []
