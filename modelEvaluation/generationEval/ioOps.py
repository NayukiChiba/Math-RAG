"""I/O helpers for generation modelEvaluation."""

from __future__ import annotations

from typing import Any

from modelEvaluation.common.ioUtils import loadJsonlFile


def loadRagResults(filepath: str) -> list[dict[str, Any]]:
    results = loadJsonlFile(filepath)
    if results:
        print(f" 加载了 {len(results)} 条 RAG 结果")
        return results
    print(f" 结果文件不存在或为空: {filepath}")
    return []


def loadGoldQueries(filepath: str) -> dict[str, dict[str, Any]]:
    gold_map: dict[str, dict[str, Any]] = {}
    rows = loadJsonlFile(filepath)
    if not rows:
        print(f" 黄金测试集不存在或为空: {filepath}")
        return gold_map

    for row in rows:
        query = row.get("query", "")
        if query:
            gold_map[query] = row

    print(f" 加载了 {len(gold_map)} 条黄金测试数据")
    return gold_map
