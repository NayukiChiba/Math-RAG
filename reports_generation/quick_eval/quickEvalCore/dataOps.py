"""Dataset loading for quick modelEvaluation."""

from __future__ import annotations

import random
from typing import Any

from core.modelEvaluation.common.ioUtils import loadJsonlFile


def loadQueries(
    filepath: str, num_queries: int | None = None, all_queries: bool = False
) -> list[dict[str, Any]]:
    raw_queries = loadJsonlFile(filepath)
    if not raw_queries:
        print(f" 查询集文件不存在或为空：{filepath}")
        return []

    queries: list[dict[str, Any]] = []
    for query in raw_queries:
        if all(k in query for k in ["query", "relevant_terms", "subject"]):
            if isinstance(query["relevant_terms"], list) and query["relevant_terms"]:
                queries.append(query)

    print(f" 加载了 {len(queries)} 条查询")

    if not all_queries and num_queries and num_queries < len(queries):
        print(f" 随机抽样 {num_queries} 条查询进行测试")
        random.seed(42)
        queries = random.sample(queries, num_queries)

    return queries


def loadCorpus(filepath: str) -> list[dict[str, Any]]:
    corpus = loadJsonlFile(filepath)
    if corpus:
        print(f" 加载了 {len(corpus)} 条语料")
        return corpus
    print(f"  语料文件不存在或为空：{filepath}")
    return []
