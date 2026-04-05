"""Data loading/merging/saving for query generation."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

from core.modelEvaluation.common.ioUtils import (
    loadJsonFile,
    loadJsonlFile,
    saveJsonlFile,
)


def loadAllTerms(chunk_dir: str) -> list[dict[str, Any]]:
    terms: list[dict[str, Any]] = []

    print(" 加载术语库...")
    for book_name in sorted(os.listdir(chunk_dir)):
        book_path = os.path.join(chunk_dir, book_name)
        if not os.path.isdir(book_path):
            continue

        json_files = sorted(
            [name for name in os.listdir(book_path) if name.endswith(".json")]
        )
        for json_file in json_files:
            filepath = os.path.join(book_path, json_file)
            data = loadJsonFile(filepath)
            if data and "term" in data and "subject" in data:
                terms.append(
                    {
                        "term": data["term"],
                        "aliases": data.get("aliases", []),
                        "related_terms": data.get("related_terms", []),
                        "subject": data["subject"],
                        "book": book_name,
                    }
                )

    print(f" 加载 {len(terms)} 个术语")
    return terms


def loadExistingQueries(filepath: str) -> list[dict[str, Any]]:
    if not os.path.exists(filepath):
        return []
    queries = loadJsonlFile(filepath)
    print(f" 加载现有查询: {len(queries)} 条")
    return queries


def mergeQueries(
    existing: list[dict[str, Any]], generated: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    seen_queries = {q["query"]: q for q in existing}
    merged = list(seen_queries.values())
    new_count = 0

    for query in generated:
        if query["query"] not in seen_queries:
            merged.append(query)
            seen_queries[query["query"]] = query
            new_count += 1

    print("\n 合并结果:")
    print(f"  - 现有: {len(existing)} 条")
    print(f"  - 新增: {new_count} 条")
    print(f"  - 总计: {len(merged)} 条")
    return merged


def saveQueries(queries: list[dict[str, Any]], filepath: str) -> None:
    by_subject: dict[str, int] = defaultdict(int)
    for query in queries:
        by_subject[query["subject"]] += 1

    print("\n 学科分布:")
    for subject, count in sorted(by_subject.items()):
        print(f"  - {subject}: {count} 条")

    saveJsonlFile(queries, filepath)
    print(f"\n 保存到: {filepath}")
