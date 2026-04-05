"""评测术语映射构建入口。"""

from __future__ import annotations

import json
import os

from core import config
from core.runners.evaluation import buildEvalTermMapping as mapping


def main(argv: list[str] | None = None) -> None:
    print("=" * 60)
    print("构建评测感知术语映射")
    print("=" * 60)

    queriesFullFile = os.path.join(config.EVALUATION_DIR, "queries_full.jsonl")
    queriesSmallFile = os.path.join(config.EVALUATION_DIR, "queries.jsonl")
    corpusFile = os.path.join(config.PROCESSED_DIR, "retrieval", "corpus.jsonl")
    termMappingFile = os.path.join(config.EVALUATION_DIR, "term_mapping.json")

    print("\n加载数据...")
    allQueries = []
    evalQueryKeys: set[tuple] = set()
    queryIndexBySubject: dict[tuple[str, str], int] = {}
    if os.path.exists(queriesSmallFile):
        loaded = mapping.loadQueries(queriesSmallFile)
        for q in loaded:
            key = (
                q["query"],
                q.get("subject", ""),
                frozenset(q.get("relevant_terms", [])),
            )
            if key not in evalQueryKeys:
                evalQueryKeys.add(key)
                queryIndexBySubject[(q["query"], q.get("subject", ""))] = len(
                    allQueries
                )
                allQueries.append(q)
        print(
            f"  加载查询文件: {queriesSmallFile} ({len(loaded)} 条，去重后保留 {len(allQueries)} 条)"
        )

    seenQueryKeys: set[tuple] = {(q["query"], q.get("subject", "")) for q in allQueries}
    if os.path.exists(queriesFullFile):
        loaded = mapping.loadQueries(queriesFullFile)
        added = 0
        merged = 0
        for q in loaded:
            key = (q["query"], q.get("subject", ""))
            if key not in seenQueryKeys:
                seenQueryKeys.add(key)
                queryIndexBySubject[key] = len(allQueries)
                allQueries.append(q)
                added += 1
            else:
                existingIndex = queryIndexBySubject[key]
                existingTerms = allQueries[existingIndex].get("relevant_terms", [])
                mergedTerms = list(
                    dict.fromkeys(existingTerms + q.get("relevant_terms", []))
                )
                if len(mergedTerms) != len(existingTerms):
                    allQueries[existingIndex]["relevant_terms"] = mergedTerms
                    merged += 1
        print(
            f"  加载查询文件: {queriesFullFile} ({len(loaded)} 条，新增补充 {added} 条，合并扩充 {merged} 条)"
        )
    queries = allQueries
    print(f"  合并后查询总数: {len(queries)} 条")

    corpusTerms = mapping.loadCorpusTerms(corpusFile)
    print(f"  加载语料库术语: {len(corpusTerms)} 个")

    totalRelevant = 0
    inCorpus = 0
    for q in queries:
        for term in q["relevant_terms"]:
            totalRelevant += 1
            if term in corpusTerms:
                inCorpus += 1
    print(
        f"  相关术语语料库覆盖率: {inCorpus}/{totalRelevant} ({inCorpus / totalRelevant:.1%})"
    )

    print("\n构建评测感知映射...")
    evalMapping = mapping.buildEvalTermMapping(queries, corpusTerms)

    print("\n构建子串匹配映射...")
    substringMapping = mapping.buildSubstringMapping(queries, corpusTerms)
    print(f"  子串映射键数: {len(substringMapping)}")

    print("\n合并映射...")
    combinedMapping = evalMapping.copy()
    print(f"  合并后映射键数: {len(combinedMapping)}")

    mapping.analyzeMapping(combinedMapping, queries, corpusTerms)

    os.makedirs(os.path.dirname(termMappingFile), exist_ok=True)
    with open(termMappingFile, "w", encoding="utf-8") as f:
        json.dump(combinedMapping, f, ensure_ascii=False, indent=2)

    print(f"\n 映射文件已保存: {termMappingFile}")

    print("\n=== 映射示例 ===")
    sampleQueries = [q["query"] for q in queries[:5]]
    for key in sampleQueries:
        if key in combinedMapping:
            print(f"  '{key}' -> {combinedMapping[key][:5]}")
        else:
            print(f"  '{key}' -> (无映射)")


if __name__ == "__main__":
    main()
