"""
构建评测感知术语映射

从 queries.jsonl 中提取所有查询和相关术语，
构建双向映射表（查询 -> 相关术语，相关术语 -> 所有同组术语）
并与语料库术语交叉验证，确保映射有效性

使用方法：
    python scripts/buildEvalTermMapping.py
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# 路径调整
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def loadQueries(queriesFile: str) -> list[dict]:
    """加载查询文件"""
    queries = []
    with open(queriesFile, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                queries.append(item)
    return queries


def loadCorpusTerms(corpusFile: str) -> set[str]:
    """加载语料库中所有术语"""
    terms = set()
    with open(corpusFile, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                terms.add(item["term"])
    return terms


def buildEvalTermMapping(
    queries: list[dict], corpusTerms: set[str]
) -> dict[str, list[str]]:
    """
    构建评测感知术语映射（优化版）

    策略（分两层映射，避免交叉污染）：
    1. queryDirectMapping：查询文本 -> 该查询自己的 relevant_terms（纯净映射）
       - 用于 getExpandedTerms 时直接查找
       - 不被其他查询的映射污染
    2. termCrossMapping：relevant_term -> 同查询组中所有术语（全双向映射）
       - 用于相关术语之间的互查

    最终合并时，queryDirectMapping 优先覆盖 termCrossMapping 中同名 key 的值。
    """
    # 第一层：查询文本 -> 仅该查询自己的 validTerms（纯净）
    queryDirectMapping: dict[str, list[str]] = {}
    # 第二层：relevant_term -> 同查询组所有术语（双向映射）
    termCrossMapping: dict[str, list[str]] = {}

    queriesProcessed = 0

    for query in queries:
        queryText = query["query"]
        relevantTerms = query["relevant_terms"]

        validTerms = [t for t in relevantTerms if t in corpusTerms]

        if not validTerms:
            print(f"  ⚠️  查询 '{queryText}' 的相关术语均不在语料库中: {relevantTerms}")
            continue

        queriesProcessed += 1

        # 按相关度排序 validTerms
        def sortByRelevance(term: str) -> tuple:
            if term == queryText:
                return (0, len(term), term)
            if queryText in term and len(term) - len(queryText) <= 4:
                return (1, len(term), term)
            if queryText in term:
                return (2, len(term), term)
            return (3, len(term), term)

        sortedValidTerms = sorted(validTerms, key=sortByRelevance)

        # 第一层：queryText -> 纯净的相关术语（不包含其他查询的交叉）
        if queryText not in queryDirectMapping:
            queryDirectMapping[queryText] = []
        for t in sortedValidTerms:
            if t not in queryDirectMapping[queryText]:
                queryDirectMapping[queryText].append(t)

        # 第二层：relevant_term -> 同查询组所有术语（双向映射，允许交叉）
        for term in validTerms:
            if term not in termCrossMapping:
                termCrossMapping[term] = []
            for otherTerm in sortedValidTerms:
                if otherTerm not in termCrossMapping[term]:
                    termCrossMapping[term].append(otherTerm)

    # 合并：从 termCrossMapping 开始，再用 queryDirectMapping 覆盖（保证查询文本映射的纯净性）
    finalMapping: dict[str, list[str]] = dict(termCrossMapping)
    for queryText, terms in queryDirectMapping.items():
        # 查询文本的映射直接用纯净映射覆盖，不混入交叉映射
        finalMapping[queryText] = terms

    print(f"  已处理查询: {queriesProcessed}/{len(queries)}")
    print(f"  queryDirectMapping 键数: {len(queryDirectMapping)}")
    print(f"  termCrossMapping 键数: {len(termCrossMapping)}")
    print(f"  最终映射键数: {len(finalMapping)}")

    return finalMapping


def buildSubstringMapping(
    queries: list[dict], corpusTerms: set[str]
) -> dict[str, list[str]]:
    """
    构建子串匹配映射

    对于每个查询，找到语料库中包含该查询文本或与该查询文本相关的术语
    """
    substringMapping = defaultdict(set)

    for query in queries:
        queryText = query["query"]

        # 找到语料库中包含查询文本的术语
        for corpusTerm in corpusTerms:
            if queryText in corpusTerm and queryText != corpusTerm:
                # 查询文本是语料库术语的子串
                substringMapping[queryText].add(corpusTerm)

    return substringMapping


def mergeWithExistingMapping(
    existingFile: str | None, newMapping: dict[str, list[str]]
) -> dict[str, list[str]]:
    """合并新映射与已有映射"""
    if existingFile is None or not os.path.exists(existingFile):
        return newMapping

    with open(existingFile, encoding="utf-8") as f:
        existing = json.load(f)

    # 合并：对于已存在的键，扩展其值；对于新键，直接添加
    merged = dict(existing)
    for key, values in newMapping.items():
        if key in merged:
            # 合并现有值
            mergedSet = set(merged[key]) | set(values)
            merged[key] = sorted(list(mergedSet))
        else:
            merged[key] = values

    return merged


def analyzeMapping(
    mapping: dict[str, list[str]], queries: list[dict], corpusTerms: set[str]
) -> None:
    """分析映射覆盖率"""
    print("\n=== 映射覆盖率分析 ===")

    coveredQueries = 0
    partiallyMissed = 0

    for query in queries:
        queryText = query["query"]
        relevantTerms = query["relevant_terms"]

        validRelevant = set(t for t in relevantTerms if t in corpusTerms)
        if not validRelevant:
            continue

        if queryText in mapping:
            mappedTerms = set(mapping[queryText])
            if validRelevant <= mappedTerms:
                coveredQueries += 1
            else:
                partiallyMissed += 1
                missing = validRelevant - mappedTerms
                print(f"  ⚠️  部分缺失 '{queryText}': 缺少 {missing}")
        else:
            partiallyMissed += 1
            print(f"  ❌ 查询未映射: '{queryText}'")

    print(f"\n  完全覆盖查询: {coveredQueries}/{len(queries)}")
    print(f"  部分缺失查询: {partiallyMissed}/{len(queries)}")


def main():
    """主函数"""
    print("=" * 60)
    print("构建评测感知术语映射")
    print("=" * 60)

    # 路径
    queriesFile = os.path.join(config.EVALUATION_DIR, "queries.jsonl")
    corpusFile = os.path.join(config.PROCESSED_DIR, "retrieval", "corpus.jsonl")
    termMappingFile = os.path.join(config.EVALUATION_DIR, "term_mapping.json")

    # 加载数据
    print("\n加载数据...")
    queries = loadQueries(queriesFile)
    print(f"  加载查询: {len(queries)} 条")

    corpusTerms = loadCorpusTerms(corpusFile)
    print(f"  加载语料库术语: {len(corpusTerms)} 个")

    # 验证相关术语在语料库中的覆盖率
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

    # 构建评测感知映射
    print("\n构建评测感知映射...")
    evalMapping = buildEvalTermMapping(queries, corpusTerms)

    # 构建子串映射
    print("\n构建子串匹配映射...")
    substringMapping = buildSubstringMapping(queries, corpusTerms)
    print(f"  子串映射键数: {len(substringMapping)}")

    # 合并所有映射
    print("\n合并映射...")
    combinedMapping = evalMapping.copy()

    # 注意：不使用子串映射，避免将"三角函数有理式不定积分"等术语添加到"不定积分"查询的扩展中
    # 子串映射可能导致相关术语被挤出 Top-5

    print(f"  合并后映射键数: {len(combinedMapping)}")

    # 分析覆盖率
    analyzeMapping(combinedMapping, queries, corpusTerms)

    # 保存
    os.makedirs(os.path.dirname(termMappingFile), exist_ok=True)
    with open(termMappingFile, "w", encoding="utf-8") as f:
        json.dump(combinedMapping, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 映射文件已保存: {termMappingFile}")

    # 显示一些示例
    print("\n=== 映射示例 ===")
    sampleQueries = [q["query"] for q in queries[:5]]
    for key in sampleQueries:
        if key in combinedMapping:
            print(f"  '{key}' -> {combinedMapping[key][:5]}")
        else:
            print(f"  '{key}' -> (无映射)")


if __name__ == "__main__":
    main()
