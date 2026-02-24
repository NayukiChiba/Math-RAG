"""
诊断分析：查看每个查询的直接查找结果和失败原因

分析 20 条抽样查询的情况，找出为何无法达到 60% Recall@5
"""

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from retrieval.retrievalBM25Plus import BM25PlusRetriever

# 加载查询
queriesFile = os.path.join(config.EVALUATION_DIR, "queries.jsonl")
queries = []
with open(queriesFile, encoding="utf-8") as f:
    for line in f:
        q = json.loads(line)
        if q.get("relevant_terms"):
            queries.append(q)

# 使用相同的随机种子抽样 20 条
random.seed(42)
sampleQueries = random.sample(queries, 20)

# 加载检索器
corpusFile = os.path.join(config.PROCESSED_DIR, "retrieval", "corpus.jsonl")
indexFile = os.path.join(config.PROCESSED_DIR, "retrieval", "bm25plus_index.pkl")
termsFile = os.path.join(config.PROCESSED_DIR, "terms", "all_terms.json")

r = BM25PlusRetriever(corpusFile, indexFile, termsFile)
r.loadIndex()
r.loadTermsMap()

print("=" * 70)
print("诊断分析：20 条抽样查询的直接查找情况")
print("=" * 70)

totalQueries = 0
totalRecall5 = 0.0
perfectQueries = 0  # Recall@5 = 1.0 的查询
missingCoverage = []  # 有相关术语不在语料库中

for q in sampleQueries:
    queryText = q["query"]
    relevantTerms = q["relevant_terms"]
    totalQueries += 1

    # 1. 检查相关术语的语料库覆盖情况
    inCorpus = [t for t in relevantTerms if t in r.termToDocMap]
    notInCorpus = [t for t in relevantTerms if t not in r.termToDocMap]

    # 2. 通过 termsMap 获取扩展术语
    expandedTerms = r.getExpandedTerms(queryText)
    directDocs = r.directLookup(expandedTerms)
    directTerms = {d["term"] for d in directDocs}

    # 3. 计算通过直接查找能找到的相关术语
    foundByDirect = [t for t in relevantTerms if t in directTerms]

    # 4. 模拟 Recall@5 计算（直接查找最多 5 个）
    # 假设直接查找文档优先，然后是 BM25 结果
    if len(directDocs) > 0:
        topTerms = {d["term"] for d in directDocs[:5]}
    else:
        topTerms = set()
    found5 = sum(1 for t in relevantTerms if t in topTerms)
    recall5 = found5 / len(relevantTerms) if relevantTerms else 0.0
    totalRecall5 += recall5

    if recall5 >= 1.0:
        perfectQueries += 1

    # 打印分析
    status = "✓" if recall5 >= 0.6 else "✗"
    print(f"\n{status} 查询: '{queryText}'")
    print(f"   总相关术语: {len(relevantTerms)}")
    print(f"   在语料库中: {len(inCorpus)} - {inCorpus[:3]}")
    if notInCorpus:
        print(f"   不在语料库: {len(notInCorpus)} - {notInCorpus}")
    print(f"   直接找到:   {len(directDocs)} 个文档")
    print(f"   Top-5覆盖: {found5}/{len(relevantTerms)} ({recall5:.1%})")

    # 检查 termsMap 中是否有此查询的映射
    if queryText in r.termsMap:
        print(f"   termsMap中: {r.termsMap[queryText][:3]}...")
    else:
        print("   termsMap中: 无此查询映射!")
        missingCoverage.append(queryText)

    # 如果直接查找找到的相关术语少于在语料库中的相关术语，说明映射不完整
    missingFromDirect = [t for t in inCorpus if t not in directTerms]
    if missingFromDirect:
        print(f"   映射缺失:   {missingFromDirect}")

avgRecall5 = totalRecall5 / totalQueries
print("\n" + "=" * 70)
print(f"平均 Recall@5（理论直接查找）: {avgRecall5:.2%}")
print(f"完美查询数（Recall@5=1.0）: {perfectQueries}/{totalQueries}")
print(f"termsMap 缺失的查询: {missingCoverage}")
print("=" * 70)

# 计算理论上限
totalMaxRecall5 = 0.0
for q in sampleQueries:
    relevantTerms = q["relevant_terms"]
    inCorpus = [t for t in relevantTerms if t in r.termToDocMap]
    # 假设所有语料库中的相关术语都能找到并排在 top-5 中
    maxFound = min(len(inCorpus), 5)
    maxRecall5 = maxFound / len(relevantTerms) if relevantTerms else 0.0
    totalMaxRecall5 += maxRecall5

avgMaxRecall5 = totalMaxRecall5 / totalQueries
print(f"\n理论最大 Recall@5（语料库限制）: {avgMaxRecall5:.2%}")
print("（假设所有在语料库中的相关术语都恰好在 Top-5 中）")
