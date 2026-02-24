"""
测试直接查找功能
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from retrieval.retrievalBM25Plus import BM25PlusRetriever

corpusFile = os.path.join(config.PROCESSED_DIR, "retrieval", "corpus.jsonl")
indexFile = os.path.join(config.PROCESSED_DIR, "retrieval", "bm25plus_index.pkl")
termsFile = os.path.join(config.PROCESSED_DIR, "terms", "all_terms.json")

print("=== 测试直接查找功能 ===")

r = BM25PlusRetriever(corpusFile, indexFile, termsFile)
loaded = r.loadIndex()
r.loadTermsMap()

print(f"索引加载: {loaded}")
print(f"termToDocMap 大小: {len(r.termToDocMap)}")
print(f"termsMap 大小: {len(r.termsMap)}")

# 测试扩展术语
testQueries = ["函数连续性", "收敛数列", "泰勒公式", "连续", "导数"]
for query in testQueries:
    expandedTerms = r.getExpandedTerms(query)
    docs = r.directLookup(expandedTerms)
    print(f"\n查询: '{query}'")
    print(f"  扩展术语: {expandedTerms[:5]}")
    print(f"  找到文档数: {len(docs)}")
    if docs:
        print(f"  术语: {[d['term'] for d in docs[:5]]}")

# 测试 search with injectDirectLookup
print("\n=== 测试 search with injectDirectLookup ===")
results = r.search("函数连续性", topK=5, expandQuery=True, injectDirectLookup=True)
print("搜索 '函数连续性' (injectDirectLookup=True):")
for result in results:
    print(
        f"  Rank {result['rank']}: {result['term']} ({result.get('lookup_type', 'bm25')})"
    )
