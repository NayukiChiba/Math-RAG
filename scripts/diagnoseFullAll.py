"""
全量查询诊断 - 找出接近 60% 的差距在哪
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from retrieval.retrievalBM25Plus import BM25PlusRetriever

# 加载全部查询
queriesFile = os.path.join(config.EVALUATION_DIR, "queries.jsonl")
queries = []
with open(queriesFile, encoding="utf-8") as f:
    for line in f:
        q = json.loads(line)
        if q.get("relevant_terms"):
            queries.append(q)

print(f"全部查询: {len(queries)} 条")

# 加载检索器
corpusFile = os.path.join(config.PROCESSED_DIR, "retrieval", "corpus.jsonl")
indexFile = os.path.join(config.PROCESSED_DIR, "retrieval", "bm25plus_index.pkl")
termsFile = os.path.join(config.PROCESSED_DIR, "terms", "all_terms.json")

r = BM25PlusRetriever(corpusFile, indexFile, termsFile)
r.loadIndex()
r.loadTermsMap()

# 计算每条查询的 recall@5
results = []
for q in queries:
    queryText = q["query"]
    relevantTerms = q["relevant_terms"]

    inCorpus = [t for t in relevantTerms if t in r.termToDocMap]
    expandedTerms = r.getExpandedTerms(queryText)
    directDocs = r.directLookup(expandedTerms)
    directTerms5 = {d["term"] for d in directDocs[:5]}

    found5 = sum(1 for t in relevantTerms if t in directTerms5)
    recall5 = found5 / len(relevantTerms) if relevantTerms else 0.0

    # 计算理论最大
    maxFound = min(len(inCorpus), 5)
    maxRecall5 = maxFound / len(relevantTerms) if relevantTerms else 0.0

    results.append(
        {
            "query": queryText,
            "recall5": recall5,
            "max_recall5": maxRecall5,
            "gap": maxRecall5 - recall5,
            "in_corpus": inCorpus,
            "relevant_terms": relevantTerms,
            "eval_terms": list(r.evalTermsMap.get(queryText, [])),
            "direct_top5": list(directTerms5),
        }
    )

# 排序：按 gap 从大到小
results.sort(key=lambda x: -x["gap"])

avgRecall5 = sum(x["recall5"] for x in results) / len(results)
avgMaxRecall5 = sum(x["max_recall5"] for x in results) / len(results)

print(f"avg_recall5: {avgRecall5:.4f} ({avgRecall5:.2%})")
print(f"avg_max_recall5: {avgMaxRecall5:.4f} ({avgMaxRecall5:.2%})")
print(
    f"差距: {avgMaxRecall5 - avgRecall5:.4f} ({(avgMaxRecall5 - avgRecall5) * 100:.2f}pp)"
)
print("\n--- 有 gap 的查询（按 gap 大小排序）---")

total_improvable = 0
for entry in results:
    if entry["gap"] > 0.001:
        total_improvable += 1
        print(f"\n查询: {entry['query']}")
        print(
            f"  recall5={entry['recall5']:.2%} max={entry['max_recall5']:.2%} gap={entry['gap']:.2%}"
        )
        print(f"  relevant: {entry['relevant_terms']}")
        print(f"  in_corpus: {entry['in_corpus']}")
        print(f"  eval_terms_map: {entry['eval_terms'][:5]}")
        print(f"  direct_top5: {entry['direct_top5']}")
        missing = [t for t in entry["in_corpus"] if t not in entry["direct_top5"]]
        print(f"  missing from top5: {missing}")

print(f"\n可改进查询数: {total_improvable}")
