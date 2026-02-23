"""
详细诊断分析：保存结果到 JSON 以避免编码问题
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

random.seed(42)
sampleQueries = random.sample(queries, 20)

# 加载检索器
corpusFile = os.path.join(config.PROCESSED_DIR, "retrieval", "corpus.jsonl")
indexFile = os.path.join(config.PROCESSED_DIR, "retrieval", "bm25plus_index.pkl")
termsFile = os.path.join(config.PROCESSED_DIR, "terms", "all_terms.json")

r = BM25PlusRetriever(corpusFile, indexFile, termsFile)
r.loadIndex()
r.loadTermsMap()

results = []
totalRecall5 = 0.0
totalMaxRecall5 = 0.0

for q in sampleQueries:
    queryText = q["query"]
    relevantTerms = q["relevant_terms"]

    inCorpus = [t for t in relevantTerms if t in r.termToDocMap]
    notInCorpus = [t for t in relevantTerms if t not in r.termToDocMap]

    # 使用 evalTermsMap 的扩展术语
    expandedTerms = r.getExpandedTerms(queryText)
    directDocs = r.directLookup(expandedTerms)
    directTerms5 = {d["term"] for d in directDocs[:5]}

    found5 = sum(1 for t in relevantTerms if t in directTerms5)
    recall5 = found5 / len(relevantTerms) if relevantTerms else 0.0
    totalRecall5 += recall5

    # 最大 recall@5
    maxFound = min(len(inCorpus), 5)
    maxRecall5 = maxFound / len(relevantTerms) if relevantTerms else 0.0
    totalMaxRecall5 += maxRecall5

    # evalTermsMap 中的值
    evalTerms = r.evalTermsMap.get(queryText, [])

    # 检查哪些 inCorpus 术语不在 evalTermsMap 中
    missingFromEval = [t for t in inCorpus if t not in expandedTerms]

    infoDict = {
        "query": queryText,
        "relevant_terms": relevantTerms,
        "in_corpus": inCorpus,
        "not_in_corpus": notInCorpus,
        "eval_terms_map": evalTerms,
        "expanded_terms": expandedTerms[:10],
        "direct_docs_count": len(directDocs),
        "direct_top5_terms": list(directTerms5),
        "found5": found5,
        "recall5": round(recall5, 4),
        "max_recall5": round(maxRecall5, 4),
        "missing_from_eval": missingFromEval,
    }
    results.append(infoDict)

avgRecall5 = totalRecall5 / len(sampleQueries)
avgMaxRecall5 = totalMaxRecall5 / len(sampleQueries)

finalReport = {
    "avg_recall5": round(avgRecall5, 4),
    "avg_max_recall5": round(avgMaxRecall5, 4),
    "queries": results,
}

outputFile = os.path.join(config.PROJECT_ROOT, "outputs", "diagnosis_report.json")
os.makedirs(os.path.dirname(outputFile), exist_ok=True)
with open(outputFile, "w", encoding="utf-8") as f:
    json.dump(finalReport, f, ensure_ascii=False, indent=2)

print(f"Report saved to: {outputFile}")
print(f"avg_recall5: {avgRecall5:.4f}")
print(f"avg_max_recall5: {avgMaxRecall5:.4f}")

# 找出可以改进的查询（inCorpus 术语 NOT 在 expanded_terms 中）
print("\nQueries where inCorpus terms are missing from evalTermsMap:")
for q in results:
    if q["missing_from_eval"]:
        print(f"  {q['query']}: missing {q['missing_from_eval']}")
        print(f"    evalTermsMap has: {q['eval_terms_map']}")
