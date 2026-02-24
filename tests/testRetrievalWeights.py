"""
快速检索权重对比测试（不启动 Qwen 模型）
对比 hybrid alpha=0.5/0.5 vs 0.7/0.3 vs RRF 的 Recall@5 和 MRR
"""

import json
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from retrieval.retrievers import HybridRetriever

retrievalDir = os.path.join(config.PROCESSED_DIR, "retrieval")

print("初始化混合检索器...")
retriever = HybridRetriever(
    os.path.join(retrievalDir, "corpus.jsonl"),
    os.path.join(retrievalDir, "bm25_index.pkl"),
    os.path.join(retrievalDir, "vector_index.faiss"),
    os.path.join(retrievalDir, "vector_embeddings.npz"),
)

# 加载有标注的评测查询
queryFile = os.path.join(config.PROJECT_ROOT, "data", "evaluation", "queries.jsonl")
queries = []
with open(queryFile, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            q = json.loads(line)
            if q.get("query") and q.get("relevant_terms"):
                queries.append(q)
print(f"有效标注查询: {len(queries)} 条\n")


def recallAtK(results, relevantTerms, k):
    if not relevantTerms:
        return 0.0
    topTerms = {r.get("term", "") for r in results[:k]}
    return sum(1 for t in relevantTerms if t in topTerms) / len(relevantTerms)


def calcMrr(results, relevantTerms):
    for rank, r in enumerate(results, 1):
        if r.get("term", "") in relevantTerms:
            return 1.0 / rank
    return 0.0


# 三种配置对比
testConfigs = [
    ("hybrid-0.5/0.5 (旧)", "weighted", 0.5, 0.5),
    ("hybrid-0.7/0.3 (新)", "weighted", 0.7, 0.3),
    ("hybrid-rrf      (新)", "rrf", 0.0, 0.0),
]

print(f"{'策略':<24} {'Recall@5':>10} {'Recall@3':>10} {'MRR':>10}")
print("-" * 56)

for name, strategy, alpha, beta in testConfigs:
    r5List, r3List, mrrList = [], [], []
    for q in queries:
        if strategy == "rrf":
            results = retriever.search(
                q["query"], topK=10, strategy="rrf", rrfK=60, verbose=False
            )
        else:
            results = retriever.search(
                q["query"],
                topK=10,
                strategy="weighted",
                alpha=alpha,
                beta=beta,
                verbose=False,
            )
        rel = q["relevant_terms"]
        r5List.append(recallAtK(results, rel, 5))
        r3List.append(recallAtK(results, rel, 3))
        mrrList.append(calcMrr(results, rel))

    avgR5 = sum(r5List) / len(r5List)
    avgR3 = sum(r3List) / len(r3List)
    avgMrr = sum(mrrList) / len(mrrList)
    print(f"{name:<24} {avgR5:>10.4f} {avgR3:>10.4f} {avgMrr:>10.4f}")

print("\n完成。")
