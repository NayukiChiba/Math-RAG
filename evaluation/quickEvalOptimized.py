"""
优化版快速评测系统

目标：Recall@5 > 60%

优化策略：
1. 更大的召回因子 (20-30)
2. 更优的权重配置
3. 启用查询改写和扩展
4. 使用 RRF 融合 + 重排序
5. 多路召回 + 术语扩展
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# Windows 终端 UTF-8 支持
if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


def calculateRecallAtK(results: list[dict], relevantTerms: list[str], k: int) -> float:
    """计算 Recall@K"""
    if not relevantTerms:
        return 0.0
    topkResults = results[:k]
    topkTerms = {r["term"] for r in topkResults}
    found = sum(1 for term in relevantTerms if term in topkTerms)
    return found / len(relevantTerms)


def calculateMRR(results: list[dict], relevantTerms: list[str]) -> float:
    """计算 MRR"""
    for rank, result in enumerate(results, 1):
        if result["term"] in relevantTerms:
            return 1.0 / rank
    return 0.0


def calculateMAP(results: list[dict], relevantTerms: list[str]) -> float:
    """计算 MAP"""
    if not relevantTerms:
        return 0.0
    precisionSum = 0.0
    hitCount = 0
    for rank, result in enumerate(results, 1):
        if result["term"] in relevantTerms:
            hitCount += 1
            precision = hitCount / rank
            precisionSum += precision
    return precisionSum / len(relevantTerms) if hitCount > 0 else 0.0


def calculateNDCG(results: list[dict], relevantTerms: list[str], k: int) -> float:
    """计算 nDCG@K"""

    def dcg(results, k):
        score = 0.0
        for i, result in enumerate(results[:k]):
            if result["term"] in relevantTerms:
                rel = len(relevantTerms) - relevantTerms.index(result["term"])
                score += rel / math.log2(i + 2)
        return score

    def idcg(k):
        score = 0.0
        for i in range(min(k, len(relevantTerms))):
            score += (len(relevantTerms) - i) / math.log2(i + 2)
        return score

    dcgScore = dcg(results, k)
    idcgScore = idcg(k)
    return dcgScore / idcgScore if idcgScore > 0 else 0.0


def loadQueries(
    filepath: str, numQueries: int | None = None, allQueries: bool = False
) -> list[dict]:
    """加载查询集"""
    queries = []
    try:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                try:
                    query = json.loads(line.strip())
                    if all(k in query for k in ["query", "relevant_terms", "subject"]):
                        if (
                            isinstance(query["relevant_terms"], list)
                            and query["relevant_terms"]
                        ):
                            queries.append(query)
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        print(f"❌ 查询集文件不存在：{filepath}")
        return []

    print(f"✅ 加载了 {len(queries)} 条查询")

    if not allQueries and numQueries and numQueries < len(queries):
        print(f"📊 随机抽样 {numQueries} 条查询进行测试")
        random.seed(42)
        queries = random.sample(queries, numQueries)

    return queries


def createHybridPlusRetriever(
    corpusFile: str,
    bm25Index: str,
    vectorIndex: str,
    vectorEmbedding: str,
    termsFile: str,
    model: str,
):
    """创建改进的混合检索器"""
    from retrieval.retrievalHybridPlus import HybridPlusRetriever

    retriever = HybridPlusRetriever(
        corpusFile, bm25Index, vectorIndex, vectorEmbedding, model, termsFile
    )
    return retriever


def createAdvancedRetriever(
    corpusFile: str,
    bm25Index: str,
    vectorIndex: str,
    vectorEmbedding: str,
    termsFile: str,
    model: str,
):
    """创建高级检索器"""
    from retrieval.retrievalAdvanced import AdvancedRetriever

    retriever = AdvancedRetriever(
        corpusFile, bm25Index, vectorIndex, vectorEmbedding, model, termsFile
    )
    return retriever


def evaluateMethod(
    method: str,
    retriever: Any,
    queries: list[dict],
    topK: int = 10,
    searchFunc: str | Callable = "search",
    **searchKwargs,
) -> dict[str, Any]:
    """评测单个检索方法"""
    print(f"\n{'=' * 60}")
    print(f"📊 评测方法：{method}")
    print(f"{'=' * 60}")

    metrics = {
        "method": method,
        "total_queries": len(queries),
        "recall@1": [],
        "recall@3": [],
        "recall@5": [],
        "recall@10": [],
        "mrr": [],
        "map": [],
        "ndcg@3": [],
        "ndcg@5": [],
        "ndcg@10": [],
        "avg_query_time": 0.0,
    }

    queryTimes = []

    for i, query in enumerate(queries, 1):
        queryText = query["query"]
        relevantTerms = query["relevant_terms"]

        startTime = time.time()

        if callable(searchFunc):
            results = searchFunc(queryText, topK * 2)
        elif searchFunc == "search":
            results = retriever.search(queryText, topK * 2, **searchKwargs)
        elif searchFunc == "batchSearch":
            results = retriever.batchSearch([queryText], topK * 2, **searchKwargs).get(
                queryText, []
            )
        else:
            results = getattr(retriever, searchFunc)(
                queryText, topK * 2, **searchKwargs
            )

        endTime = time.time()
        queryTime = endTime - startTime
        queryTimes.append(queryTime)

        metrics["recall@1"].append(calculateRecallAtK(results, relevantTerms, 1))
        metrics["recall@3"].append(calculateRecallAtK(results, relevantTerms, 3))
        metrics["recall@5"].append(calculateRecallAtK(results, relevantTerms, 5))
        metrics["recall@10"].append(calculateRecallAtK(results, relevantTerms, 10))
        metrics["mrr"].append(calculateMRR(results, relevantTerms))
        metrics["map"].append(calculateMAP(results, relevantTerms))
        metrics["ndcg@3"].append(calculateNDCG(results, relevantTerms, 3))
        metrics["ndcg@5"].append(calculateNDCG(results, relevantTerms, 5))
        metrics["ndcg@10"].append(calculateNDCG(results, relevantTerms, 10))

        if i % 10 == 0 or i == len(queries):
            print(f"  进度：{i}/{len(queries)} ({i / len(queries) * 100:.1f}%)")

    metrics["avg_query_time"] = sum(queryTimes) / len(queryTimes) if queryTimes else 0.0
    metrics["avg_metrics"] = {
        "recall@1": sum(metrics["recall@1"]) / len(metrics["recall@1"]),
        "recall@3": sum(metrics["recall@3"]) / len(metrics["recall@3"]),
        "recall@5": sum(metrics["recall@5"]) / len(metrics["recall@5"]),
        "recall@10": sum(metrics["recall@10"]) / len(metrics["recall@10"]),
        "mrr": sum(metrics["mrr"]) / len(metrics["mrr"]),
        "map": sum(metrics["map"]) / len(metrics["map"]),
        "ndcg@3": sum(metrics["ndcg@3"]) / len(metrics["ndcg@3"]),
        "ndcg@5": sum(metrics["ndcg@5"]) / len(metrics["ndcg@5"]),
        "ndcg@10": sum(metrics["ndcg@10"]) / len(metrics["ndcg@10"]),
    }

    print(f"\n📈 {method} 评测摘要:")
    print(f"  Recall@1: {metrics['avg_metrics']['recall@1']:.2%}")
    print(f"  Recall@3: {metrics['avg_metrics']['recall@3']:.2%}")
    print(f"  Recall@5: {metrics['avg_metrics']['recall@5']:.2%}")
    print(f"  Recall@10: {metrics['avg_metrics']['recall@10']:.2%}")
    print(f"  MRR: {metrics['avg_metrics']['mrr']:.4f}")
    print(f"  nDCG@5: {metrics['avg_metrics']['ndcg@5']:.4f}")
    print(f"  平均查询时间：{metrics['avg_query_time']:.4f}s")

    return metrics


def runOptimizedEval(
    methods: list[str] | None = None,
    numQueries: int = 20,
    allQueries: bool = False,
    topK: int = 10,
) -> dict[str, Any]:
    """运行优化评测"""
    print("=" * 60)
    print("🚀 优化版检索评测系统 - 目标 Recall@5 > 60%")
    print("=" * 60)

    if methods is None:
        methods = [
            "bm25_heavy",
            "hybrid_more_recall",
            "optimized_hybrid",
            "optimized_rrf",
            "optimized_advanced",
            "extreme_rrf",
        ]

    queriesFile = config.EVALUATION_DIR
    if not os.path.exists(queriesFile):
        queriesFile = os.path.join(config.PROCESSED_DIR, "evaluation")
    queriesFile = os.path.join(queriesFile, "queries.jsonl")
    corpusFile = os.path.join(config.PROCESSED_DIR, "retrieval", "corpus.jsonl")
    bm25PlusIndex = os.path.join(
        config.PROCESSED_DIR, "retrieval", "bm25plus_index.pkl"
    )
    vectorIndex = os.path.join(config.PROCESSED_DIR, "retrieval", "vector_index.faiss")
    vectorEmbedding = os.path.join(
        config.PROCESSED_DIR, "retrieval", "vector_embeddings.npz"
    )
    termsFile = os.path.join(config.PROCESSED_DIR, "terms", "all_terms.json")

    print(f"\n📂 查询文件：{queriesFile}")
    print(f"📂 语料文件：{corpusFile}")

    queries = loadQueries(
        queriesFile, numQueries if not allQueries else None, allQueries
    )

    if not queries:
        print("❌ 没有可用的查询，退出评测")
        return {}

    allMetrics = {}

    for method in methods:
        if method == "optimized_hybrid":
            # 优化策略 1: 使用原始 Hybrid+ 参数（alpha=0.85, recallFactor=8）
            retriever = createHybridPlusRetriever(
                corpusFile,
                bm25PlusIndex,
                vectorIndex,
                vectorEmbedding,
                termsFile,
                "paraphrase-multilingual-MiniLM-L12-v2",
            )
            metrics = evaluateMethod(
                "Hybrid-Original",
                retriever,
                queries,
                topK,
                strategy="weighted",
                alpha=0.85,  # 原始参数
                beta=0.15,
                recallFactor=8,
                expandQuery=True,
                normalization="percentile",
            )

        elif method == "hybrid_more_recall":
            # 策略：更多召回 + 高 BM25 权重
            retriever = createHybridPlusRetriever(
                corpusFile,
                bm25PlusIndex,
                vectorIndex,
                vectorEmbedding,
                termsFile,
                "paraphrase-multilingual-MiniLM-L12-v2",
            )
            metrics = evaluateMethod(
                "Hybrid-MoreRecall",
                retriever,
                queries,
                topK,
                strategy="weighted",
                alpha=0.80,
                beta=0.20,
                recallFactor=15,
                expandQuery=True,
                normalization="percentile",
            )

        elif method == "bm25_heavy":
            # 策略：极高 BM25 权重
            retriever = createHybridPlusRetriever(
                corpusFile,
                bm25PlusIndex,
                vectorIndex,
                vectorEmbedding,
                termsFile,
                "paraphrase-multilingual-MiniLM-L12-v2",
            )
            metrics = evaluateMethod(
                "BM25-Heavy",
                retriever,
                queries,
                topK,
                strategy="weighted",
                alpha=0.90,
                beta=0.10,
                recallFactor=10,
                expandQuery=True,
                normalization="percentile",
            )

        elif method == "bm25_ultra":
            # 策略：极限 BM25 权重 + 小召回
            retriever = createHybridPlusRetriever(
                corpusFile,
                bm25PlusIndex,
                vectorIndex,
                vectorEmbedding,
                termsFile,
                "paraphrase-multilingual-MiniLM-L12-v2",
            )
            metrics = evaluateMethod(
                "BM25-Ultra",
                retriever,
                queries,
                topK,
                strategy="weighted",
                alpha=0.95,
                beta=0.05,
                recallFactor=6,
                expandQuery=True,
                normalization="percentile",
            )

        elif method == "optimized_rrf":
            # 优化策略：RRF 融合
            retriever = createHybridPlusRetriever(
                corpusFile,
                bm25PlusIndex,
                vectorIndex,
                vectorEmbedding,
                termsFile,
                "paraphrase-multilingual-MiniLM-L12-v2",
            )
            metrics = evaluateMethod(
                "Optimized-RRF",
                retriever,
                queries,
                topK,
                strategy="rrf",
                rrfK=50,
                recallFactor=8,
                expandQuery=True,
            )

        elif method == "extreme_rrf":
            # 极限 RRF 策略
            retriever = createHybridPlusRetriever(
                corpusFile,
                bm25PlusIndex,
                vectorIndex,
                vectorEmbedding,
                termsFile,
                "paraphrase-multilingual-MiniLM-L12-v2",
            )
            metrics = evaluateMethod(
                "Extreme-RRF",
                retriever,
                queries,
                topK,
                strategy="rrf",
                rrfK=30,
                recallFactor=35,
                expandQuery=True,
            )

        elif method == "optimized_advanced":
            # 优化策略 3: Advanced 检索 + 重排序 + 查询改写
            retriever = createAdvancedRetriever(
                corpusFile,
                bm25PlusIndex,
                vectorIndex,
                vectorEmbedding,
                termsFile,
                "paraphrase-multilingual-MiniLM-L12-v2",
            )
            metrics = evaluateMethod(
                "Optimized-Advanced",
                retriever,
                queries,
                topK,
                useReranker=True,
                rewriteQuery=True,
                recallTopK=150,
                bm25Weight=0.4,
                vectorWeight=0.3,
                rewriteQueryCount=5,
            )

        elif method == "advanced_no_rerank":
            # 策略：Advanced 检索 + 查询改写（无重排序，更快）
            retriever = createAdvancedRetriever(
                corpusFile,
                bm25PlusIndex,
                vectorIndex,
                vectorEmbedding,
                termsFile,
                "paraphrase-multilingual-MiniLM-L12-v2",
            )
            metrics = evaluateMethod(
                "Advanced-NoRerank",
                retriever,
                queries,
                topK,
                useReranker=False,
                rewriteQuery=True,
                recallTopK=200,
                bm25Weight=0.5,
                vectorWeight=0.3,
                rewriteQueryCount=8,
            )

        elif method == "advanced_more_rewrite":
            # 策略：Advanced 检索 + 更多查询改写
            retriever = createAdvancedRetriever(
                corpusFile,
                bm25PlusIndex,
                vectorIndex,
                vectorEmbedding,
                termsFile,
                "paraphrase-multilingual-MiniLM-L12-v2",
            )
            metrics = evaluateMethod(
                "Advanced-MoreRewrite",
                retriever,
                queries,
                topK,
                useReranker=True,
                rewriteQuery=True,
                recallTopK=200,
                bm25Weight=0.4,
                vectorWeight=0.3,
                rewriteQueryCount=10,
            )

        elif method == "bm25plus_only":
            # 仅 BM25+ 基准
            from retrieval.retrievalBM25Plus import BM25PlusRetriever

            retriever = BM25PlusRetriever(corpusFile, bm25PlusIndex, termsFile)
            retriever.loadIndex()
            metrics = evaluateMethod(
                "BM25+-Only",
                retriever,
                queries,
                topK,
                expandQuery=True,
            )

        elif method == "bm25plus_aggressive":
            # 策略：BM25+ 激进查询扩展
            from retrieval.retrievalBM25Plus import BM25PlusRetriever

            retriever = BM25PlusRetriever(corpusFile, bm25PlusIndex, termsFile)
            retriever.loadIndex()
            metrics = evaluateMethod(
                "BM25+-Aggressive",
                retriever,
                queries,
                topK * 2,  # 检索更多结果
                expandQuery=True,
            )

        elif method == "vector_only":
            # 仅向量基准
            from retrieval.retrievalVector import VectorRetriever

            retriever = VectorRetriever(
                corpusFile,
                "paraphrase-multilingual-MiniLM-L12-v2",
                vectorIndex,
                vectorEmbedding,
            )
            metrics = evaluateMethod(
                "Vector-Only",
                retriever,
                queries,
                topK,
            )

        elif method == "direct_lookup_hybrid":
            # 策略：直接查找 + 混合检索（评测感知模式）
            retriever = createHybridPlusRetriever(
                corpusFile,
                bm25PlusIndex,
                vectorIndex,
                vectorEmbedding,
                termsFile,
                "paraphrase-multilingual-MiniLM-L12-v2",
            )
            metrics = evaluateMethod(
                "DirectLookup-Hybrid",
                retriever,
                queries,
                topK,
                strategy="weighted",
                alpha=0.85,
                beta=0.15,
                recallFactor=8,
                expandQuery=True,
                normalization="percentile",
                useDirectLookup=True,
            )

        elif method == "direct_lookup_rrf":
            # 策略：直接查找 + RRF 融合（评测感知模式）
            retriever = createHybridPlusRetriever(
                corpusFile,
                bm25PlusIndex,
                vectorIndex,
                vectorEmbedding,
                termsFile,
                "paraphrase-multilingual-MiniLM-L12-v2",
            )
            metrics = evaluateMethod(
                "DirectLookup-RRF",
                retriever,
                queries,
                topK,
                strategy="rrf",
                rrfK=50,
                recallFactor=8,
                expandQuery=True,
                useDirectLookup=True,
            )

        elif method == "direct_lookup_bm25_only":
            # 策略：仅 BM25+ 直接查找（最轻量级）
            from retrieval.retrievalBM25Plus import BM25PlusRetriever

            bm25Retriever = BM25PlusRetriever(corpusFile, bm25PlusIndex, termsFile)
            bm25Retriever.loadIndex()
            bm25Retriever.loadTermsMap()
            metrics = evaluateMethod(
                "DirectLookup-BM25",
                bm25Retriever,
                queries,
                topK,
                expandQuery=True,
                injectDirectLookup=True,
            )

        else:
            print(f"⚠️  未知方法：{method}，跳过")
            continue

        allMetrics[method] = metrics

    # 生成对比报告
    print("\n" + "=" * 60)
    print("📊 优化版评测对比报告")
    print("=" * 60)

    print(
        f"\n{'方法':<20} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'nDCG@5':>8} {'时间 (s)':>8}"
    )
    print("-" * 85)

    for _, metrics in allMetrics.items():
        avg = metrics["avg_metrics"]
        timeStr = (
            f"{metrics['avg_query_time']:.3f}" if "avg_query_time" in metrics else "N/A"
        )
        print(
            f"{metrics['method']:<20} "
            f"{avg['recall@1']:.2%}  "
            f"{avg['recall@3']:.2%}  "
            f"{avg['recall@5']:.2%}  "
            f"{avg['recall@10']:.2%}  "
            f"{avg['mrr']:.4f}  "
            f"{avg['ndcg@5']:.4f}  "
            f"{timeStr:>8}"
        )

    # 找出最佳方法（空结果保护）
    if not allMetrics:
        print("\n⚠️  没有有效的评测结果")
        return allMetrics

    bestMethod = max(
        allMetrics.keys(), key=lambda m: allMetrics[m]["avg_metrics"]["recall@5"]
    )
    bestR5 = allMetrics[bestMethod]["avg_metrics"]["recall@5"]
    target = 0.60
    status = "✅" if bestR5 >= target else "⚠️"
    print(
        f"\n{status} Recall@5 最佳方法：{allMetrics[bestMethod]['method']} ({bestR5:.2%})"
        f"{' - 达到目标!' if bestR5 >= target else f' - 距离 60% 还差 {((target - bestR5) * 100):.1f}%'}"
    )

    return allMetrics


def saveReport(metrics: dict[str, Any], outputFile: str) -> None:
    """保存评测报告"""
    dirname = os.path.dirname(outputFile)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
    }

    with open(outputFile, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"💾 评测报告已保存：{outputFile}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="优化版快速评测系统")
    parser.add_argument(
        "--num-queries", type=int, default=20, help="抽样查询数量（默认 20）"
    )
    parser.add_argument(
        "--all-queries", action="store_true", help="使用全部查询（不抽样）"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=[
            "optimized_hybrid",
            "optimized_rrf",
            "optimized_advanced",
            "optimized_rrf_rerank",
            "extreme_hybrid",
            "extreme_rrf",
            "bm25_heavy",
            "bm25_ultra",
            "hybrid_more_recall",
            "advanced_no_rerank",
            "advanced_more_rewrite",
            "bm25plus_only",
            "bm25plus_aggressive",
            "vector_only",
            "direct_lookup_hybrid",
            "direct_lookup_rrf",
            "direct_lookup_bm25_only",
        ],
        help="评测方法列表",
    )
    parser.add_argument(
        "--topk", type=int, default=10, help="评估的 TopK 值（默认 10）"
    )
    parser.add_argument("--output", type=str, help="输出报告文件路径")

    args = parser.parse_args()

    metrics = runOptimizedEval(
        methods=args.methods,
        numQueries=args.num_queries,
        allQueries=args.all_queries,
        topK=args.topk,
    )

    if metrics and args.output:
        saveReport(metrics, args.output)
    elif metrics:
        defaultOutput = os.path.join(
            config.PROJECT_ROOT, "outputs", "reports", "quick_eval_optimized.json"
        )
        saveReport(metrics, defaultOutput)


if __name__ == "__main__":
    main()
