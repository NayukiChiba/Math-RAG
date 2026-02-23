"""
快速检索测试系统

功能：
1. 快速评估检索系统的召回率
2. 支持多种检索策略对比
3. 生成简洁的评测报告
4. 支持抽样测试（无需全量评测）

使用方法：
    # 快速测试（默认 20 条查询）
    python evaluation/quickEval.py

    # 指定测试数量
    python evaluation/quickEval.py --num-queries 50

    # 测试特定检索方法
    python evaluation/quickEval.py --methods bm25plus hybrid_plus

    # 关闭抽样，使用全部查询
    python evaluation/quickEval.py --all-queries
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

# 路径调整
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 终端 UTF-8 支持
if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

import config

# ==================== 指标计算函数 ====================


def calculateRecallAtK(results: list[dict], relevantTerms: list[str], k: int) -> float:
    """计算 Recall@K"""
    if not relevantTerms:
        return 0.0

    topkResults = results[:k]
    topkTerms = {r["term"] for r in topkResults}
    found = sum(1 for term in relevantTerms if term in topkTerms)

    return found / len(relevantTerms)


def calculateMRR(results: list[dict], relevantTerms: list[str]) -> float:
    """计算 MRR (Mean Reciprocal Rank)"""
    for rank, result in enumerate(results, 1):
        if result["term"] in relevantTerms:
            return 1.0 / rank
    return 0.0


def calculateMAP(results: list[dict], relevantTerms: list[str]) -> float:
    """计算 MAP (Mean Average Precision)"""
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


# ==================== 数据加载函数 ====================


def loadQueries(
    filepath: str, numQueries: int | None = None, allQueries: bool = False
) -> list[dict]:
    """
    加载查询集

    Args:
        filepath: 查询文件路径
        numQueries: 抽样数量
        allQueries: 使用全部查询

    Returns:
        查询列表
    """
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

    # 抽样
    if not allQueries and numQueries and numQueries < len(queries):
        print(f"📊 随机抽样 {numQueries} 条查询进行测试")
        random.seed(42)  # 固定随机种子，保证可复现
        queries = random.sample(queries, numQueries)

    return queries


def loadCorpus(filepath: str) -> list[dict]:
    """加载语料库"""
    corpus = []
    try:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                corpus.append(json.loads(line.strip()))
        print(f"✅ 加载了 {len(corpus)} 条语料")
    except FileNotFoundError:
        print(f"⚠️  语料文件不存在：{filepath}")
    return corpus


# ==================== 检索器加载函数 ====================


def createBM25Retriever(corpusFile: str, indexFile: str):
    """创建 BM25 检索器"""
    from retrieval.retrievalBM25 import BM25Retriever

    retriever = BM25Retriever(corpusFile, indexFile)
    if not retriever.loadIndex():
        print("⚠️  BM25 索引不存在，正在构建...")
        retriever.buildIndex()
        retriever.saveIndex()
    return retriever


def createBM25PlusRetriever(corpusFile: str, indexFile: str, termsFile: str):
    """创建 BM25+ 检索器"""
    from retrieval.retrievalBM25Plus import BM25PlusRetriever

    retriever = BM25PlusRetriever(corpusFile, indexFile, termsFile)
    if not retriever.loadIndex():
        print("⚠️  BM25+ 索引不存在，正在构建...")
        retriever.loadTermsMap()
        retriever.buildIndex()
        retriever.saveIndex()
    return retriever


def createVectorRetriever(
    corpusFile: str, indexFile: str, embeddingFile: str, model: str
):
    """创建向量检索器"""
    from retrieval.retrievalVector import VectorRetriever

    retriever = VectorRetriever(corpusFile, model, indexFile, embeddingFile)
    if not retriever.loadIndex():
        print("⚠️  向量索引不存在，正在构建...")
        retriever.buildIndex()
        retriever.saveIndex()
    return retriever


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


# ==================== 评测函数 ====================


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

        # 执行检索
        startTime = time.time()

        # 支持可调用函数作为 searchFunc
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

        # 计算指标
        metrics["recall@1"].append(calculateRecallAtK(results, relevantTerms, 1))
        metrics["recall@3"].append(calculateRecallAtK(results, relevantTerms, 3))
        metrics["recall@5"].append(calculateRecallAtK(results, relevantTerms, 5))
        metrics["recall@10"].append(calculateRecallAtK(results, relevantTerms, 10))
        metrics["mrr"].append(calculateMRR(results, relevantTerms))
        metrics["map"].append(calculateMAP(results, relevantTerms))
        metrics["ndcg@3"].append(calculateNDCG(results, relevantTerms, 3))
        metrics["ndcg@5"].append(calculateNDCG(results, relevantTerms, 5))
        metrics["ndcg@10"].append(calculateNDCG(results, relevantTerms, 10))

        # 进度显示
        if i % 10 == 0 or i == len(queries):
            print(f"  进度：{i}/{len(queries)} ({i / len(queries) * 100:.1f}%)")

    # 计算平均值
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

    # 打印摘要
    print(f"\n📈 {method} 评测摘要:")
    print(f"  Recall@5: {metrics['avg_metrics']['recall@5']:.2%}")
    print(f"  Recall@10: {metrics['avg_metrics']['recall@10']:.2%}")
    print(f"  MRR: {metrics['avg_metrics']['mrr']:.4f}")
    print(f"  nDCG@5: {metrics['avg_metrics']['ndcg@5']:.4f}")
    print(f"  平均查询时间：{metrics['avg_query_time']:.4f}s")

    return metrics


def runQuickEval(
    methods: list[str] | None = None,
    numQueries: int = 20,
    allQueries: bool = False,
    topK: int = 10,
) -> dict[str, Any]:
    """
    运行快速评测

    Args:
        methods: 评测方法列表
        numQueries: 抽样查询数量
        allQueries: 使用全部查询
        topK: 评估的 TopK 值

    Returns:
        评测报告
    """
    print("=" * 60)
    print("🚀 快速检索评测系统")
    print("=" * 60)

    # 默认方法
    if methods is None:
        methods = ["bm25", "bm25plus", "hybrid_plus"]

    # 加载数据
    # 注意：查询集在 data/evaluation 而非 data/processed/evaluation
    queriesFile = config.EVALUATION_DIR
    if not os.path.exists(queriesFile):
        queriesFile = os.path.join(config.PROCESSED_DIR, "evaluation")
    queriesFile = os.path.join(queriesFile, "queries.jsonl")
    corpusFile = os.path.join(config.PROCESSED_DIR, "retrieval", "corpus.jsonl")
    bm25Index = os.path.join(config.PROCESSED_DIR, "retrieval", "bm25_index.pkl")
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

    # 加载语料（用于检查）
    loadCorpus(corpusFile)

    # 评测每种方法
    allMetrics = {}

    for method in methods:
        if method == "bm25":
            retriever = createBM25Retriever(corpusFile, bm25Index)
            metrics = evaluateMethod("BM25", retriever, queries, topK)
        elif method == "bm25plus":
            retriever = createBM25PlusRetriever(corpusFile, bm25PlusIndex, termsFile)
            metrics = evaluateMethod(
                "BM25+", retriever, queries, topK, expandQuery=True
            )
        elif method == "vector":
            retriever = createVectorRetriever(
                corpusFile,
                vectorIndex,
                vectorEmbedding,
                "paraphrase-multilingual-MiniLM-L12-v2",
            )
            metrics = evaluateMethod("Vector", retriever, queries, topK)
        elif method == "hybrid_plus":
            # 优化：使用更高的 BM25 权重和更大的召回因子
            retriever = createHybridPlusRetriever(
                corpusFile,
                bm25PlusIndex,
                vectorIndex,
                vectorEmbedding,
                termsFile,
                "paraphrase-multilingual-MiniLM-L12-v2",
            )
            metrics = evaluateMethod(
                "Hybrid+",
                retriever,
                queries,
                topK,
                strategy="weighted",
                alpha=0.85,  # BM25 权重提高到 0.85
                beta=0.15,  # Vector 权重降低到 0.15
                recallFactor=8,  # 增加召回因子到 8
                expandQuery=False,  # 禁用查询扩展
            )
        elif method == "hybrid_rrf":
            retriever = createHybridPlusRetriever(
                corpusFile,
                bm25PlusIndex,
                vectorIndex,
                vectorEmbedding,
                termsFile,
                "paraphrase-multilingual-MiniLM-L12-v2",
            )
            metrics = evaluateMethod(
                "Hybrid+RRF",
                retriever,
                queries,
                topK,
                strategy="rrf",
                recallFactor=8,  # 增加召回因子
            )
        elif method == "advanced":
            # 高级检索：使用 RRF 融合策略 + 更高召回
            from retrieval.retrievalHybridPlus import HybridPlusRetriever

            retriever = HybridPlusRetriever(
                corpusFile,
                bm25PlusIndex,  # 使用 BM25+ 索引
                vectorIndex,
                vectorEmbedding,
                "paraphrase-multilingual-MiniLM-L12-v2",
                termsFile,
            )

            def advancedSearch(query, topK):
                return retriever.search(
                    query,
                    topK,
                    strategy="rrf",  # 使用 RRF 融合
                    recallFactor=8,  # 增加召回因子
                    expandQuery=True,
                )

            metrics = evaluateMethod(
                "Advanced", retriever, queries, topK, searchFunc=advancedSearch
            )
        else:
            print(f"⚠️  未知方法：{method}，跳过")
            continue

        allMetrics[method] = metrics

    # 生成对比报告
    print("\n" + "=" * 60)
    print("📊 评测对比报告")
    print("=" * 60)

    print(
        f"\n{'方法':<15} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'nDCG@5':>8} {'时间 (s)':>8}"
    )
    print("-" * 75)

    for _, metrics in allMetrics.items():
        avg = metrics["avg_metrics"]
        timeStr = (
            f"{metrics['avg_query_time']:.3f}" if "avg_query_time" in metrics else "N/A"
        )
        print(
            f"{metrics['method']:<15} "
            f"{avg['recall@1']:.2%}  "
            f"{avg['recall@3']:.2%}  "
            f"{avg['recall@5']:.2%}  "
            f"{avg['recall@10']:.2%}  "
            f"{avg['mrr']:.4f}  "
            f"{avg['ndcg@5']:.4f}  "
            f"{timeStr:>8}"
        )

    # 找出最佳方法
    bestMethod = max(
        allMetrics.keys(), key=lambda m: allMetrics[m]["avg_metrics"]["recall@5"]
    )
    print(
        f"\n🏆 Recall@5 最佳方法：{allMetrics[bestMethod]['method']} ({allMetrics[bestMethod]['avg_metrics']['recall@5']:.2%})"
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


# ==================== 主函数 ====================


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="快速检索评测系统")
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
            "bm25",
            "bm25plus",
            "vector",
            "hybrid_plus",
            "hybrid_rrf",
            "advanced",
        ],
        help="评测方法列表",
    )
    parser.add_argument(
        "--topk", type=int, default=10, help="评估的 TopK 值（默认 10）"
    )
    parser.add_argument("--output", type=str, help="输出报告文件路径")

    args = parser.parse_args()

    # 运行评测
    metrics = runQuickEval(
        methods=args.methods,
        numQueries=args.num_queries,
        allQueries=args.all_queries,
        topK=args.topk,
    )

    # 保存报告
    if metrics and args.output:
        saveReport(metrics, args.output)
    elif metrics:
        # 默认保存到 outputs/reports/quick_eval.json
        defaultOutput = os.path.join(
            config.PROJECT_ROOT, "outputs", "reports", "quick_eval.json"
        )
        saveReport(metrics, defaultOutput)


if __name__ == "__main__":
    main()
