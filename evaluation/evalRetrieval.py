"""
检索评测脚本

功能：
1. 加载评测查询集
2. 调用多种检索方法（BM25、Vector、Hybrid）
3. 计算评测指标：Recall@K、MRR、nDCG@K、MAP
4. 生成评测报告和对比图表

评测指标说明：
- Recall@K：前 K 个结果中找到的相关文档比例
- MRR (Mean Reciprocal Rank)：第一个相关文档排名倒数的平均值
- nDCG@K (Normalized Discounted Cumulative Gain)：考虑排名位置的相关性评分
- MAP (Mean Average Precision)：所有相关文档的 Precision 平均值

使用方法：
    python evaluation/evalRetrieval.py
    python evaluation/evalRetrieval.py --methods bm25 vector
    python evaluation/evalRetrieval.py --topk 20
    python evaluation/evalRetrieval.py --visualize
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# 路径调整
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def loadQueries(filepath: str) -> list[dict[str, Any]]:
    """
    加载评测查询集

    Args:
        filepath: 查询集文件路径（JSONL 格式）

    Returns:
        查询列表
    """
    queries = []
    try:
        with open(filepath, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                try:
                    query = json.loads(line.strip())
                    # 验证必需字段
                    if not all(
                        k in query for k in ["query", "relevant_terms", "subject"]
                    ):
                        print(f"⚠️  第 {i} 行缺少必需字段，跳过")
                        continue
                    if (
                        not isinstance(query["relevant_terms"], list)
                        or not query["relevant_terms"]
                    ):
                        print(f"⚠️  第 {i} 行 relevant_terms 格式错误，跳过")
                        continue
                    queries.append(query)
                except json.JSONDecodeError as e:
                    print(f"⚠️  第 {i} 行 JSON 解析失败: {e}")
        print(f"✅ 加载了 {len(queries)} 条查询")
        return queries
    except FileNotFoundError:
        print(f"❌ 查询集文件不存在: {filepath}")
        return []
    except Exception as e:
        print(f"❌ 加载查询集失败: {e}")
        return []


def calculateRecallAtK(results: list[dict], relevantTerms: list[str], k: int) -> float:
    """
    计算 Recall@K

    Args:
        results: 检索结果列表（每项包含 term 字段）
        relevantTerms: 相关术语列表
        k: TopK 阈值

    Returns:
        Recall@K 值（0-1）
    """
    if not relevantTerms:
        return 0.0

    topkResults = results[:k]
    topkTerms = {r["term"] for r in topkResults}
    found = sum(1 for term in relevantTerms if term in topkTerms)

    return found / len(relevantTerms)


def calculateMRR(results: list[dict], relevantTerms: list[str]) -> float:
    """
    计算 MRR (Mean Reciprocal Rank)

    Args:
        results: 检索结果列表
        relevantTerms: 相关术语列表

    Returns:
        MRR 值（0-1）
    """
    for rank, result in enumerate(results, 1):
        if result["term"] in relevantTerms:
            return 1.0 / rank
    return 0.0


def calculateAP(results: list[dict], relevantTerms: list[str]) -> float:
    """
    计算 AP (Average Precision)

    Args:
        results: 检索结果列表
        relevantTerms: 相关术语列表

    Returns:
        AP 值（0-1）
    """
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


def calculateDCG(results: list[dict], relevantTerms: list[str], k: int) -> float:
    """
    计算 DCG@K (Discounted Cumulative Gain)

    Args:
        results: 检索结果列表
        relevantTerms: 相关术语列表
        k: TopK 阈值

    Returns:
        DCG@K 值
    """
    dcg = 0.0
    for i, result in enumerate(results[:k], 1):
        term = result["term"]
        # 相关性评分：在 relevant_terms 中的位置越靠前，相关性越高
        if term in relevantTerms:
            # 第一个相关术语得分最高，后续递减
            relevance = len(relevantTerms) - relevantTerms.index(term)
        else:
            relevance = 0

        # DCG 公式：rel / log2(i+1)
        dcg += relevance / math.log2(i + 1)

    return dcg


def calculateIDCG(relevantTerms: list[str], k: int) -> float:
    """
    计算 IDCG@K (Ideal DCG)

    Args:
        relevantTerms: 相关术语列表（已按相关性排序）
        k: TopK 阈值

    Returns:
        IDCG@K 值
    """
    idcg = 0.0
    for i in range(min(k, len(relevantTerms))):
        relevance = len(relevantTerms) - i
        idcg += relevance / math.log2(i + 2)  # i+2 因为从 i=0 开始

    return idcg


def calculateNDCG(results: list[dict], relevantTerms: list[str], k: int) -> float:
    """
    计算 nDCG@K (Normalized DCG)

    Args:
        results: 检索结果列表
        relevantTerms: 相关术语列表
        k: TopK 阈值

    Returns:
        nDCG@K 值（0-1）
    """
    dcg = calculateDCG(results, relevantTerms, k)
    idcg = calculateIDCG(relevantTerms, k)

    return dcg / idcg if idcg > 0 else 0.0


def evaluateMethod(
    method: str,
    retriever: Any,
    queries: list[dict],
    topK: int = 10,
) -> dict[str, Any]:
    """
    评测单个检索方法

    Args:
        method: 方法名称（BM25/Vector/Hybrid-Weighted/Hybrid-RRF）
        retriever: 检索器实例
        queries: 查询列表
        topK: TopK 阈值

    Returns:
        评测结果字典
    """
    print(f"\n{'=' * 60}")
    print(f"📊 评测方法: {method}")
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

        print(
            f"  查询 {i}/{len(queries)}: {queryText} (相关术语: {len(relevantTerms)})"
        )

        # 执行检索
        startTime = time.time()
        try:
            if method.startswith("Hybrid-"):
                strategy = "weighted" if method == "Hybrid-Weighted" else "rrf"
                results = retriever.search(
                    queryText,
                    topK=topK,
                    strategy=strategy,
                    alpha=0.5,
                    beta=0.5,
                )
            else:
                results = retriever.search(queryText, topK=topK)
        except Exception as e:
            print(f"    ❌ 检索失败: {e}")
            continue

        queryTime = time.time() - startTime
        queryTimes.append(queryTime)

        # 计算指标
        metrics["recall@1"].append(calculateRecallAtK(results, relevantTerms, 1))
        metrics["recall@3"].append(calculateRecallAtK(results, relevantTerms, 3))
        metrics["recall@5"].append(calculateRecallAtK(results, relevantTerms, 5))
        metrics["recall@10"].append(calculateRecallAtK(results, relevantTerms, 10))
        metrics["mrr"].append(calculateMRR(results, relevantTerms))
        metrics["map"].append(calculateAP(results, relevantTerms))
        metrics["ndcg@3"].append(calculateNDCG(results, relevantTerms, 3))
        metrics["ndcg@5"].append(calculateNDCG(results, relevantTerms, 5))
        metrics["ndcg@10"].append(calculateNDCG(results, relevantTerms, 10))

        print(f"    ⏱️  查询时间: {queryTime * 1000:.2f}ms")

    # 计算平均值
    if queryTimes:
        metrics["avg_query_time"] = sum(queryTimes) / len(queryTimes)

    avgMetrics = {
        "recall@1": sum(metrics["recall@1"]) / len(metrics["recall@1"])
        if metrics["recall@1"]
        else 0.0,
        "recall@3": sum(metrics["recall@3"]) / len(metrics["recall@3"])
        if metrics["recall@3"]
        else 0.0,
        "recall@5": sum(metrics["recall@5"]) / len(metrics["recall@5"])
        if metrics["recall@5"]
        else 0.0,
        "recall@10": sum(metrics["recall@10"]) / len(metrics["recall@10"])
        if metrics["recall@10"]
        else 0.0,
        "mrr": sum(metrics["mrr"]) / len(metrics["mrr"]) if metrics["mrr"] else 0.0,
        "map": sum(metrics["map"]) / len(metrics["map"]) if metrics["map"] else 0.0,
        "ndcg@3": sum(metrics["ndcg@3"]) / len(metrics["ndcg@3"])
        if metrics["ndcg@3"]
        else 0.0,
        "ndcg@5": sum(metrics["ndcg@5"]) / len(metrics["ndcg@5"])
        if metrics["ndcg@5"]
        else 0.0,
        "ndcg@10": sum(metrics["ndcg@10"]) / len(metrics["ndcg@10"])
        if metrics["ndcg@10"]
        else 0.0,
    }

    print("\n📈 平均指标:")
    print(f"  Recall@1:  {avgMetrics['recall@1']:.4f}")
    print(f"  Recall@3:  {avgMetrics['recall@3']:.4f}")
    print(f"  Recall@5:  {avgMetrics['recall@5']:.4f}")
    print(f"  Recall@10: {avgMetrics['recall@10']:.4f}")
    print(f"  MRR:       {avgMetrics['mrr']:.4f}")
    print(f"  MAP:       {avgMetrics['map']:.4f}")
    print(f"  nDCG@3:    {avgMetrics['ndcg@3']:.4f}")
    print(f"  nDCG@5:    {avgMetrics['ndcg@5']:.4f}")
    print(f"  nDCG@10:   {avgMetrics['ndcg@10']:.4f}")
    print(f"  平均查询时间: {metrics['avg_query_time'] * 1000:.2f}ms")

    metrics["avg_metrics"] = avgMetrics
    return metrics


def generateComparisonChart(allMetrics: list[dict], outputDir: str) -> None:
    """
    生成对比图表

    Args:
        allMetrics: 所有方法的评测结果
        outputDir: 输出目录
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 中文字体
        plt.rcParams["axes.unicode_minus"] = False  # 负号显示

        # 提取数据
        methods = [m["method"] for m in allMetrics]
        recallK = [1, 3, 5, 10]
        ndcgK = [3, 5, 10]

        # 图1: Recall@K 对比
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("检索评测指标对比", fontsize=16, fontweight="bold")

        # Recall@K
        ax = axes[0, 0]
        x = np.arange(len(methods))
        width = 0.2
        for i, k in enumerate(recallK):
            values = [m["avg_metrics"][f"recall@{k}"] for m in allMetrics]
            ax.bar(x + i * width, values, width, label=f"Recall@{k}")
        ax.set_ylabel("Recall")
        ax.set_title("Recall@K")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # nDCG@K
        ax = axes[0, 1]
        x = np.arange(len(methods))
        width = 0.25
        for i, k in enumerate(ndcgK):
            values = [m["avg_metrics"][f"ndcg@{k}"] for m in allMetrics]
            ax.bar(x + i * width, values, width, label=f"nDCG@{k}")
        ax.set_ylabel("nDCG")
        ax.set_title("nDCG@K")
        ax.set_xticks(x + width)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # MRR 和 MAP
        ax = axes[1, 0]
        x = np.arange(len(methods))
        width = 0.35
        mrr = [m["avg_metrics"]["mrr"] for m in allMetrics]
        map_scores = [m["avg_metrics"]["map"] for m in allMetrics]
        ax.bar(x - width / 2, mrr, width, label="MRR")
        ax.bar(x + width / 2, map_scores, width, label="MAP")
        ax.set_ylabel("Score")
        ax.set_title("MRR 和 MAP 对比")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # 查询时间
        ax = axes[1, 1]
        x = np.arange(len(methods))
        times = [m["avg_query_time"] * 1000 for m in allMetrics]  # 转换为毫秒
        bars = ax.bar(x, times, color="skyblue")
        ax.set_ylabel("时间 (ms)")
        ax.set_title("平均查询时间")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.3)

        # 在柱状图上显示数值
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{time_val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()

        # 保存图表
        os.makedirs(outputDir, exist_ok=True)
        chartPath = os.path.join(outputDir, "retrieval_comparison.png")
        plt.savefig(chartPath, dpi=300, bbox_inches="tight")
        print(f"✅ 对比图表已保存: {chartPath}")

    except ImportError:
        print("⚠️  跳过图表生成：matplotlib 未安装")
    except Exception as e:
        print(f"⚠️  图表生成失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="检索评测脚本")
    parser.add_argument(
        "--queries",
        type=str,
        default=os.path.join(
            config.PROJECT_ROOT, "data", "evaluation", "queries.jsonl"
        ),
        help="查询集文件路径",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bm25", "vector", "hybrid-weighted", "hybrid-rrf"],
        choices=["bm25", "vector", "hybrid-weighted", "hybrid-rrf"],
        help="要评测的检索方法",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="TopK 阈值",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="生成对比图表",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            config.PROJECT_ROOT, "outputs", "reports", "retrieval_metrics.json"
        ),
        help="输出报告路径",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("📊 Math-RAG 检索评测")
    print("=" * 60)
    print(f"查询集: {args.queries}")
    print(f"评测方法: {', '.join(args.methods)}")
    print(f"TopK: {args.topk}")
    print("=" * 60)

    # 加载查询集
    queries = loadQueries(args.queries)
    if not queries:
        print("❌ 无有效查询，退出")
        return

    # 按学科统计
    subjectCount = defaultdict(int)
    for q in queries:
        subjectCount[q["subject"]] += 1
    print("\n📚 学科分布:")
    for subject, count in sorted(subjectCount.items()):
        print(f"  {subject}: {count} 条")

    # 初始化检索器
    retrievers = {}
    corpusPath = os.path.join(config.PROCESSED_DIR, "retrieval", "corpus.jsonl")

    for method in args.methods:
        print(f"\n🔄 初始化检索器: {method.upper()}")
        try:
            if method == "bm25":
                from retrieval.retrievalBM25 import BM25Retriever

                retriever = BM25Retriever(corpusPath)
                # 尝试加载索引，如果不存在则构建
                if not retriever.loadIndex():
                    print("  索引不存在，开始构建...")
                    retriever.buildIndex()
                    retriever.saveIndex()
                retrievers["BM25"] = retriever
            elif method == "vector":
                from retrieval.retrievalVector import VectorRetriever

                retriever = VectorRetriever(corpusPath)
                # 尝试加载索引，如果不存在则构建
                if not retriever.loadIndex():
                    print("  索引不存在，开始构建...")
                    retriever.buildIndex()
                    retriever.saveIndex()
                retrievers["Vector"] = retriever
            elif method == "hybrid-weighted":
                from retrieval.retrievalHybrid import HybridRetriever

                retriever = HybridRetriever(corpusPath)
                # Hybrid 会自动初始化子检索器
                retrievers["Hybrid-Weighted"] = retriever
            elif method == "hybrid-rrf":
                from retrieval.retrievalHybrid import HybridRetriever

                retriever = HybridRetriever(corpusPath)
                # Hybrid 会自动初始化子检索器
                retrievers["Hybrid-RRF"] = retriever
        except ImportError as e:
            print(f"❌ 初始化失败（缺少依赖）: {e}")
            print(f"💡 提示: 请检查 {method.upper()} 所需的依赖库是否已安装")
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            import traceback

            traceback.print_exc()

    if not retrievers:
        print("❌ 没有可用的检索器，退出")
        return

    # 评测各方法
    allMetrics = []
    for methodName, retriever in retrievers.items():
        try:
            metrics = evaluateMethod(methodName, retriever, queries, args.topk)
            allMetrics.append(metrics)
        except Exception as e:
            print(f"❌ 评测 {methodName} 失败: {e}")

    if not allMetrics:
        print("❌ 评测失败，无结果")
        return

    # 保存报告
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "queries_file": args.queries,
        "total_queries": len(queries),
        "subject_distribution": dict(subjectCount),
        "topk": args.topk,
        "results": allMetrics,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 评测报告已保存: {args.output}")

    # 生成对比表格
    print(f"\n{'=' * 60}")
    print("📊 评测结果汇总")
    print(f"{'=' * 60}")

    # 表头
    print(
        f"{'方法':<20} {'Recall@1':<10} {'Recall@10':<10} {'MRR':<10} {'MAP':<10} {'nDCG@10':<10} {'查询时间':<10}"
    )
    print("-" * 90)

    # 数据行
    for m in allMetrics:
        avg = m["avg_metrics"]
        print(
            f"{m['method']:<20} "
            f"{avg['recall@1']:<10.4f} "
            f"{avg['recall@10']:<10.4f} "
            f"{avg['mrr']:<10.4f} "
            f"{avg['map']:<10.4f} "
            f"{avg['ndcg@10']:<10.4f} "
            f"{m['avg_query_time'] * 1000:<10.2f}ms"
        )

    # 生成图表
    if args.visualize and len(allMetrics) > 1:
        outputDir = os.path.dirname(args.output)
        generateComparisonChart(allMetrics, outputDir)

    print("\n✅ 评测完成！")


if __name__ == "__main__":
    main()
