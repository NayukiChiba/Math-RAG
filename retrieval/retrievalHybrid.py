"""
混合检索

功能：
1. 结合 BM25 和向量检索的优势
2. 支持多种归一化策略（min-max、z-score）
3. 支持多种融合策略（加权融合、RRF）
4. 可配置权重参数

使用方法：
    # 加权融合（默认）
    python retrieval/retrievalHybrid.py --query "泰勒展开" --topk 10

    # 指定权重
    python retrieval/retrievalHybrid.py --query "泰勒展开" --alpha 0.7 --beta 0.3

    # 使用 RRF 融合
    python retrieval/retrievalHybrid.py --query "泰勒展开" --strategy rrf

    # 批量查询
    python retrieval/retrievalHybrid.py --query-file queries.txt --output results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# 路径调整
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from retrieval.retrievalBM25 import BM25Retriever
from retrieval.retrievalVector import VectorRetriever


class HybridRetriever:
    """混合检索器"""

    def __init__(
        self,
        corpusFile: str,
        bm25IndexFile: str,
        vectorIndexFile: str,
        vectorEmbeddingFile: str,
        modelName: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        初始化混合检索器

        Args:
            corpusFile: 语料文件路径
            bm25IndexFile: BM25 索引文件路径
            vectorIndexFile: 向量索引文件路径
            vectorEmbeddingFile: 向量嵌入文件路径
            modelName: Sentence Transformer 模型名称
        """
        self.corpusFile = corpusFile

        # 初始化 BM25 检索器
        print("🔧 初始化 BM25 检索器...")
        self.bm25Retriever = BM25Retriever(corpusFile, bm25IndexFile)
        if not self.bm25Retriever.loadIndex():
            print("⚠️  BM25 索引不存在，正在构建...")
            self.bm25Retriever.buildIndex()
            self.bm25Retriever.saveIndex()

        # 初始化向量检索器
        print("🔧 初始化向量检索器...")
        self.vectorRetriever = VectorRetriever(
            corpusFile, modelName, vectorIndexFile, vectorEmbeddingFile
        )
        if not self.vectorRetriever.loadIndex():
            print("⚠️  向量索引不存在，正在构建...")
            self.vectorRetriever.buildIndex()
            self.vectorRetriever.saveIndex()

        print("✅ 混合检索器初始化完成\n")

    def normalizeMinMax(self, scores: list[float]) -> list[float]:
        """
        Min-Max 归一化

        Args:
            scores: 原始分数列表

        Returns:
            归一化后的分数列表
        """
        if not scores:
            return []

        minScore = min(scores)
        maxScore = max(scores)

        if maxScore == minScore:
            return [1.0] * len(scores)

        return [(s - minScore) / (maxScore - minScore) for s in scores]

    def normalizeZScore(self, scores: list[float]) -> list[float]:
        """
        Z-Score 归一化

        Args:
            scores: 原始分数列表

        Returns:
            归一化后的分数列表
        """
        if not scores:
            return []

        mean = np.mean(scores)
        std = np.std(scores)

        if std == 0:
            return [0.0] * len(scores)

        return [(s - mean) / std for s in scores]

    def fuseWeighted(
        self,
        bm25Results: list[dict[str, Any]],
        vectorResults: list[dict[str, Any]],
        alpha: float = 0.5,
        beta: float = 0.5,
        normalization: str = "minmax",
    ) -> list[dict[str, Any]]:
        """
        加权融合策略

        Args:
            bm25Results: BM25 检索结果
            vectorResults: 向量检索结果
            alpha: BM25 权重
            beta: 向量检索权重
            normalization: 归一化方法（minmax 或 zscore）

        Returns:
            融合后的结果列表
        """
        # 提取分数
        bm25Scores = [r["score"] for r in bm25Results]
        vectorScores = [r["score"] for r in vectorResults]

        # 归一化
        if normalization == "minmax":
            bm25NormScores = self.normalizeMinMax(bm25Scores)
            vectorNormScores = self.normalizeMinMax(vectorScores)
        elif normalization == "zscore":
            bm25NormScores = self.normalizeZScore(bm25Scores)
            vectorNormScores = self.normalizeZScore(vectorScores)
        else:
            raise ValueError(f"不支持的归一化方法: {normalization}")

        # 构建 doc_id 到归一化分数的映射
        bm25ScoreMap = {
            r["doc_id"]: bm25NormScores[i] for i, r in enumerate(bm25Results)
        }
        vectorScoreMap = {
            r["doc_id"]: vectorNormScores[i] for i, r in enumerate(vectorResults)
        }

        # 收集所有唯一的 doc_id
        allDocIds = set(bm25ScoreMap.keys()) | set(vectorScoreMap.keys())

        # 计算融合分数
        fusedScores = {}
        for docId in allDocIds:
            bm25Score = bm25ScoreMap.get(docId, 0.0)
            vectorScore = vectorScoreMap.get(docId, 0.0)
            fusedScores[docId] = alpha * bm25Score + beta * vectorScore

        # 排序并构建结果
        sortedDocIds = sorted(
            fusedScores.keys(), key=lambda x: fusedScores[x], reverse=True
        )

        # 获取文档详细信息（优先从 BM25 结果中获取，因为包含更多字段）
        docInfoMap = {r["doc_id"]: r for r in bm25Results}
        docInfoMap.update({r["doc_id"]: r for r in vectorResults})

        results = []
        for rank, docId in enumerate(sortedDocIds, 1):
            docInfo = docInfoMap[docId]
            results.append(
                {
                    "rank": rank,
                    "doc_id": docId,
                    "term": docInfo["term"],
                    "subject": docInfo.get("subject", ""),
                    "score": fusedScores[docId],
                    "bm25_score": bm25ScoreMap.get(docId, 0.0),
                    "vector_score": vectorScoreMap.get(docId, 0.0),
                    "source": docInfo.get("source", ""),
                    "page": docInfo.get("page", None),
                }
            )

        return results

    def fuseRRF(
        self,
        bm25Results: list[dict[str, Any]],
        vectorResults: list[dict[str, Any]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) 融合策略

        Args:
            bm25Results: BM25 检索结果
            vectorResults: 向量检索结果
            k: RRF 参数（默认 60）

        Returns:
            融合后的结果列表
        """
        # 构建 doc_id 到排名的映射
        bm25RankMap = {r["doc_id"]: r["rank"] for r in bm25Results}
        vectorRankMap = {r["doc_id"]: r["rank"] for r in vectorResults}

        # 收集所有唯一的 doc_id
        allDocIds = set(bm25RankMap.keys()) | set(vectorRankMap.keys())

        # 计算 RRF 分数（标准 RRF：仅对有排名的结果求和，未命中时贡献为 0）
        rrfScores = {}
        for docId in allDocIds:
            rrfScore = 0.0
            # BM25 贡献（如果存在）
            if docId in bm25RankMap:
                rrfScore += 1.0 / (k + bm25RankMap[docId])
            # 向量检索贡献（如果存在）
            if docId in vectorRankMap:
                rrfScore += 1.0 / (k + vectorRankMap[docId])
            rrfScores[docId] = rrfScore

        # 排序并构建结果
        sortedDocIds = sorted(
            rrfScores.keys(), key=lambda x: rrfScores[x], reverse=True
        )

        # 获取文档详细信息
        docInfoMap = {r["doc_id"]: r for r in bm25Results}
        docInfoMap.update({r["doc_id"]: r for r in vectorResults})

        results = []
        for rank, docId in enumerate(sortedDocIds, 1):
            docInfo = docInfoMap[docId]
            results.append(
                {
                    "rank": rank,
                    "doc_id": docId,
                    "term": docInfo["term"],
                    "subject": docInfo.get("subject", ""),
                    "score": rrfScores[docId],
                    "bm25_rank": bm25RankMap.get(docId, None),
                    "vector_rank": vectorRankMap.get(docId, None),
                    "source": docInfo.get("source", ""),
                    "page": docInfo.get("page", None),
                }
            )

        return results

    def search(
        self,
        query: str,
        topK: int = 10,
        strategy: str = "weighted",
        alpha: float = 0.5,
        beta: float = 0.5,
        normalization: str = "minmax",
        rrfK: int = 60,
    ) -> list[dict[str, Any]]:
        """
        混合检索

        Args:
            query: 查询字符串
            topK: 返回的结果数量
            strategy: 融合策略（weighted 或 rrf）
            alpha: BM25 权重（仅 weighted 策略）
            beta: 向量检索权重（仅 weighted 策略）
            normalization: 归一化方法（minmax 或 zscore，仅 weighted 策略）
            rrfK: RRF 参数（仅 rrf 策略）

        Returns:
            融合后的结果列表
        """
        # 执行两种检索
        print("🔍 执行 BM25 检索...")
        bm25Results = self.bm25Retriever.search(query, topK * 2)  # 获取更多结果用于融合

        print("🔍 执行向量检索...")
        vectorResults = self.vectorRetriever.search(query, topK * 2)

        # 融合结果
        print(f"🔀 融合结果（策略: {strategy}）...")
        if strategy == "weighted":
            fusedResults = self.fuseWeighted(
                bm25Results, vectorResults, alpha, beta, normalization
            )
        elif strategy == "rrf":
            fusedResults = self.fuseRRF(bm25Results, vectorResults, rrfK)
        else:
            raise ValueError(f"不支持的融合策略: {strategy}")

        # 返回 TopK
        return fusedResults[:topK]

    def batchSearch(
        self,
        queries: list[str],
        topK: int = 10,
        strategy: str = "weighted",
        alpha: float = 0.5,
        beta: float = 0.5,
        normalization: str = "minmax",
        rrfK: int = 60,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        批量混合检索

        Args:
            queries: 查询字符串列表
            topK: 每个查询返回的结果数量
            strategy: 融合策略
            alpha: BM25 权重
            beta: 向量检索权重
            normalization: 归一化方法
            rrfK: RRF 参数

        Returns:
            字典，键为查询字符串，值为结果列表
        """
        results = {}
        for query in queries:
            results[query] = self.search(
                query, topK, strategy, alpha, beta, normalization, rrfK
            )
        return results


def loadQueriesFromFile(filepath: str) -> list[str]:
    """
    从文件加载查询

    Args:
        filepath: 查询文件路径（每行一个查询）

    Returns:
        查询列表
    """
    queries = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(line)
    return queries


def saveResults(results: dict[str, list[dict[str, Any]]], outputFile: str) -> None:
    """
    保存查询结果到文件

    Args:
        results: 查询结果字典
        outputFile: 输出文件路径
    """
    # 确保输出目录存在
    dirname = os.path.dirname(outputFile)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(outputFile, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 结果已保存: {outputFile}")


def printResults(query: str, results: list[dict[str, Any]], strategy: str) -> None:
    """
    打印查询结果

    Args:
        query: 查询字符串
        results: 结果列表
        strategy: 融合策略
    """
    print("\n" + "=" * 80)
    print(f"🔍 查询: {query}")
    print(f"🔀 融合策略: {strategy}")
    print("=" * 80)

    if not results:
        print("❌ 未找到相关结果")
        return

    for result in results:
        print(f"\n🏆 Rank {result['rank']}")
        print(f"  📄 Doc ID: {result['doc_id']}")
        print(f"  📚 术语: {result['term']}")
        print(f"  📖 学科: {result['subject']}")
        print(f"  📊 融合分数: {result['score']:.4f}")

        if strategy == "weighted":
            print(f"     ├─ BM25: {result.get('bm25_score', 0):.4f}")
            print(f"     └─ 向量: {result.get('vector_score', 0):.4f}")
        elif strategy == "rrf":
            print(f"     ├─ BM25 Rank: {result.get('bm25_rank', 'N/A')}")
            print(f"     └─ 向量 Rank: {result.get('vector_rank', 'N/A')}")

        print(f"  📗 来源: {result['source']}")
        if result.get("page"):
            print(f"  📄 页码: {result['page']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="混合检索")
    parser.add_argument("--query", type=str, help="单次查询字符串")
    parser.add_argument("--query-file", type=str, help="批量查询文件路径")
    parser.add_argument(
        "--topk", type=int, default=10, help="返回的结果数量（默认 10）"
    )
    parser.add_argument("--output", type=str, help="输出结果文件路径（JSON 格式）")
    parser.add_argument(
        "--strategy",
        type=str,
        default="weighted",
        choices=["weighted", "rrf"],
        help="融合策略（weighted 或 rrf，默认 weighted）",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="BM25 权重（默认 0.5）"
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, help="向量检索权重（默认 0.5）"
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="minmax",
        choices=["minmax", "zscore"],
        help="归一化方法（minmax 或 zscore，默认 minmax）",
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF 参数 k（默认 60）")
    parser.add_argument("--corpus", type=str, help="语料文件路径")
    parser.add_argument("--bm25-index", type=str, help="BM25 索引文件路径")
    parser.add_argument("--vector-index", type=str, help="向量索引文件路径")
    parser.add_argument("--vector-embedding", type=str, help="向量嵌入文件路径")
    parser.add_argument(
        "--model",
        type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence Transformer 模型名称",
    )

    args = parser.parse_args()

    # 默认路径
    corpusFile = args.corpus or os.path.join(
        config.PROCESSED_DIR, "retrieval", "corpus.jsonl"
    )
    bm25IndexFile = args.bm25_index or os.path.join(
        config.PROCESSED_DIR, "retrieval", "bm25_index.pkl"
    )
    vectorIndexFile = args.vector_index or os.path.join(
        config.PROCESSED_DIR, "retrieval", "vector_index.faiss"
    )
    vectorEmbeddingFile = args.vector_embedding or os.path.join(
        config.PROCESSED_DIR, "retrieval", "vector_embeddings.npz"
    )

    print("=" * 80)
    print("🔍 混合检索")
    print("=" * 80)
    print(f"📂 语料文件: {corpusFile}")
    print(f"📂 BM25 索引: {bm25IndexFile}")
    print(f"📂 向量索引: {vectorIndexFile}")
    print(f"📂 向量嵌入: {vectorEmbeddingFile}")
    print(f"🤖 模型: {args.model}")
    print(f"🔀 融合策略: {args.strategy}")
    if args.strategy == "weighted":
        print(f"⚖️  权重: BM25={args.alpha}, 向量={args.beta}")
        print(f"📐 归一化: {args.normalization}")
    elif args.strategy == "rrf":
        print(f"🔢 RRF k: {args.rrf_k}")
    print()

    # 初始化混合检索器
    retriever = HybridRetriever(
        corpusFile,
        bm25IndexFile,
        vectorIndexFile,
        vectorEmbeddingFile,
        args.model,
    )

    # 执行查询
    if args.query:
        # 单次查询
        results = retriever.search(
            args.query,
            args.topk,
            args.strategy,
            args.alpha,
            args.beta,
            args.normalization,
            args.rrf_k,
        )
        printResults(args.query, results, args.strategy)

        if args.output:
            saveResults({args.query: results}, args.output)

    elif args.query_file:
        # 批量查询
        print(f"📂 加载查询: {args.query_file}")
        queries = loadQueriesFromFile(args.query_file)
        print(f"✅ 已加载 {len(queries)} 个查询\n")

        results = retriever.batchSearch(
            queries,
            args.topk,
            args.strategy,
            args.alpha,
            args.beta,
            args.normalization,
            args.rrf_k,
        )

        # 打印每个查询的结果
        for query, queryResults in results.items():
            printResults(query, queryResults, args.strategy)

        # 保存结果
        if args.output:
            saveResults(results, args.output)
        else:
            # 默认输出文件
            defaultOutput = os.path.join(
                config.PROJECT_ROOT, "outputs", "hybrid_results.json"
            )
            saveResults(results, defaultOutput)

    else:
        print("⚠️  请提供查询参数：")
        print("  --query 'your query'  # 单次查询")
        print("  --query-file queries.txt  # 批量查询")
        parser.print_help()


if __name__ == "__main__":
    main()
