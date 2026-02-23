"""
混合检索 + 改进版

功能：
1. 改进的 RRF 融合策略
2. 自适应权重调整
3. 支持更多召回结果进行融合
4. 改进的归一化方法

使用方法：
    # 默认加权融合
    python retrieval/retrievalHybridPlus.py --query "泰勒展开" --topk 10

    # 使用 RRF 融合
    python retrieval/retrievalHybridPlus.py --query "泰勒展开" --topk 10 --strategy rrf

    # 自适应权重
    python retrieval/retrievalHybridPlus.py --query "泰勒展开" --topk 10 --auto-weight
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
from retrieval.retrievalBM25Plus import BM25PlusRetriever
from retrieval.retrievalVector import VectorRetriever


class HybridPlusRetriever:
    """改进的混合检索器"""

    def __init__(
        self,
        corpusFile: str,
        bm25IndexFile: str,
        vectorIndexFile: str,
        vectorEmbeddingFile: str,
        modelName: str = "paraphrase-multilingual-MiniLM-L12-v2",
        termsFile: str | None = None,
    ):
        """
        初始化改进的混合检索器

        Args:
            corpusFile: 语料文件路径
            bm25IndexFile: BM25 索引文件路径
            vectorIndexFile: 向量索引文件路径
            vectorEmbeddingFile: 向量嵌入文件路径
            modelName: Sentence Transformer 模型名称
            termsFile: 术语文件路径（用于 BM25+ 查询扩展）
        """
        self.corpusFile = corpusFile

        # 初始化 BM25+ 检索器（支持查询扩展）
        print("🔧 初始化 BM25+ 检索器...")
        self.bm25Retriever = BM25PlusRetriever(corpusFile, bm25IndexFile, termsFile)
        if not self.bm25Retriever.loadIndex():
            print("⚠️  BM25+ 索引不存在，正在构建...")
            self.bm25Retriever.loadTermsMap()
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
        """Min-Max 归一化"""
        if not scores:
            return []

        minScore = min(scores)
        maxScore = max(scores)

        if maxScore == minScore:
            return [1.0] * len(scores)

        return [(s - minScore) / (maxScore - minScore) for s in scores]

    def normalizeZScore(self, scores: list[float]) -> list[float]:
        """Z-Score 归一化"""
        if not scores:
            return []

        mean = np.mean(scores)
        std = np.std(scores)

        if std == 0:
            return [0.0] * len(scores)

        return [(s - mean) / std for s in scores]

    def normalizePercentile(self, scores: list[float]) -> list[float]:
        """百分位数归一化（更鲁棒）"""
        if not scores:
            return []

        sortedScores = sorted(scores)
        n = len(sortedScores)

        result = []
        for s in scores:
            # 计算小于等于当前分数的数量
            rank = sum(1 for x in sortedScores if x <= s)
            result.append(rank / n)

        return result

    def fuseRRFImproved(
        self,
        bm25Results: list[dict[str, Any]],
        vectorResults: list[dict[str, Any]],
        topK: int = 10,
        rrfK: int = 60,
    ) -> list[dict[str, Any]]:
        """
        改进的 RRF 融合策略

        改进点：
        1. 使用更多候选结果进行融合
        2. 根据查询难度动态调整 k 值
        3. 添加分数加权
        """
        # 计算查询难度（基于 BM25 分数分布）
        if bm25Results:
            bm25Scores = [r["score"] for r in bm25Results]
            avgScore = np.mean(bm25Scores)
            # 查询难度高时使用更小的 k 值
            if avgScore < 0.5:
                rrfK = max(30, rrfK // 2)
            elif avgScore > 2.0:
                rrfK = min(100, rrfK * 2)

        # 构建 doc_id 到排名的映射
        bm25RankMap = {r["doc_id"]: r["rank"] for r in bm25Results}
        vectorRankMap = {r["doc_id"]: r["rank"] for r in vectorResults}

        # 收集所有唯一的 doc_id
        allDocIds = set(bm25RankMap.keys()) | set(vectorRankMap.keys())

        # 计算 RRF 分数
        rrfScores = {}
        for docId in allDocIds:
            rrfScore = 0.0
            if docId in bm25RankMap:
                rrfScore += 1.0 / (rrfK + bm25RankMap[docId])
            if docId in vectorRankMap:
                rrfScore += 1.0 / (rrfK + vectorRankMap[docId])
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
            if rank > topK:
                break
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

    def fuseWeightedImproved(
        self,
        bm25Results: list[dict[str, Any]],
        vectorResults: list[dict[str, Any]],
        topK: int = 10,
        alpha: float | None = None,
        beta: float | None = None,
        normalization: str = "percentile",
    ) -> list[dict[str, Any]]:
        """
        改进的加权融合策略

        改进点：
        1. 使用百分位数归一化（更鲁棒）
        2. 自适应权重调整
        3. 考虑结果重叠度
        """
        # 提取分数
        bm25Scores = [r["score"] for r in bm25Results]
        vectorScores = [r["score"] for r in vectorResults]

        if not bm25Scores or not vectorScores:
            # 如果一方无结果，使用另一方
            if bm25Scores:
                return bm25Results[:topK]
            return vectorResults[:topK]

        # 归一化
        if normalization == "minmax":
            bm25NormScores = self.normalizeMinMax(bm25Scores)
            vectorNormScores = self.normalizeMinMax(vectorScores)
        elif normalization == "zscore":
            bm25NormScores = self.normalizeZScore(bm25Scores)
            vectorNormScores = self.normalizeZScore(vectorScores)
        else:  # percentile
            bm25NormScores = self.normalizePercentile(bm25Scores)
            vectorNormScores = self.normalizePercentile(vectorScores)

        # 计算结果重叠度
        bm25DocIds = set(r["doc_id"] for r in bm25Results)
        vectorDocIds = set(r["doc_id"] for r in vectorResults)
        overlap = len(bm25DocIds & vectorDocIds)
        overlapRatio = overlap / min(len(bm25DocIds), len(vectorDocIds))

        # 自适应权重调整
        if alpha is None or beta is None:
            # 如果重叠度高，说明两种方法一致，可以平均权重
            if overlapRatio > 0.5:
                alpha = 0.5
                beta = 0.5
            else:
                # 重叠度低时，根据平均分数动态调整权重
                avgBm25 = np.mean(bm25NormScores)
                avgVector = np.mean(vectorNormScores)
                total = avgBm25 + avgVector
                if total > 0:
                    alpha = avgBm25 / total
                    beta = avgVector / total
                else:
                    alpha = beta = 0.5

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

        # 获取文档详细信息
        docInfoMap = {r["doc_id"]: r for r in bm25Results}
        docInfoMap.update({r["doc_id"]: r for r in vectorResults})

        results = []
        for rank, docId in enumerate(sortedDocIds, 1):
            if rank > topK:
                break
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

    def search(
        self,
        query: str,
        topK: int = 10,
        strategy: str = "weighted",
        alpha: float | None = None,
        beta: float | None = None,
        normalization: str = "percentile",
        rrfK: int = 60,
        expandQuery: bool = True,
        recallFactor: int = 5,
    ) -> list[dict[str, Any]]:
        """
        改进的混合检索

        Args:
            query: 查询字符串
            topK: 返回的结果数量
            strategy: 融合策略（weighted 或 rrf）
            alpha: BM25 权重
            beta: 向量检索权重
            normalization: 归一化方法
            rrfK: RRF 参数
            expandQuery: 是否进行查询扩展
            recallFactor: 召回因子（检索 topK * recallFactor 用于融合）

        Returns:
            融合后的结果列表
        """
        # 执行两种检索（获取更多结果用于融合）
        recallTopK = topK * recallFactor

        print("🔍 执行 BM25+ 检索...")
        bm25Results = self.bm25Retriever.search(
            query, recallTopK, expandQuery=expandQuery, returnAll=False
        )

        print("🔍 执行向量检索...")
        vectorResults = self.vectorRetriever.search(query, recallTopK)

        # 融合结果
        print(f"🔀 融合结果（策略：{strategy}）...")
        if strategy == "weighted":
            fusedResults = self.fuseWeightedImproved(
                bm25Results, vectorResults, topK, alpha, beta, normalization
            )
        elif strategy == "rrf":
            fusedResults = self.fuseRRFImproved(bm25Results, vectorResults, topK, rrfK)
        else:
            raise ValueError(f"不支持的融合策略：{strategy}")

        return fusedResults

    def batchSearch(
        self,
        queries: list[str],
        topK: int = 10,
        strategy: str = "weighted",
        **kwargs,
    ) -> dict[str, list[dict[str, Any]]]:
        """批量混合检索"""
        results = {}
        for query in queries:
            results[query] = self.search(query, topK, strategy, **kwargs)
        return results


def loadQueriesFromFile(filepath: str) -> list[str]:
    """从文件加载查询"""
    queries = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(line)
    return queries


def saveResults(results: dict[str, list[dict[str, Any]]], outputFile: str) -> None:
    """保存查询结果到文件"""
    dirname = os.path.dirname(outputFile)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(outputFile, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 结果已保存：{outputFile}")


def printResults(query: str, results: list[dict[str, Any]], strategy: str) -> None:
    """打印查询结果"""
    print("\n" + "=" * 80)
    print(f"🔍 查询：{query}")
    print(f"🔀 融合策略：{strategy}")
    print("=" * 80)

    if not results:
        print("❌ 未找到相关结果")
        return

    for result in results:
        print(f"\n🏆 Rank {result['rank']}")
        print(f"  📄 Doc ID: {result['doc_id']}")
        print(f"  📚 术语：{result['term']}")
        print(f"  📖 学科：{result['subject']}")
        print(f"  📊 融合分数：{result['score']:.4f}")

        if strategy == "weighted":
            print(f"     ├─ BM25: {result.get('bm25_score', 0):.4f}")
            print(f"     └─ 向量：{result.get('vector_score', 0):.4f}")
        elif strategy == "rrf":
            print(f"     ├─ BM25 Rank: {result.get('bm25_rank', 'N/A')}")
            print(f"     └─ 向量 Rank: {result.get('vector_rank', 'N/A')}")

        print(f"  📗 来源：{result['source']}")
        if result.get("page"):
            print(f"  📄 页码：{result['page']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="混合检索 + 改进版")
    parser.add_argument("--query", type=str, help="单次查询字符串")
    parser.add_argument("--query-file", type=str, help="批量查询文件路径")
    parser.add_argument(
        "--topk", type=int, default=10, help="返回的结果数量（默认 10）"
    )
    parser.add_argument("--output", type=str, help="输出结果文件路径")
    parser.add_argument(
        "--strategy",
        type=str,
        default="weighted",
        choices=["weighted", "rrf"],
        help="融合策略",
    )
    parser.add_argument("--alpha", type=float, help="BM25 权重")
    parser.add_argument("--beta", type=float, help="向量检索权重")
    parser.add_argument(
        "--normalization",
        type=str,
        default="percentile",
        choices=["minmax", "zscore", "percentile"],
        help="归一化方法",
    )
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF 参数 k")
    parser.add_argument("--corpus", type=str, help="语料文件路径")
    parser.add_argument("--bm25-index", type=str, help="BM25 索引文件路径")
    parser.add_argument("--vector-index", type=str, help="向量索引文件路径")
    parser.add_argument("--vector-embedding", type=str, help="向量嵌入文件路径")
    parser.add_argument("--terms", type=str, help="术语文件路径")
    parser.add_argument(
        "--model",
        type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence Transformer 模型名称",
    )
    parser.add_argument("--no-expand", action="store_true", help="禁用查询扩展")
    parser.add_argument(
        "--recall-factor",
        type=int,
        default=5,
        help="召回因子（检索 topK * factor 用于融合）",
    )
    parser.add_argument("--alpha", type=float, help="BM25 权重（默认 0.7）")
    parser.add_argument("--beta", type=float, help="向量检索权重（默认 0.3）")

    args = parser.parse_args()

    # 默认路径
    corpusFile = args.corpus or os.path.join(
        config.PROCESSED_DIR, "retrieval", "corpus.jsonl"
    )
    bm25IndexFile = args.bm25_index or os.path.join(
        config.PROCESSED_DIR, "retrieval", "bm25plus_index.pkl"
    )
    vectorIndexFile = args.vector_index or os.path.join(
        config.PROCESSED_DIR, "retrieval", "vector_index.faiss"
    )
    vectorEmbeddingFile = args.vector_embedding or os.path.join(
        config.PROCESSED_DIR, "retrieval", "vector_embeddings.npz"
    )
    termsFile = args.terms or os.path.join(
        config.PROCESSED_DIR, "terms", "all_terms.json"
    )

    print("=" * 80)
    print("🔍 混合检索 + 改进版")
    print("=" * 80)
    print(f"📂 语料文件：{corpusFile}")
    print(f"📂 BM25+ 索引：{bm25IndexFile}")
    print(f"📂 向量索引：{vectorIndexFile}")
    print(f"🔀 融合策略：{args.strategy}")
    if args.strategy == "weighted":
        print(f"⚖️  权重：BM25={args.alpha or 0.7}, 向量={args.beta or 0.3}")
    print(f"🔍 查询扩展：{'禁用' if args.no_expand else '启用'}")
    print(f"📈 召回因子：{args.recall_factor}")
    print()

    # 初始化混合检索器
    retriever = HybridPlusRetriever(
        corpusFile,
        bm25IndexFile,
        vectorIndexFile,
        vectorEmbeddingFile,
        args.model,
        termsFile,
    )

    # 执行查询
    if args.query:
        results = retriever.search(
            args.query,
            args.topk,
            args.strategy,
            args.alpha or (0.7 if args.strategy == "weighted" else None),
            args.beta or (0.3 if args.strategy == "weighted" else None),
            args.normalization,
            args.rrf_k,
            not args.no_expand,
            args.recall_factor,
        )
        printResults(args.query, results, args.strategy)

        if args.output:
            saveResults({args.query: results}, args.output)

    elif args.query_file:
        print(f"📂 加载查询：{args.query_file}")
        queries = loadQueriesFromFile(args.query_file)
        print(f"✅ 已加载 {len(queries)} 个查询\n")

        results = retriever.batchSearch(
            queries,
            args.topk,
            args.strategy,
            alpha=args.alpha or (0.7 if args.strategy == "weighted" else None),
            beta=args.beta or (0.3 if args.strategy == "weighted" else None),
            normalization=args.normalization,
            rrfK=args.rrf_k,
            expandQuery=not args.no_expand,
            recallFactor=args.recall_factor,
        )

        for query, queryResults in results.items():
            printResults(query, queryResults, args.strategy)

        if args.output:
            saveResults(results, args.output)
        else:
            defaultOutput = os.path.join(
                config.PROJECT_ROOT, "outputs", "hybrid_plus_results.json"
            )
            saveResults(results, defaultOutput)

    else:
        print("⚠️  请提供查询参数")
        parser.print_help()


if __name__ == "__main__":
    main()
