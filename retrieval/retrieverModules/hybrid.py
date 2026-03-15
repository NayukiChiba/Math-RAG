from __future__ import annotations

from typing import Any

import numpy as np

from retrieval.retrieverModules.bm25 import BM25Retriever
from retrieval.retrieverModules.shared import _DEFAULT_VECTOR_MODEL
from retrieval.retrieverModules.vector import VectorRetriever


class HybridRetriever:
    """混合检索器"""

    def __init__(
        self,
        corpusFile: str,
        bm25IndexFile: str,
        vectorIndexFile: str,
        vectorEmbeddingFile: str,
        modelName: str = _DEFAULT_VECTOR_MODEL,
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

        # 计算 RRF 分数
        rrfScores = {}
        for docId in allDocIds:
            rrfScore = 0.0
            if docId in bm25RankMap:
                rrfScore += 1.0 / (k + bm25RankMap[docId])
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
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """
        混合检索

        Args:
            query: 查询字符串
            topK: 返回的结果数量
            strategy: 融合策略（weighted 或 rrf）
            alpha: BM25 权重
            beta: 向量检索权重
            normalization: 归一化方法
            rrfK: RRF 参数
            verbose: 是否打印过程日志

        Returns:
            融合后的结果列表
        """
        # 执行两种检索
        if verbose:
            print("🔍 执行 BM25 检索...")
        bm25Results = self.bm25Retriever.search(query, topK * 2)

        if verbose:
            print("🔍 执行向量检索...")
        vectorResults = self.vectorRetriever.search(query, topK * 2)

        # 融合结果
        if verbose:
            print(f"🔀 融合结果（策略: {strategy}）...")
        if strategy == "weighted":
            fusedResults = self.fuseWeighted(
                bm25Results, vectorResults, alpha, beta, normalization
            )
        elif strategy == "rrf":
            fusedResults = self.fuseRRF(bm25Results, vectorResults, rrfK)
        else:
            raise ValueError(f"不支持的融合策略: {strategy}")

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


# ============================================================
# HybridPlusRetriever - 改进混合检索
# ============================================================
