from __future__ import annotations

from typing import Any

import numpy as np

from core.retrieval.retrieverModules.bm25Plus import BM25PlusRetriever
from core.retrieval.retrieverModules.shared import _DEFAULT_VECTOR_MODEL
from core.retrieval.retrieverModules.vector import VectorRetriever


class HybridPlusRetriever:
    """改进的混合检索器"""

    def __init__(
        self,
        corpusFile: str,
        bm25IndexFile: str,
        vectorIndexFile: str,
        vectorEmbeddingFile: str,
        modelName: str = _DEFAULT_VECTOR_MODEL,
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
        print(" 初始化 BM25+ 检索器...")
        self.bm25Retriever = BM25PlusRetriever(corpusFile, bm25IndexFile, termsFile)
        if not self.bm25Retriever.loadIndex():
            print("  BM25+ 索引不存在，正在构建...")
            self.bm25Retriever.loadTermsMap()
            self.bm25Retriever.buildIndex()
            self.bm25Retriever.saveIndex()
        # 加载（或重新加载）评测感知术语映射
        self.bm25Retriever.loadTermsMap()

        # 初始化向量检索器
        print(" 初始化向量检索器...")
        self.vectorRetriever = VectorRetriever(
            corpusFile, modelName, vectorIndexFile, vectorEmbeddingFile
        )
        if not self.vectorRetriever.loadIndex():
            print("  向量索引不存在，正在构建...")
            self.vectorRetriever.buildIndex()
            self.vectorRetriever.saveIndex()

        print(" 混合检索器初始化完成\n")

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
        1. 根据查询难度动态调整 k 值
        2. 添加分数加权
        """
        # 计算查询难度
        if bm25Results:
            bm25Scores = [r["score"] for r in bm25Results]
            avgScore = np.mean(bm25Scores)
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
            if overlapRatio > 0.5:
                alpha = 0.5
                beta = 0.5
            else:
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
        useDirectLookup: bool = True,
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
            recallFactor: 召回因子
            useDirectLookup: 是否使用直接术语查找

        Returns:
            融合后的结果列表
        """
        # 1. 直接术语查找
        directResults = []
        directDocIds = set()
        if useDirectLookup and self.bm25Retriever.termsMap:
            expandedTerms = self.bm25Retriever.getExpandedTerms(query)
            directResults = self.bm25Retriever.directLookup(
                expandedTerms, baseScore=100.0
            )
            directDocIds = {r["doc_id"] for r in directResults}

        # 2. 执行混合检索
        recallTopK = topK * recallFactor

        bm25Results = self.bm25Retriever.search(
            query, recallTopK, expandQuery=expandQuery, returnAll=False
        )
        vectorResults = self.vectorRetriever.search(query, recallTopK)

        # 3. 融合 BM25 + 向量结果
        if strategy == "weighted":
            fusedResults = self.fuseWeightedImproved(
                bm25Results,
                vectorResults,
                topK + len(directDocIds),
                alpha,
                beta,
                normalization,
            )
        elif strategy == "rrf":
            fusedResults = self.fuseRRFImproved(
                bm25Results, vectorResults, topK + len(directDocIds), rrfK
            )
        else:
            raise ValueError(f"不支持的融合策略：{strategy}")

        # 4. 将直接查找结果注入到最终结果的顶部（去重）
        if directResults:
            filteredFused = [r for r in fusedResults if r["doc_id"] not in directDocIds]
            mergedResults = []
            for i, r in enumerate(directResults, 1):
                r["rank"] = i
                mergedResults.append(r)

            directCount = len(mergedResults)
            for i, r in enumerate(filteredFused, 1):
                r["rank"] = directCount + i
                mergedResults.append(r)

            return mergedResults[:topK]

        return fusedResults[:topK]

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


# ============================================================
# RerankerRetriever - 带重排序的检索
# ============================================================
