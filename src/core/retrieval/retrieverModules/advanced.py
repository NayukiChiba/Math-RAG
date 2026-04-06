from __future__ import annotations

import os
import pickle
import time
from typing import Any

import faiss
from sentence_transformers import SentenceTransformer

from core.retrieval.retrieverModules.shared import (
    _DEFAULT_RERANKER_MODEL,
    _DEFAULT_VECTOR_MODEL,
    _LOADER,
)


class AdvancedRetriever:
    """高级检索器 - 多路召回 + 重排序"""

    def __init__(
        self,
        corpusFile: str,
        bm25IndexFile: str,
        vectorIndexFile: str,
        vectorEmbeddingFile: str,
        modelName: str = _DEFAULT_VECTOR_MODEL,
        rerankerModel: str = _DEFAULT_RERANKER_MODEL,
        termsFile: str | None = None,
    ):
        """
        初始化高级检索器

        Args:
            corpusFile: 语料文件路径
            bm25IndexFile: BM25 索引文件路径
            vectorIndexFile: 向量索引文件路径
            vectorEmbeddingFile: 向量嵌入文件路径
            modelName: Sentence Transformer 模型名称
            rerankerModel: 重排序模型名称
            termsFile: 术语文件路径
        """
        self.corpusFile = corpusFile
        self.bm25IndexFile = bm25IndexFile
        self.vectorIndexFile = vectorIndexFile
        self.vectorEmbeddingFile = vectorEmbeddingFile
        self.modelName = modelName
        self.rerankerModelName = rerankerModel
        self.termsFile = termsFile

        # 延迟加载
        self._bm25 = None
        self._vectorModel = None
        self._vectorIndex = None
        self._reranker = None
        self._queryRewriter = None
        self._corpus = None

        # 预加载语料
        self._loadCorpus()

    def _loadCorpus(self) -> None:
        """加载语料库"""
        print(f" 加载语料：{self.corpusFile}")
        self._corpus = []
        if not os.path.exists(self.corpusFile):
            raise FileNotFoundError(
                f"语料文件不存在：{self.corpusFile}，请先运行语料构建流程"
            )
        self._corpus = _LOADER.jsonl(self.corpusFile)
        print(f" 已加载 {len(self._corpus)} 条语料")

    def _loadBM25(self):
        """懒加载 BM25"""
        if self._bm25 is not None:
            return

        print(" 加载 BM25 索引...")
        with open(self.bm25IndexFile, "rb") as f:
            indexData = pickle.load(f)

        self._bm25 = indexData["bm25"]
        print(" BM25 索引加载完成")

    def _loadVectorIndex(self):
        """懒加载向量索引"""
        if self._vectorIndex is not None:
            return

        print(f" 加载向量模型：{self.modelName}")
        self._vectorModel = SentenceTransformer(self.modelName)

        print(" 加载向量索引...")
        self._vectorIndex = faiss.read_index(self.vectorIndexFile)
        print(" 向量索引加载完成")

    def _loadReranker(self):
        """懒加载重排序器"""
        if self._reranker is not None:
            return

        # 检查是否已标记为不可用
        if getattr(self, "_rerankerUnavailable", False):
            return

        from sentence_transformers import CrossEncoder

        print(f" 加载重排序模型：{self.rerankerModelName}")
        try:
            self._reranker = CrossEncoder(self.rerankerModelName)
            print(" 重排序模型加载完成")
        except Exception as e:
            print(f"  重排序模型加载失败：{e}，将不使用重排序")
            self._rerankerUnavailable = True
            self._reranker = None

    def _loadQueryRewriter(self, termsFile: str | None = None):
        """懒加载查询改写器"""
        if self._queryRewriter is not None:
            return

        from core.retrieval.queryRewriter import QueryRewriter

        self._queryRewriter = QueryRewriter(termsFile)
        print(" 查询改写器加载完成")

    def _bm25Search(self, query: str, topK: int = 50) -> list[tuple[int, float]]:
        """BM25 检索"""
        self._loadBM25()

        tokens = query.split()
        scores = self._bm25.get_scores(tokens)

        topIndices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :topK
        ]
        return [(idx, float(scores[idx])) for idx in topIndices if scores[idx] > 0]

    def _vectorSearch(self, query: str, topK: int = 50) -> list[tuple[int, float]]:
        """向量检索"""
        self._loadVectorIndex()

        queryEmbedding = self._vectorModel.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(queryEmbedding)

        scores, indices = self._vectorIndex.search(queryEmbedding, topK)
        return [
            (idx, float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]

    def _rerankScores(
        self, query: str, candidates: list[tuple[int, str]]
    ) -> list[float] | None:
        """使用 Cross-Encoder 计算重排序分数"""
        self._loadReranker()

        if self._reranker is None:
            return None

        pairs = [[query, text] for _, text in candidates]
        scores = self._reranker.predict(pairs)
        return [float(s) for s in scores]

    def _getDocText(self, idx: int) -> str:
        """获取文档文本"""
        return self._corpus[idx].get("text", "")

    def search(
        self,
        query: str,
        topK: int = 10,
        recallTopK: int = 100,
        useReranker: bool = True,
        rewriteQuery: bool = True,
        bm25Weight: float = 0.4,
        vectorWeight: float = 0.3,
        rewriteWeight: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        高级检索 - 多路召回 + 重排序

        Args:
            query: 查询字符串
            topK: 返回的结果数量
            recallTopK: 每路召回的数量
            useReranker: 是否使用重排序
            rewriteQuery: 是否使用查询改写
            bm25Weight: BM25 权重
            vectorWeight: 向量检索权重
            rewriteWeight: 查询改写权重

        Returns:
            检索结果列表
        """
        startTime = time.time()

        # 1. 查询改写
        if rewriteQuery:
            self._loadQueryRewriter(self.termsFile)
            rewrittenQueries = self._queryRewriter.rewrite(query)
            print(f" 查询改写：{query} -> {rewrittenQueries}")
        else:
            rewrittenQueries = [query]

        # 2. 多路召回
        allCandidates = {}

        # BM25 召回
        bm25Results = self._bm25Search(query, recallTopK)
        for idx, score in bm25Results:
            allCandidates[idx] = {
                "bm25_score": score,
                "vector_score": 0.0,
                "rewrite_score": 0.0,
            }

        # 向量召回
        vectorResults = self._vectorSearch(query, recallTopK)
        for idx, score in vectorResults:
            if idx in allCandidates:
                allCandidates[idx]["vector_score"] = score
            else:
                allCandidates[idx] = {
                    "bm25_score": 0.0,
                    "vector_score": score,
                    "rewrite_score": 0.0,
                }

        # 查询改写召回（独立追踪 rewrite_score，使 rewriteWeight 参与融合）
        if rewriteQuery and len(rewrittenQueries) > 1:
            for rewrittenQuery in rewrittenQueries[1:4]:
                rewriteBm25 = self._bm25Search(rewrittenQuery, recallTopK // 3)
                for idx, score in rewriteBm25:
                    if idx in allCandidates:
                        allCandidates[idx]["rewrite_score"] = max(
                            allCandidates[idx]["rewrite_score"], score
                        )
                    else:
                        allCandidates[idx] = {
                            "bm25_score": 0.0,
                            "vector_score": 0.0,
                            "rewrite_score": score,
                        }

        print(f" 召回 {len(allCandidates)} 个候选文档")

        # 3. 计算融合分数
        if not allCandidates:
            return []

        # 百分位数归一化 - 使用 bisect 将复杂度从 O(n²) 降低到 O(n log n)
        import bisect

        def percentileNorm(scores: list[float]) -> list[float]:
            if not scores:
                return []
            sortedScores = sorted(scores)
            n = len(sortedScores)
            # bisect_right 返回 <= s 的元素个数，等价于之前的 sum(1 for x in ... if x <= s)
            return [bisect.bisect_right(sortedScores, s) / n for s in scores]

        bm25Scores = [c["bm25_score"] for c in allCandidates.values()]
        vectorScores = [c["vector_score"] for c in allCandidates.values()]
        rewriteScores = [c["rewrite_score"] for c in allCandidates.values()]

        bm25NormScores = percentileNorm(bm25Scores)
        vectorNormScores = percentileNorm(vectorScores)
        rewriteNormScores = percentileNorm(rewriteScores)

        docIds = list(allCandidates.keys())
        bm25ScoreMap = {docIds[i]: bm25NormScores[i] for i in range(len(docIds))}
        vectorScoreMap = {docIds[i]: vectorNormScores[i] for i in range(len(docIds))}
        rewriteScoreMap = {docIds[i]: rewriteNormScores[i] for i in range(len(docIds))}

        # 三路加权融合：bm25Weight + vectorWeight + rewriteWeight
        for idx, data in allCandidates.items():
            data["fused_score"] = (
                bm25Weight * bm25ScoreMap[idx]
                + vectorWeight * vectorScoreMap[idx]
                + rewriteWeight * rewriteScoreMap[idx]
            )

        # 4. 重排序
        if useReranker and len(allCandidates) > 0:
            sortedCandidates = sorted(
                allCandidates.items(),
                key=lambda x: x[1]["fused_score"],
                reverse=True,
            )[:50]

            candidates = [(idx, self._getDocText(idx)) for idx, _ in sortedCandidates]
            rerankScores = self._rerankScores(query, candidates)

            if rerankScores is not None:
                for (idx, _), score in zip(sortedCandidates, rerankScores):
                    allCandidates[idx]["reranker_score"] = score

                finalRanking = sorted(
                    allCandidates.items(),
                    key=lambda x: x[1].get("reranker_score", 0),
                    reverse=True,
                )
            else:
                print("  重排序不可用，使用融合分数排序")
                finalRanking = sorted(
                    allCandidates.items(),
                    key=lambda x: x[1]["fused_score"],
                    reverse=True,
                )
        else:
            finalRanking = sorted(
                allCandidates.items(),
                key=lambda x: x[1]["fused_score"],
                reverse=True,
            )

        # 5. 构建结果
        results = []
        for rank, (idx, data) in enumerate(finalRanking[:topK], 1):
            doc = self._corpus[idx]
            results.append(
                {
                    "rank": rank,
                    "doc_id": doc["doc_id"],
                    "term": doc["term"],
                    "subject": doc.get("subject", ""),
                    "score": data.get("reranker_score", data["fused_score"]),
                    "bm25_score": data["bm25_score"],
                    "vector_score": data["vector_score"],
                    "source": doc.get("source", ""),
                    "page": doc.get("page", None),
                }
            )

        endTime = time.time()
        print(f"⏱  检索耗时：{(endTime - startTime) * 1000:.2f}ms")

        return results

    def batchSearch(
        self,
        queries: list[str],
        topK: int = 10,
        **kwargs,
    ) -> dict[str, list[dict[str, Any]]]:
        """批量检索"""
        results = {}
        for query in queries:
            results[query] = self.search(query, topK, **kwargs)
        return results
