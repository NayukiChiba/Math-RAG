from __future__ import annotations

import os
import pickle
from typing import Any

import faiss
from sentence_transformers import SentenceTransformer

from retrieval.retrieverModules.shared import _DEFAULT_VECTOR_MODEL


class RerankerRetriever:
    """带重排序的检索器"""

    def __init__(
        self,
        corpusFile: str,
        bm25IndexFile: str,
        vectorIndexFile: str,
        vectorEmbeddingFile: str,
        modelName: str = _DEFAULT_VECTOR_MODEL,
        rerankerModel: str = "bge-reranker-base",
    ):
        """
        初始化带重排序的检索器

        Args:
            corpusFile: 语料文件路径
            bm25IndexFile: BM25 索引文件路径
            vectorIndexFile: 向量索引文件路径
            vectorEmbeddingFile: 向量嵌入文件路径
            modelName: Sentence Transformer 模型名称
            rerankerModel: 重排序模型名称
        """
        self.corpusFile = corpusFile
        self.rerankerModelName = rerankerModel
        self.corpus = []
        self.reranker = None
        self.bm25 = None
        self.vectorIndex = None
        self.embeddings = None
        self.vectorModel = None

        # 加载 BM25 索引
        self._loadBM25Index(bm25IndexFile)

        # 加载向量索引
        self._loadVectorIndex(vectorIndexFile, vectorEmbeddingFile, modelName)

        # 加载重排序模型
        self._loadReranker()

    def _loadBM25Index(self, indexFile: str) -> None:
        """加载 BM25 索引"""
        print("📂 加载 BM25 索引...")

        if not os.path.exists(indexFile):
            raise FileNotFoundError(f"BM25 索引文件不存在：{indexFile}")

        with open(indexFile, "rb") as f:
            indexData = pickle.load(f)

        self.bm25 = indexData["bm25"]
        self.corpus = indexData["corpus"]
        print(f"✅ 已加载 BM25 索引（{len(self.corpus)} 条文档）")

    def _loadVectorIndex(
        self, indexFile: str, embeddingFile: str, modelName: str
    ) -> None:
        """加载向量索引"""
        print("📂 加载向量索引...")

        # 加载向量模型
        print(f"🤖 加载向量模型：{modelName}")
        self.vectorModel = SentenceTransformer(modelName)

        # 加载 FAISS 索引
        if os.path.exists(indexFile):
            self.vectorIndex = faiss.read_index(indexFile)
            print("✅ 已加载 FAISS 索引")
        else:
            print(f"⚠️  向量索引不存在：{indexFile}")
            self.vectorIndex = None

    def _loadReranker(self) -> None:
        """加载重排序模型"""
        print(f"🤖 加载重排序模型：{self.rerankerModelName}")

        try:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(self.rerankerModelName)
            print("✅ 重排序模型加载完成")
        except ImportError:
            print("⚠️  未安装 CrossEncoder，重排序功能将不可用")
            self.reranker = None
        except Exception as e:
            print(f"⚠️  重排序模型加载失败：{e}")
            self.reranker = None

    def _retrieveCandidates(self, query: str, topK: int = 50) -> list[dict[str, Any]]:
        """
        检索候选文档

        Args:
            query: 查询字符串
            topK: 候选数量

        Returns:
            候选文档列表
        """
        candidates = {}

        # BM25 检索
        if self.bm25 is not None:
            tokens = query.split()
            scores = self.bm25.get_scores(tokens)

            topIndices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[: topK // 2]

            for idx in topIndices:
                if scores[idx] > 0:
                    doc = self.corpus[idx]
                    candidates[idx] = {
                        "doc_idx": idx,
                        "doc_id": doc["doc_id"],
                        "term": doc["term"],
                        "subject": doc.get("subject", ""),
                        "text": doc["text"],
                        "bm25_score": float(scores[idx]),
                        "source": doc.get("source", ""),
                        "page": doc.get("page", None),
                    }

        # 向量检索
        if self.vectorIndex is not None and self.vectorModel is not None:
            queryEmbedding = self.vectorModel.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(queryEmbedding)

            scores, indices = self.vectorIndex.search(queryEmbedding, topK // 2)

            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                doc = self.corpus[idx]
                if idx not in candidates:
                    candidates[idx] = {
                        "doc_idx": idx,
                        "doc_id": doc["doc_id"],
                        "term": doc["term"],
                        "subject": doc.get("subject", ""),
                        "text": doc["text"],
                        "vector_score": float(score),
                        "source": doc.get("source", ""),
                        "page": doc.get("page", None),
                    }
                else:
                    candidates[idx]["vector_score"] = float(score)

        return list(candidates.values())

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        topK: int = 10,
        useReranker: bool = True,
    ) -> list[dict[str, Any]]:
        """
        重排序候选文档

        Args:
            query: 查询字符串
            candidates: 候选文档列表
            topK: 返回的结果数量
            useReranker: 是否使用 Cross-Encoder 重排序

        Returns:
            重排序后的结果列表
        """
        if not candidates:
            return []

        if useReranker and self.reranker is not None:
            print(f"🔄 使用重排序模型对 {len(candidates)} 个候选进行重排序...")

            pairs = [[query, c["text"]] for c in candidates]
            rerankScores = self.reranker.predict(pairs)

            for i, candidate in enumerate(candidates):
                candidate["reranker_score"] = float(rerankScores[i])

            sortedCandidates = sorted(
                candidates, key=lambda x: x["reranker_score"], reverse=True
            )
        else:
            print("📊 按综合分数排序...")
            for candidate in candidates:
                bm25Score = candidate.get("bm25_score", 0)
                vectorScore = candidate.get("vector_score", 0)
                candidate["combined_score"] = 0.5 * bm25Score + 0.5 * vectorScore

            sortedCandidates = sorted(
                candidates, key=lambda x: x["combined_score"], reverse=True
            )

        # 返回 topK
        results = []
        for rank, candidate in enumerate(sortedCandidates[:topK], 1):
            results.append(
                {
                    "rank": rank,
                    "doc_id": candidate["doc_id"],
                    "term": candidate["term"],
                    "subject": candidate["subject"],
                    "score": candidate.get(
                        "reranker_score", candidate.get("combined_score", 0)
                    ),
                    "source": candidate["source"],
                    "page": candidate.get("page"),
                    "bm25_score": candidate.get("bm25_score", 0),
                    "vector_score": candidate.get("vector_score", 0),
                }
            )

        return results

    def search(
        self,
        query: str,
        topK: int = 10,
        recallTopK: int = 50,
        useReranker: bool = True,
    ) -> list[dict[str, Any]]:
        """
        带重排序的检索

        Args:
            query: 查询字符串
            topK: 返回的结果数量
            recallTopK: 召回候选数量
            useReranker: 是否使用重排序

        Returns:
            检索结果列表
        """
        print(f"📥 召回候选文档（top{recallTopK}）...")
        candidates = self._retrieveCandidates(query, recallTopK)

        print(f"✅ 召回 {len(candidates)} 个候选文档")

        results = self.rerank(query, candidates, topK, useReranker)
        return results

    def batchSearch(
        self,
        queries: list[str],
        topK: int = 10,
        recallTopK: int = 50,
        useReranker: bool = True,
    ) -> dict[str, list[dict[str, Any]]]:
        """批量检索"""
        results = {}
        for query in queries:
            results[query] = self.search(query, topK, recallTopK, useReranker)
        return results


# ============================================================
# AdvancedRetriever - 高级多路召回 + 重排序
# ============================================================
