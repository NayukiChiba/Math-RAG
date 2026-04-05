from __future__ import annotations

import json
import os
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval.retrieverModules.shared import _DEFAULT_VECTOR_MODEL, _LOADER, USE_GPU


class VectorRetriever:
    """向量检索器"""

    # BGE 模型查询指令前缀（仅用于查询，不用于语料编码）
    BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索中文维基百科中的相关文章："

    def __init__(
        self,
        corpusFile: str,
        modelName: str = _DEFAULT_VECTOR_MODEL,
        indexFile: str | None = None,
        embeddingFile: str | None = None,
        termsFile: str | None = None,
    ):
        """
        初始化向量检索器

        Args:
            corpusFile: 语料文件路径（JSONL 格式）
            modelName: Sentence Transformer 模型名称
            indexFile: FAISS 索引文件路径，如果为 None 则不保存
            embeddingFile: 嵌入向量文件路径（.npz），如果为 None 则不保存
            termsFile: 术语文件路径（可选，用于查询同义词扩展）
        """
        self.corpusFile = corpusFile
        self.modelName = modelName
        self.indexFile = indexFile
        self.embeddingFile = embeddingFile
        self.corpus = []
        self.model = None
        self.index = None
        self.embeddings = None

        # 是否为 BGE 模型（需要查询指令前缀）
        self._isBgeModel = "bge" in self.modelName.lower()

        # 查询同义词扩展（仅在 termsFile 可用时初始化，否则退化为原始查询）
        if termsFile is not None:
            from retrieval.queryRewriter import QueryRewriter

            self.queryRewriter = QueryRewriter(termsFile)
        else:
            self.queryRewriter = None

    def loadModel(self) -> None:
        """加载 Sentence Transformer 模型"""
        if self.model is None:
            print(f" 加载模型: {self.modelName}")
            self.model = SentenceTransformer(self.modelName)
            print(
                f" 模型加载完成（维度: {self.model.get_sentence_embedding_dimension()}）"
            )

    def loadCorpus(self) -> None:
        """加载语料文件"""
        print(f" 加载语料: {self.corpusFile}")

        if not os.path.exists(self.corpusFile):
            raise FileNotFoundError(f"语料文件不存在: {self.corpusFile}")

        self.corpus = _LOADER.jsonl(self.corpusFile)

        print(f" 已加载 {len(self.corpus)} 条语料")

    def buildIndex(self, batchSize: int = 32) -> None:
        """
        构建 FAISS 索引

        Args:
            batchSize: 嵌入计算的批次大小
        """
        print(" 构建向量索引...")

        if not self.corpus:
            self.loadCorpus()

        # 加载模型
        self.loadModel()

        # 提取文本字段
        texts = [doc["text"] for doc in self.corpus]

        # 生成嵌入向量（批量处理）
        print(f" 生成嵌入向量（批次大小: {batchSize}）...")
        self.embeddings = self.model.encode(
            texts,
            batch_size=batchSize,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # 标准化向量（用于余弦相似度）
        print(" 标准化向量...")
        faiss.normalize_L2(self.embeddings)

        # 构建 FAISS 索引（内积，因向量已标准化，等价于余弦相似度）
        dimension = self.embeddings.shape[1]
        cpuIndex = faiss.IndexFlatIP(dimension)

        # 如果有 GPU，将索引迁移到 GPU
        if USE_GPU:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpuIndex)
            print(" 索引已迁移到 GPU")
        else:
            self.index = cpuIndex

        self.index.add(self.embeddings)

        deviceType = "GPU" if USE_GPU else "CPU"
        print(
            f" 索引构建完成（{self.index.ntotal} 条文档，维度: {dimension}，设备: {deviceType}）"
        )

    def saveIndex(self) -> None:
        """保存索引和嵌入到文件"""
        if self.index is None or self.embeddings is None:
            print("  索引未构建，无法保存")
            return

        # 保存 FAISS 索引
        if self.indexFile:
            print(f" 保存 FAISS 索引: {self.indexFile}")
            dirname = os.path.dirname(self.indexFile)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

            # 保存索引和元数据
            metadata = {
                "corpusFile": self.corpusFile,
                "corpusModTime": os.path.getmtime(self.corpusFile),
                "modelName": self.modelName,
                "dimension": self.embeddings.shape[1],
                "numDocs": len(self.corpus),
            }

            # FAISS 索引保存（GPU 索引需要先转回 CPU）
            if USE_GPU:
                cpuIndex = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpuIndex, self.indexFile)
            else:
                faiss.write_index(self.index, self.indexFile)

            # 元数据保存
            metadataFile = self.indexFile + ".meta.json"
            with open(metadataFile, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            print(" FAISS 索引已保存")

        # 保存嵌入向量
        if self.embeddingFile:
            print(f" 保存嵌入向量: {self.embeddingFile}")
            dirname = os.path.dirname(self.embeddingFile)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

            np.savez_compressed(
                self.embeddingFile,
                embeddings=self.embeddings,
                corpus=np.array(self.corpus, dtype=object),
            )

            print(" 嵌入向量已保存")

    def loadIndex(self) -> bool:
        """
        从文件加载索引和嵌入

        Returns:
            是否成功加载
        """
        if self.indexFile is None or not os.path.exists(self.indexFile):
            return False

        # 校验语料文件是否存在
        if not os.path.exists(self.corpusFile):
            print(f"  语料文件不存在: {self.corpusFile}")
            return False

        print(f" 加载索引: {self.indexFile}")

        try:
            # 加载元数据
            metadataFile = self.indexFile + ".meta.json"
            if not os.path.exists(metadataFile):
                print("  索引元数据不存在，建议重建索引")
                return False

            metadata = _LOADER.json(metadataFile)

            # 校验语料文件是否已变更
            currentCorpusModTime = os.path.getmtime(self.corpusFile)
            savedCorpusModTime = metadata.get("corpusModTime")

            if savedCorpusModTime is None:
                print("  索引中缺少语料时间戳，建议重建索引")
                return False

            if abs(currentCorpusModTime - savedCorpusModTime) > 1:
                print("  语料文件已更新，索引已过期，需要重建")
                return False

            # 校验模型是否一致
            if metadata.get("modelName") != self.modelName:
                print(
                    f"  模型不一致（保存: {metadata.get('modelName')}, 当前: {self.modelName}）"
                )
                print("建议重建索引或使用相同模型")
                return False

            # 加载 FAISS 索引
            cpuIndex = faiss.read_index(self.indexFile)

            # 如果有 GPU，将索引迁移到 GPU
            if USE_GPU:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpuIndex)
                print(" 索引已迁移到 GPU")
            else:
                self.index = cpuIndex

            # 加载嵌入和语料
            if self.embeddingFile and os.path.exists(self.embeddingFile):
                data = np.load(self.embeddingFile, allow_pickle=True)
                self.embeddings = data["embeddings"]
                self.corpus = data["corpus"].tolist()
            else:
                print("  嵌入文件不存在，重新加载语料")
                self.loadCorpus()

            # 加载模型（用于查询嵌入）
            self.loadModel()

            print(
                f" 已加载索引（{self.index.ntotal} 条文档，维度: {metadata['dimension']}）"
            )
            return True

        except Exception as e:
            print(f"  加载索引失败: {e}")
            return False

    def search(self, query: str, topK: int = 10) -> list[dict[str, Any]]:
        """
        单次查询（支持 BGE 指令前缀 + 同义词扩展平均嵌入）

        Args:
            query: 查询字符串
            topK: 返回的结果数量

        Returns:
            结果列表，每个结果包含 doc_id、term、score、rank
        """
        if self.index is None:
            raise RuntimeError("索引未构建，请先调用 buildIndex() 或 loadIndex()")

        if self.model is None:
            self.loadModel()

        # 1. 查询同义词扩展（扩展到更多近义词以提升召回；无 QueryRewriter 时退化为原始查询）
        if self.queryRewriter is not None:
            expandedTerms = self.queryRewriter.rewrite(query, maxTerms=8)
        else:
            expandedTerms = [query]

        # 2. 添加 BGE 查询指令前缀
        if self._isBgeModel:
            encodingTexts = [self.BGE_QUERY_INSTRUCTION + t for t in expandedTerms]
        else:
            encodingTexts = expandedTerms

        # 3. 编码所有扩展词
        allEmbeddings = self.model.encode(encodingTexts, convert_to_numpy=True)

        # 4. 取加权平均（原始查询权重更高）
        weights = np.array([2.0] + [1.0] * (len(expandedTerms) - 1), dtype=np.float32)
        weights = weights / weights.sum()
        queryEmbedding = np.average(allEmbeddings, axis=0, weights=weights)
        queryEmbedding = queryEmbedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(queryEmbedding)

        # 执行搜索
        scores, indices = self.index.search(queryEmbedding, topK)

        # 构建结果
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            if idx == -1:
                continue

            doc = self.corpus[idx]
            results.append(
                {
                    "rank": rank,
                    "doc_id": doc["doc_id"],
                    "term": doc["term"],
                    "subject": doc.get("subject", ""),
                    "score": float(score),
                    "source": doc.get("source", ""),
                    "page": doc.get("page", None),
                }
            )

        return results

    def batchSearch(
        self, queries: list[str], topK: int = 10
    ) -> dict[str, list[dict[str, Any]]]:
        """
        批量查询

        Args:
            queries: 查询字符串列表
            topK: 每个查询返回的结果数量

        Returns:
            字典，键为查询字符串，值为结果列表
        """
        results = {}
        for query in queries:
            results[query] = self.search(query, topK)
        return results


# ============================================================
# BM25PlusRetriever - BM25+ 改进检索
# ============================================================
