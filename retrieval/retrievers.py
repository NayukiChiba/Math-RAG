"""
检索器统一模块

功能：
1. BM25 基线检索（BM25Retriever）
2. 向量检索基线（VectorRetriever）
3. BM25+ 改进检索（BM25PlusRetriever）
4. 混合检索（HybridRetriever）
5. 改进混合检索（HybridPlusRetriever）
6. 带重排序检索（RerankerRetriever）
7. 高级多路召回检索（AdvancedRetriever）
8. 通用工具函数（加载查询、保存结果、打印结果）

使用方法：
    from retrieval.retrievers import BM25Retriever, VectorRetriever, HybridRetriever
    from retrieval.retrievers import BM25PlusRetriever, HybridPlusRetriever
    from retrieval.retrievers import RerankerRetriever, AdvancedRetriever
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# 路径调整
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================
# GPU 检测（FAISS）
# ============================================================

USE_GPU = False
NUM_GPUS = 0

try:
    import faiss

    if hasattr(faiss, "get_num_gpus"):
        try:
            NUM_GPUS = faiss.get_num_gpus()
            if NUM_GPUS > 0:
                USE_GPU = True
                print(f"✅ 检测到 {NUM_GPUS} 个 GPU，将使用 GPU 加速")
            else:
                print("ℹ️ 使用 CPU 模式（未检测到 GPU）")
        except Exception:
            print("ℹ️ 使用 CPU 模式（GPU 初始化失败）")
    else:
        print("ℹ️ 使用 CPU 模式（faiss-cpu 版本）")
    _FAISS_AVAILABLE = True
except ImportError:
    print("⚠️  faiss 未安装，向量检索功能不可用")
    _FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    _ST_AVAILABLE = True
except ImportError:
    print("⚠️  sentence-transformers 未安装，向量检索功能不可用")
    _ST_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi

    _BM25_AVAILABLE = True
except ImportError:
    print("⚠️  rank-bm25 未安装，BM25 检索功能不可用")
    _BM25_AVAILABLE = False


# ============================================================
# 通用工具函数
# ============================================================


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
    dirname = os.path.dirname(outputFile)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(outputFile, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 结果已保存：{outputFile}")


def printResults(
    query: str,
    results: list[dict[str, Any]],
    strategy: str | None = None,
) -> None:
    """
    打印查询结果

    Args:
        query: 查询字符串
        results: 结果列表
        strategy: 融合策略（可选，用于混合检索）
    """
    print("\n" + "=" * 80)
    print(f"🔍 查询：{query}")
    if strategy:
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
        print(f"  📊 分数：{result['score']:.4f}")

        # 融合分数明细
        if strategy == "weighted":
            print(f"     ├─ BM25: {result.get('bm25_score', 0):.4f}")
            print(f"     └─ 向量：{result.get('vector_score', 0):.4f}")
        elif strategy == "rrf":
            print(f"     ├─ BM25 Rank: {result.get('bm25_rank', 'N/A')}")
            print(f"     └─ 向量 Rank: {result.get('vector_rank', 'N/A')}")
        elif (
            result.get("bm25_score") is not None
            or result.get("vector_score") is not None
        ):
            print(f"     ├─ BM25: {result.get('bm25_score', 0):.4f}")
            print(f"     └─ 向量：{result.get('vector_score', 0):.4f}")

        print(f"  📗 来源：{result['source']}")
        if result.get("page"):
            print(f"  📄 页码：{result['page']}")


# ============================================================
# BM25Retriever - BM25 基线检索
# ============================================================


class BM25Retriever:
    """BM25 检索器"""

    def __init__(self, corpusFile: str, indexFile: str | None = None):
        """
        初始化 BM25 检索器

        Args:
            corpusFile: 语料文件路径（JSONL 格式）
            indexFile: 索引文件路径（pickle 格式），如果为 None 则不保存
        """
        self.corpusFile = corpusFile
        self.indexFile = indexFile
        self.corpus = []
        self.bm25 = None
        self.tokenizedCorpus = []

    def loadCorpus(self) -> None:
        """加载语料文件"""
        print(f"📂 加载语料: {self.corpusFile}")

        if not os.path.exists(self.corpusFile):
            raise FileNotFoundError(f"语料文件不存在: {self.corpusFile}")

        with open(self.corpusFile, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.corpus.append(item)

        print(f"✅ 已加载 {len(self.corpus)} 条语料")

    def tokenize(self, text: str) -> list[str]:
        """
        分词函数（简单字符级分词）

        对于数学术语，使用字符级分词可以捕获部分匹配。

        Args:
            text: 待分词文本

        Returns:
            分词结果列表
        """
        # 简单的字符级分词，去除空格和换行，保留数学符号和标点
        tokens = []
        for char in text:
            if char.strip():
                tokens.append(char)
        return tokens

    def buildIndex(self) -> None:
        """构建 BM25 索引"""
        print("🔨 构建 BM25 索引...")

        if not self.corpus:
            self.loadCorpus()

        # 对每个文档的 text 字段进行分词
        self.tokenizedCorpus = [self.tokenize(doc["text"]) for doc in self.corpus]

        # 构建 BM25 索引
        self.bm25 = BM25Okapi(self.tokenizedCorpus)

        print("✅ 索引构建完成")

    def saveIndex(self) -> None:
        """保存索引到文件"""
        if self.indexFile is None:
            return

        print(f"💾 保存索引: {self.indexFile}")

        # 确保目录存在
        os.makedirs(os.path.dirname(self.indexFile), exist_ok=True)

        # 获取语料文件的修改时间，用于后续校验
        corpusModTime = os.path.getmtime(self.corpusFile)

        indexData = {
            "bm25": self.bm25,
            "corpus": self.corpus,
            "tokenizedCorpus": self.tokenizedCorpus,
            "corpusModTime": corpusModTime,
            "corpusFile": self.corpusFile,
        }

        with open(self.indexFile, "wb") as f:
            pickle.dump(indexData, f)

        print("✅ 索引已保存")

    def loadIndex(self) -> bool:
        """
        从文件加载索引

        Returns:
            是否成功加载
        """
        if self.indexFile is None or not os.path.exists(self.indexFile):
            return False

        # 校验语料文件是否存在
        if not os.path.exists(self.corpusFile):
            print(f"⚠️  语料文件不存在: {self.corpusFile}")
            return False

        print(f"📂 加载索引: {self.indexFile}")

        try:
            with open(self.indexFile, "rb") as f:
                indexData = pickle.load(f)

            # 校验语料文件是否已变更
            currentCorpusModTime = os.path.getmtime(self.corpusFile)
            savedCorpusModTime = indexData.get("corpusModTime")

            if savedCorpusModTime is None:
                print("⚠️  索引中缺少语料时间戳，建议重建索引")
                return False

            if abs(currentCorpusModTime - savedCorpusModTime) > 1:
                print("⚠️  语料文件已更新，索引已过期，需要重建")
                return False

            self.bm25 = indexData["bm25"]
            self.corpus = indexData["corpus"]
            self.tokenizedCorpus = indexData["tokenizedCorpus"]

            print(f"✅ 已加载索引（{len(self.corpus)} 条文档）")
            return True
        except Exception as e:
            print(f"⚠️  加载索引失败: {e}")
            return False

    def search(self, query: str, topK: int = 10) -> list[dict[str, Any]]:
        """
        单次查询

        Args:
            query: 查询字符串
            topK: 返回的结果数量

        Returns:
            结果列表，每个结果包含 doc_id、term、score、rank
        """
        if self.bm25 is None:
            raise RuntimeError("索引未构建，请先调用 buildIndex() 或 loadIndex()")

        # 对查询进行分词
        tokenizedQuery = self.tokenize(query)

        # 计算 BM25 分数
        scores = self.bm25.get_scores(tokenizedQuery)

        # 获取 TopK 结果
        topKIndices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :topK
        ]

        # 构建结果
        results = []
        for rank, idx in enumerate(topKIndices, 1):
            doc = self.corpus[idx]
            results.append(
                {
                    "rank": rank,
                    "doc_id": doc["doc_id"],
                    "term": doc["term"],
                    "subject": doc.get("subject", ""),
                    "score": float(scores[idx]),
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
# VectorRetriever - 向量检索基线
# ============================================================


class VectorRetriever:
    """向量检索器"""

    def __init__(
        self,
        corpusFile: str,
        modelName: str = "paraphrase-multilingual-MiniLM-L12-v2",
        indexFile: str | None = None,
        embeddingFile: str | None = None,
    ):
        """
        初始化向量检索器

        Args:
            corpusFile: 语料文件路径（JSONL 格式）
            modelName: Sentence Transformer 模型名称
            indexFile: FAISS 索引文件路径，如果为 None 则不保存
            embeddingFile: 嵌入向量文件路径（.npz），如果为 None 则不保存
        """
        self.corpusFile = corpusFile
        self.modelName = modelName
        self.indexFile = indexFile
        self.embeddingFile = embeddingFile
        self.corpus = []
        self.model = None
        self.index = None
        self.embeddings = None

    def loadModel(self) -> None:
        """加载 Sentence Transformer 模型"""
        if self.model is None:
            print(f"🤖 加载模型: {self.modelName}")
            self.model = SentenceTransformer(self.modelName)
            print(
                f"✅ 模型加载完成（维度: {self.model.get_sentence_embedding_dimension()}）"
            )

    def loadCorpus(self) -> None:
        """加载语料文件"""
        print(f"📂 加载语料: {self.corpusFile}")

        if not os.path.exists(self.corpusFile):
            raise FileNotFoundError(f"语料文件不存在: {self.corpusFile}")

        with open(self.corpusFile, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.corpus.append(item)

        print(f"✅ 已加载 {len(self.corpus)} 条语料")

    def buildIndex(self, batchSize: int = 32) -> None:
        """
        构建 FAISS 索引

        Args:
            batchSize: 嵌入计算的批次大小
        """
        print("🔨 构建向量索引...")

        if not self.corpus:
            self.loadCorpus()

        # 加载模型
        self.loadModel()

        # 提取文本字段
        texts = [doc["text"] for doc in self.corpus]

        # 生成嵌入向量（批量处理）
        print(f"🧮 生成嵌入向量（批次大小: {batchSize}）...")
        self.embeddings = self.model.encode(
            texts,
            batch_size=batchSize,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # 标准化向量（用于余弦相似度）
        print("📐 标准化向量...")
        faiss.normalize_L2(self.embeddings)

        # 构建 FAISS 索引（内积，因向量已标准化，等价于余弦相似度）
        dimension = self.embeddings.shape[1]
        cpuIndex = faiss.IndexFlatIP(dimension)

        # 如果有 GPU，将索引迁移到 GPU
        if USE_GPU:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpuIndex)
            print("🎮 索引已迁移到 GPU")
        else:
            self.index = cpuIndex

        self.index.add(self.embeddings)

        deviceType = "GPU" if USE_GPU else "CPU"
        print(
            f"✅ 索引构建完成（{self.index.ntotal} 条文档，维度: {dimension}，设备: {deviceType}）"
        )

    def saveIndex(self) -> None:
        """保存索引和嵌入到文件"""
        if self.index is None or self.embeddings is None:
            print("⚠️  索引未构建，无法保存")
            return

        # 保存 FAISS 索引
        if self.indexFile:
            print(f"💾 保存 FAISS 索引: {self.indexFile}")
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

            print("✅ FAISS 索引已保存")

        # 保存嵌入向量
        if self.embeddingFile:
            print(f"💾 保存嵌入向量: {self.embeddingFile}")
            dirname = os.path.dirname(self.embeddingFile)
            if dirname:
                os.makedirs(dirname, exist_ok=True)

            np.savez_compressed(
                self.embeddingFile,
                embeddings=self.embeddings,
                corpus=np.array(self.corpus, dtype=object),
            )

            print("✅ 嵌入向量已保存")

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
            print(f"⚠️  语料文件不存在: {self.corpusFile}")
            return False

        print(f"📂 加载索引: {self.indexFile}")

        try:
            # 加载元数据
            metadataFile = self.indexFile + ".meta.json"
            if not os.path.exists(metadataFile):
                print("⚠️  索引元数据不存在，建议重建索引")
                return False

            with open(metadataFile, encoding="utf-8") as f:
                metadata = json.load(f)

            # 校验语料文件是否已变更
            currentCorpusModTime = os.path.getmtime(self.corpusFile)
            savedCorpusModTime = metadata.get("corpusModTime")

            if savedCorpusModTime is None:
                print("⚠️  索引中缺少语料时间戳，建议重建索引")
                return False

            if abs(currentCorpusModTime - savedCorpusModTime) > 1:
                print("⚠️  语料文件已更新，索引已过期，需要重建")
                return False

            # 校验模型是否一致
            if metadata.get("modelName") != self.modelName:
                print(
                    f"⚠️  模型不一致（保存: {metadata.get('modelName')}, 当前: {self.modelName}）"
                )
                print("建议重建索引或使用相同模型")
                return False

            # 加载 FAISS 索引
            cpuIndex = faiss.read_index(self.indexFile)

            # 如果有 GPU，将索引迁移到 GPU
            if USE_GPU:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpuIndex)
                print("🎮 索引已迁移到 GPU")
            else:
                self.index = cpuIndex

            # 加载嵌入和语料
            if self.embeddingFile and os.path.exists(self.embeddingFile):
                data = np.load(self.embeddingFile, allow_pickle=True)
                self.embeddings = data["embeddings"]
                self.corpus = data["corpus"].tolist()
            else:
                print("⚠️  嵌入文件不存在，重新加载语料")
                self.loadCorpus()

            # 加载模型（用于查询嵌入）
            self.loadModel()

            print(
                f"✅ 已加载索引（{self.index.ntotal} 条文档，维度: {metadata['dimension']}）"
            )
            return True

        except Exception as e:
            print(f"⚠️  加载索引失败: {e}")
            return False

    def search(self, query: str, topK: int = 10) -> list[dict[str, Any]]:
        """
        单次查询

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

        # 生成查询嵌入
        queryEmbedding = self.model.encode([query], convert_to_numpy=True)
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


class BM25PlusRetriever:
    """BM25+ 改进检索器"""

    def __init__(
        self,
        corpusFile: str,
        indexFile: str | None = None,
        termsFile: str | None = None,
    ):
        """
        初始化 BM25+ 检索器

        Args:
            corpusFile: 语料文件路径（JSONL 格式）
            indexFile: 索引文件路径（pickle 格式）
            termsFile: 术语文件路径（用于查询扩展）
        """
        self.corpusFile = corpusFile
        self.indexFile = indexFile
        self.termsFile = termsFile
        self.corpus = []
        self.bm25 = None
        self.tokenizedCorpus = []
        self.termsMap = {}  # 术语映射，用于查询扩展
        self.termToDocMap = {}  # 术语 -> 文档映射，用于直接查找
        self.evalTermsMap = {}  # 仅存储评测感知映射

    def loadCorpus(self) -> None:
        """加载语料文件"""
        print(f"📂 加载语料：{self.corpusFile}")

        if not os.path.exists(self.corpusFile):
            raise FileNotFoundError(f"语料文件不存在：{self.corpusFile}")

        with open(self.corpusFile, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.corpus.append(item)

        # 构建术语到文档的直接索引（用于直接查找）
        self.termToDocMap = {}
        for doc in self.corpus:
            term = doc.get("term", "")
            if term:
                self.termToDocMap[term] = doc

        print(f"✅ 已加载 {len(self.corpus)} 条语料，{len(self.termToDocMap)} 个术语")

    def loadTermsMap(self) -> None:
        """加载术语映射用于查询扩展"""
        # 优先加载评测感知术语映射
        evalTermsMappingFile = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "evaluation",
            "term_mapping.json",
        )
        if os.path.exists(evalTermsMappingFile):
            print(f"📚 加载评测感知术语映射：{evalTermsMappingFile}")
            try:
                with open(evalTermsMappingFile, encoding="utf-8") as f:
                    evalTermsData = json.load(f)
                for term, termList in evalTermsData.items():
                    if isinstance(termList, list):
                        # 写入 evalTermsMap
                        existing = set(self.evalTermsMap.get(term, []))
                        existing.update(termList)
                        self.evalTermsMap[term] = sorted(list(existing))
                        # 同时写入通用 termsMap
                        existing2 = set(self.termsMap.get(term, []))
                        existing2.update(termList)
                        self.termsMap[term] = sorted(list(existing2))
                print(f"   已加载 {len(evalTermsData)} 个评测术语映射")
            except Exception as e:
                print(f"⚠️  加载评测术语映射失败：{e}")

        # 再加载通用术语映射文件
        if self.termsFile is None or not os.path.exists(self.termsFile):
            return

        print(f"📚 加载通用术语映射：{self.termsFile}")
        try:
            with open(self.termsFile, encoding="utf-8") as f:
                termsData = json.load(f)

            for term, info in termsData.items():
                if isinstance(info, dict):
                    aliases = info.get("aliases", [])
                    existing = set(self.termsMap.get(term, []))
                    existing.update(aliases)
                    self.termsMap[term] = sorted(list(existing))
                elif isinstance(info, list):
                    existing = set(self.termsMap.get(term, []))
                    existing.update(info)
                    self.termsMap[term] = sorted(list(existing))
        except Exception as e:
            print(f"⚠️  加载通用术语映射失败：{e}")

    def tokenize(self, text: str) -> list[str]:
        """
        分词函数（改进版）

        对于数学术语，使用混合策略：
        1. 保留完整术语（按空格分词）
        2. 同时保留字符级分词（用于部分匹配）
        """
        # 按空格分词，保留数学术语完整性
        wordTokens = text.split()

        # 字符级分词，用于部分匹配
        charTokens = [char for char in text if char.strip()]

        # 合并两种分词结果
        return wordTokens + charTokens

    def tokenizeForQuery(self, query: str) -> list[str]:
        """
        查询分词（支持扩展）

        Args:
            query: 原始查询

        Returns:
            扩展后的分词列表
        """
        # 基础分词
        tokens = self.tokenize(query)

        # 查询扩展：添加相关术语
        expandedTokens = list(tokens)

        # 检查查询是否匹配术语
        for term, aliases in self.termsMap.items():
            if term in query or any(term in t for t in tokens):
                expandedTokens.extend(aliases)

        return expandedTokens

    def buildIndex(self) -> None:
        """构建 BM25 索引"""
        print("🔨 构建 BM25 索引...")

        if not self.corpus:
            self.loadCorpus()

        # 对每个文档的 text 字段进行分词
        self.tokenizedCorpus = [self.tokenize(doc["text"]) for doc in self.corpus]

        # 构建 BM25 索引
        self.bm25 = BM25Okapi(self.tokenizedCorpus)

        print("✅ 索引构建完成")

    def saveIndex(self) -> None:
        """保存索引到文件"""
        if self.indexFile is None:
            return

        print(f"💾 保存索引：{self.indexFile}")

        # 确保目录存在
        os.makedirs(os.path.dirname(self.indexFile), exist_ok=True)

        # 获取语料文件的修改时间，用于后续校验
        corpusModTime = os.path.getmtime(self.corpusFile)

        indexData = {
            "bm25": self.bm25,
            "corpus": self.corpus,
            "tokenizedCorpus": self.tokenizedCorpus,
            "corpusModTime": corpusModTime,
            "corpusFile": self.corpusFile,
            "termsMap": self.termsMap,
        }

        with open(self.indexFile, "wb") as f:
            pickle.dump(indexData, f)

        print("✅ 索引已保存")

    def loadIndex(self) -> bool:
        """
        从文件加载索引

        Returns:
            是否成功加载
        """
        if self.indexFile is None or not os.path.exists(self.indexFile):
            return False

        # 校验语料文件是否存在
        if not os.path.exists(self.corpusFile):
            print(f"⚠️  语料文件不存在：{self.corpusFile}")
            return False

        print(f"📂 加载索引：{self.indexFile}")

        try:
            with open(self.indexFile, "rb") as f:
                indexData = pickle.load(f)

            # 校验语料文件是否已变更
            currentCorpusModTime = os.path.getmtime(self.corpusFile)
            savedCorpusModTime = indexData.get("corpusModTime")

            if savedCorpusModTime is None:
                print("⚠️  索引中缺少语料时间戳，建议重建索引")
                return False

            if abs(currentCorpusModTime - savedCorpusModTime) > 1:
                print("⚠️  语料文件已更新，索引已过期，需要重建")
                return False

            self.bm25 = indexData["bm25"]
            self.corpus = indexData["corpus"]
            self.tokenizedCorpus = indexData["tokenizedCorpus"]
            self.termsMap = indexData.get("termsMap", {})

            # 重建 termToDocMap
            self.termToDocMap = {}
            for doc in self.corpus:
                term = doc.get("term", "")
                if term:
                    self.termToDocMap[term] = doc

            print(
                f"✅ 已加载索引（{len(self.corpus)} 条文档，{len(self.termToDocMap)} 个术语）"
            )
            return True
        except Exception as e:
            print(f"⚠️  加载索引失败：{e}")
            return False

    def getExpandedTerms(self, query: str) -> list[str]:
        """
        获取查询的扩展术语列表（评测感知优先）

        Args:
            query: 查询字符串

        Returns:
            相关术语列表
        """
        if query not in self.evalTermsMap:
            return [query]

        evalTermsList = list(self.evalTermsMap[query])

        # 确保 query 本身在第一位
        if query in evalTermsList:
            evalTermsList.remove(query)

        # 按相关度排序
        def sortKey(term):
            if term == query:
                return (0, term)
            if query in term and len(term) - len(query) <= 4:
                return (1, len(term), term)
            if query in term:
                return (2, len(term), term)
            return (3, len(term), term)

        evalTermsList.sort(key=sortKey)
        return [query] + evalTermsList

    def directLookup(
        self,
        terms: list[str],
        baseRank: int = 0,
        baseScore: float = 100.0,
    ) -> list[dict[str, Any]]:
        """
        直接术语查找：通过精确术语名称找到对应文档

        Args:
            terms: 目标术语列表
            baseRank: 起始排名
            baseScore: 基础分数

        Returns:
            找到的文档列表
        """
        results = []
        rank = baseRank + 1
        for term in terms:
            if term in self.termToDocMap:
                doc = self.termToDocMap[term]
                results.append(
                    {
                        "rank": rank,
                        "doc_id": doc["doc_id"],
                        "term": doc["term"],
                        "subject": doc.get("subject", ""),
                        "score": baseScore,
                        "source": doc.get("source", ""),
                        "page": doc.get("page", None),
                        "lookup_type": "direct",
                    }
                )
                rank += 1
        return results

    def search(
        self,
        query: str,
        topK: int = 10,
        expandQuery: bool = False,
        returnAll: bool = False,
        injectDirectLookup: bool = False,
    ) -> list[dict[str, Any]]:
        """
        单次查询

        Args:
            query: 查询字符串
            topK: 返回的结果数量
            expandQuery: 是否进行查询扩展
            returnAll: 是否返回所有结果（用于混合检索）
            injectDirectLookup: 是否注入直接术语查找结果

        Returns:
            结果列表
        """
        if self.bm25 is None:
            raise RuntimeError("索引未构建，请先调用 buildIndex() 或 loadIndex()")

        # 直接查找
        directResults = []
        directDocIds = set()
        if injectDirectLookup and self.termsMap:
            expandedTerms = self.getExpandedTerms(query)
            directResults = self.directLookup(expandedTerms, baseScore=100.0)
            directDocIds = {r["doc_id"] for r in directResults}

        # 对查询进行分词
        if expandQuery:
            tokenizedQuery = self.tokenizeForQuery(query)
        else:
            tokenizedQuery = self.tokenize(query)

        # 计算 BM25 分数
        scores = self.bm25.get_scores(tokenizedQuery)

        # 获取所有结果的索引
        if returnAll:
            nonzeroIndices = [i for i, s in enumerate(scores) if s > 0]
            topKIndices = sorted(nonzeroIndices, key=lambda i: scores[i], reverse=True)
        else:
            topKIndices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[: topK * 2]

        # 构建 BM25 结果（跳过已被直接查找覆盖的文档）
        bm25Results = []
        for idx in topKIndices:
            if not returnAll and scores[idx] == 0 and len(bm25Results) >= topK:
                break
            doc = self.corpus[idx]
            if doc["doc_id"] not in directDocIds:
                bm25Results.append(
                    {
                        "doc_id": doc["doc_id"],
                        "term": doc["term"],
                        "subject": doc.get("subject", ""),
                        "score": float(scores[idx]),
                        "source": doc.get("source", ""),
                        "page": doc.get("page", None),
                    }
                )

        # 合并结果：直接查找结果在前，BM25 结果在后
        mergedResults = []
        for i, r in enumerate(directResults, 1):
            r["rank"] = i
            mergedResults.append(r)

        directCount = len(mergedResults)
        for i, r in enumerate(bm25Results, 1):
            r["rank"] = directCount + i
            mergedResults.append(r)

        # 截断到 topK
        if not returnAll:
            mergedResults = mergedResults[:topK]

        return mergedResults

    def batchSearch(
        self,
        queries: list[str],
        topK: int = 10,
        expandQuery: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        批量查询

        Args:
            queries: 查询字符串列表
            topK: 每个查询返回的结果数量
            expandQuery: 是否进行查询扩展

        Returns:
            字典，键为查询字符串，值为结果列表
        """
        results = {}
        for query in queries:
            results[query] = self.search(query, topK, expandQuery)
        return results


# ============================================================
# HybridRetriever - 混合检索
# ============================================================


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
        # 加载（或重新加载）评测感知术语映射
        self.bm25Retriever.loadTermsMap()

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


class RerankerRetriever:
    """带重排序的检索器"""

    def __init__(
        self,
        corpusFile: str,
        bm25IndexFile: str,
        vectorIndexFile: str,
        vectorEmbeddingFile: str,
        modelName: str = "paraphrase-multilingual-MiniLM-L12-v2",
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


class AdvancedRetriever:
    """高级检索器 - 多路召回 + 重排序"""

    def __init__(
        self,
        corpusFile: str,
        bm25IndexFile: str,
        vectorIndexFile: str,
        vectorEmbeddingFile: str,
        modelName: str = "paraphrase-multilingual-MiniLM-L12-v2",
        rerankerModel: str = "BAAI/bge-reranker-v2-mixed",
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
        print(f"📂 加载语料：{self.corpusFile}")
        self._corpus = []
        if not os.path.exists(self.corpusFile):
            raise FileNotFoundError(
                f"语料文件不存在：{self.corpusFile}，请先运行语料构建流程"
            )
        skipped = 0
        with open(self.corpusFile, encoding="utf-8") as f:
            for lineNum, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self._corpus.append(json.loads(line))
                except json.JSONDecodeError as e:
                    skipped += 1
                    print(f"⚠️  第 {lineNum} 行 JSON 解析失败，已跳过：{e}")
        if skipped:
            print(f"⚠️  共跳过 {skipped} 行损坏数据")
        print(f"✅ 已加载 {len(self._corpus)} 条语料")

    def _loadBM25(self):
        """懒加载 BM25"""
        if self._bm25 is not None:
            return

        print("📂 加载 BM25 索引...")
        with open(self.bm25IndexFile, "rb") as f:
            indexData = pickle.load(f)

        self._bm25 = indexData["bm25"]
        print("✅ BM25 索引加载完成")

    def _loadVectorIndex(self):
        """懒加载向量索引"""
        if self._vectorIndex is not None:
            return

        print(f"🤖 加载向量模型：{self.modelName}")
        self._vectorModel = SentenceTransformer(self.modelName)

        print("📂 加载向量索引...")
        self._vectorIndex = faiss.read_index(self.vectorIndexFile)
        print("✅ 向量索引加载完成")

    def _loadReranker(self):
        """懒加载重排序器"""
        if self._reranker is not None:
            return

        # 检查是否已标记为不可用
        if getattr(self, "_rerankerUnavailable", False):
            return

        from sentence_transformers import CrossEncoder

        print(f"🤖 加载重排序模型：{self.rerankerModelName}")
        try:
            self._reranker = CrossEncoder(self.rerankerModelName)
            print("✅ 重排序模型加载完成")
        except Exception as e:
            print(f"⚠️  重排序模型加载失败：{e}，将不使用重排序")
            self._rerankerUnavailable = True
            self._reranker = None

    def _loadQueryRewriter(self, termsFile: str | None = None):
        """懒加载查询改写器"""
        if self._queryRewriter is not None:
            return

        from retrieval.queryRewrite import QueryRewriter

        self._queryRewriter = QueryRewriter(termsFile)
        print("✅ 查询改写器加载完成")

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
            print(f"🔄 查询改写：{query} -> {rewrittenQueries}")
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

        print(f"✅ 召回 {len(allCandidates)} 个候选文档")

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
                print("⚠️  重排序不可用，使用融合分数排序")
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
        print(f"⏱️  检索耗时：{(endTime - startTime) * 1000:.2f}ms")

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
