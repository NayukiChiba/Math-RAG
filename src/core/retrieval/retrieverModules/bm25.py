from __future__ import annotations

import os
import pickle
from typing import Any

from rank_bm25 import BM25Okapi

from core.retrieval.retrieverModules.shared import _DEFAULT_BM25_NGRAM_MAX, _LOADER


class BM25Retriever:
    """BM25 检索器"""

    def __init__(
        self, corpusFile: str, indexFile: str | None = None, ngramMax: int | None = None
    ):
        """
        初始化 BM25 检索器

        Args:
            corpusFile: 语料文件路径（JSONL 格式）
            indexFile: 索引文件路径（pickle 格式），如果为 None 则不保存
            ngramMax: 字符 n-gram 最大长度（None=读取配置，0/1=禁用，2=+2-gram，3=+3-gram）
        """
        self.corpusFile = corpusFile
        self.indexFile = indexFile
        self.ngramMax = ngramMax if ngramMax is not None else _DEFAULT_BM25_NGRAM_MAX
        self.corpus = []
        self.bm25 = None
        self.tokenizedCorpus = []

    def loadCorpus(self) -> None:
        """加载语料文件"""
        print(f" 加载语料: {self.corpusFile}")

        if not os.path.exists(self.corpusFile):
            raise FileNotFoundError(f"语料文件不存在: {self.corpusFile}")

        self.corpus = _LOADER.jsonl(self.corpusFile)

        print(f" 已加载 {len(self.corpus)} 条语料")

    def tokenize(self, text: str) -> list[str]:
        """
        分词函数（字符级 + 2-gram + 3-gram）

        对于数学术语，使用字符级分词 + n-gram 可以捕获更多部分匹配：
        - 字符级：保留每个字符
        - 2-gram：每两个连续字符
        - 3-gram：每三个连续字符

        Args:
            text: 待分词文本

        Returns:
            分词结果列表
        """
        tokens = []
        chars = [c for c in text if c.strip()]
        # 字符级 token
        tokens.extend(chars)
        # 2-gram
        if self.ngramMax >= 2:
            for i in range(len(chars) - 1):
                tokens.append(chars[i] + chars[i + 1])
        # 3-gram
        if self.ngramMax >= 3:
            for i in range(len(chars) - 2):
                tokens.append(chars[i] + chars[i + 1] + chars[i + 2])
        return tokens

    def buildIndex(self) -> None:
        """构建 BM25 索引"""
        print(" 构建 BM25 索引...")

        if not self.corpus:
            self.loadCorpus()

        # 对每个文档的 text 字段进行分词
        self.tokenizedCorpus = [self.tokenize(doc["text"]) for doc in self.corpus]

        # 构建 BM25 索引
        self.bm25 = BM25Okapi(self.tokenizedCorpus)

        print(" 索引构建完成")

    def saveIndex(self) -> None:
        """保存索引到文件"""
        if self.indexFile is None:
            return

        print(f" 保存索引: {self.indexFile}")

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

        print(" 索引已保存")

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
            print(f"  语料文件不存在: {self.corpusFile}")
            return False

        print(f" 加载索引: {self.indexFile}")

        try:
            with open(self.indexFile, "rb") as f:
                indexData = pickle.load(f)

            # 校验语料文件是否已变更
            currentCorpusModTime = os.path.getmtime(self.corpusFile)
            savedCorpusModTime = indexData.get("corpusModTime")

            if savedCorpusModTime is None:
                print("  索引中缺少语料时间戳，建议重建索引")
                return False

            if abs(currentCorpusModTime - savedCorpusModTime) > 1:
                print("  语料文件已更新，索引已过期，需要重建")
                return False

            self.bm25 = indexData["bm25"]
            self.corpus = indexData["corpus"]
            self.tokenizedCorpus = indexData["tokenizedCorpus"]

            print(f" 已加载索引（{len(self.corpus)} 条文档）")
            return True
        except Exception as e:
            print(f"  加载索引失败: {e}")
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
