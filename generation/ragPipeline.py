"""
端到端 RAG 问答流程模块

功能：
1. 接收用户查询，检索相关术语，拼接上下文，Qwen 生成回答
2. 支持单条查询和批量查询
3. 支持检索策略切换（BM25 / 向量 / 混合）
4. 输出结构化结果（query、retrieved_terms、answer、sources、latency）
5. 异常处理（检索为空时给出提示而非崩溃）

使用方法：
    from generation.ragPipeline import RagPipeline

    # 初始化
    pipeline = RagPipeline()

    # 单条查询
    result = pipeline.query("什么是一致收敛？")
    print(result["answer"])

    # 批量查询
    results = pipeline.batchQuery(["问题1", "问题2"])
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal

# 路径调整
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from generation.promptTemplates import buildMessages
from generation.qwenInference import QwenInference
from retrieval.retrievers import BM25Retriever, HybridRetriever, VectorRetriever

# 检索策略类型
RetrievalStrategy = Literal["bm25", "vector", "hybrid"]


class RagPipeline:
    """端到端 RAG 问答流程"""

    def __init__(
        self,
        strategy: RetrievalStrategy = "hybrid",
        topK: int = 5,
        corpusFile: str | None = None,
        bm25IndexFile: str | None = None,
        vectorIndexFile: str | None = None,
        vectorEmbeddingFile: str | None = None,
        modelName: str = "paraphrase-multilingual-MiniLM-L12-v2",
        hybridAlpha: float = 0.5,
        hybridBeta: float = 0.5,
    ):
        """
        初始化 RAG 流程

        Args:
            strategy: 检索策略（bm25 / vector / hybrid）
            topK: 检索返回的结果数量
            corpusFile: 语料文件路径
            bm25IndexFile: BM25 索引文件路径
            vectorIndexFile: 向量索引文件路径
            vectorEmbeddingFile: 向量嵌入文件路径
            modelName: Sentence Transformer 模型名称
            hybridAlpha: 混合检索 BM25 权重
            hybridBeta: 混合检索向量权重
        """
        self.strategy = strategy
        self.topK = topK
        self.modelName = modelName
        self.hybridAlpha = hybridAlpha
        self.hybridBeta = hybridBeta

        # 默认路径
        retrievalDir = os.path.join(config.PROCESSED_DIR, "retrieval")
        self.corpusFile = corpusFile or os.path.join(retrievalDir, "corpus.jsonl")
        self.bm25IndexFile = bm25IndexFile or os.path.join(
            retrievalDir, "bm25_index.pkl"
        )
        self.vectorIndexFile = vectorIndexFile or os.path.join(
            retrievalDir, "vector_index.faiss"
        )
        self.vectorEmbeddingFile = vectorEmbeddingFile or os.path.join(
            retrievalDir, "vector_embeddings.npz"
        )

        # 延迟加载
        self._retriever = None
        self._qwen = None
        self._corpus = None

    def _loadCorpus(self) -> dict[str, dict[str, Any]]:
        """加载语料文件，构建 doc_id 到文档的映射"""
        if self._corpus is not None:
            return self._corpus

        self._corpus = {}
        if os.path.exists(self.corpusFile):
            with open(self.corpusFile, encoding="utf-8") as f:
                for lineNum, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                        docId = doc.get("doc_id")
                        if docId:
                            self._corpus[docId] = doc
                        else:
                            print(f"⚠️ 语料第 {lineNum} 行缺少 doc_id，已跳过")
                    except json.JSONDecodeError as e:
                        print(f"⚠️ 语料第 {lineNum} 行 JSON 解析失败: {e}")
        return self._corpus

    def _initRetriever(self) -> None:
        """初始化检索器"""
        if self._retriever is not None:
            return

        print(f"🔧 初始化检索器（策略: {self.strategy}）...")

        if self.strategy == "bm25":
            self._retriever = BM25Retriever(self.corpusFile, self.bm25IndexFile)
            if not self._retriever.loadIndex():
                print("BM25 索引不存在，正在构建...")
                self._retriever.buildIndex()
                self._retriever.saveIndex()

        elif self.strategy == "vector":
            self._retriever = VectorRetriever(
                self.corpusFile,
                self.modelName,
                self.vectorIndexFile,
                self.vectorEmbeddingFile,
            )
            if not self._retriever.loadIndex():
                print("向量索引不存在，正在构建...")
                self._retriever.buildIndex()
                self._retriever.saveIndex()

        elif self.strategy == "hybrid":
            self._retriever = HybridRetriever(
                self.corpusFile,
                self.bm25IndexFile,
                self.vectorIndexFile,
                self.vectorEmbeddingFile,
                self.modelName,
            )

        else:
            raise ValueError(
                f"不支持的检索策略: {self.strategy}，"
                f"请在 ['bm25', 'vector', 'hybrid'] 中选择"
            )

        print("检索器初始化完成")

    def _initQwen(self) -> None:
        """初始化 Qwen 推理实例"""
        if self._qwen is not None:
            return

        print("初始化 Qwen 推理...")
        self._qwen = QwenInference()
        print("Qwen 初始化完成")

    def _retrieve(self, queryText: str) -> list[dict[str, Any]]:
        """
        执行检索

        Args:
            queryText: 查询文本

        Returns:
            检索结果列表
        """
        self._initRetriever()

        if self.strategy == "hybrid":
            results = self._retriever.search(
                queryText,
                topK=self.topK,
                strategy="weighted",
                alpha=self.hybridAlpha,
                beta=self.hybridBeta,
            )
        else:
            results = self._retriever.search(queryText, topK=self.topK)

        return results

    def _enrichResults(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        补充检索结果的完整文本信息

        Args:
            results: 检索结果列表

        Returns:
            补充后的结果列表
        """
        corpus = self._loadCorpus()
        enriched = []

        for r in results:
            docId = r.get("doc_id")
            if docId and docId in corpus:
                doc = corpus[docId]
                enriched.append(
                    {
                        "rank": r.get("rank"),
                        "doc_id": docId,
                        "term": doc.get("term", r.get("term", "")),
                        "subject": doc.get("subject", r.get("subject", "")),
                        "text": doc.get("text", ""),
                        "source": doc.get("source", r.get("source", "")),
                        "page": doc.get("page", r.get("page")),
                        "score": r.get("score", 0.0),
                    }
                )
            else:
                enriched.append(r)

        return enriched

    def query(
        self,
        queryText: str,
        temperature: float | None = None,
        topP: float | None = None,
        maxNewTokens: int | None = None,
    ) -> dict[str, Any]:
        """
        单条查询

        Args:
            queryText: 查询文本
            temperature: 采样温度
            topP: top-p 采样参数
            maxNewTokens: 最大生成 token 数

        Returns:
            结果字典，包含：
                - query: 原始查询
                - retrieved_terms: 检索到的术语列表
                - answer: 生成的回答
                - sources: 来源列表
                - latency: 耗时信息
        """
        result = {
            "query": queryText,
            "retrieved_terms": [],
            "answer": "",
            "sources": [],
            "latency": {
                "retrieval_ms": 0,
                "generation_ms": 0,
                "total_ms": 0,
            },
        }

        totalStart = time.time()

        # 检索阶段
        retrievalStart = time.time()
        try:
            rawResults = self._retrieve(queryText)
            retrievalResults = self._enrichResults(rawResults)
        except Exception as e:
            print(f"检索失败: {e}")
            retrievalResults = []

        retrievalEnd = time.time()
        result["latency"]["retrieval_ms"] = int((retrievalEnd - retrievalStart) * 1000)

        # 记录检索到的术语
        result["retrieved_terms"] = [
            {
                "term": r.get("term", ""),
                "subject": r.get("subject", ""),
                "source": r.get("source", ""),
                "page": r.get("page"),
                "score": r.get("score", 0.0),
            }
            for r in retrievalResults
        ]

        # 记录来源
        result["sources"] = [
            {
                "source": r.get("source", ""),
                "page": r.get("page"),
            }
            for r in retrievalResults
            if r.get("source")
        ]

        # 生成阶段
        generationStart = time.time()
        try:
            self._initQwen()

            # 构建 messages
            messages = buildMessages(
                query=queryText,
                retrievalResults=retrievalResults if retrievalResults else None,
            )

            # 生成回答
            answer = self._qwen.generateFromMessages(
                messages=messages,
                temperature=temperature,
                topP=topP,
                maxNewTokens=maxNewTokens,
            )
            result["answer"] = answer

        except Exception as e:
            print(f"生成失败: {e}")
            result["answer"] = f"生成失败: {e}"

        generationEnd = time.time()
        result["latency"]["generation_ms"] = int(
            (generationEnd - generationStart) * 1000
        )

        totalEnd = time.time()
        result["latency"]["total_ms"] = int((totalEnd - totalStart) * 1000)

        return result

    def batchQuery(
        self,
        queries: list[str],
        temperature: float | None = None,
        topP: float | None = None,
        maxNewTokens: int | None = None,
        showProgress: bool = True,
    ) -> list[dict[str, Any]]:
        """
        批量查询

        Args:
            queries: 查询文本列表
            temperature: 采样温度
            topP: top-p 采样参数
            maxNewTokens: 最大生成 token 数
            showProgress: 是否显示进度

        Returns:
            结果列表
        """
        results = []
        total = len(queries)

        for i, q in enumerate(queries):
            if showProgress:
                print(f"处理查询 {i + 1}/{total}: {q[:30]}...")

            result = self.query(
                q,
                temperature=temperature,
                topP=topP,
                maxNewTokens=maxNewTokens,
            )
            results.append(result)

        return results

    def setStrategy(self, strategy: RetrievalStrategy) -> None:
        """
        切换检索策略

        Args:
            strategy: 新的检索策略
        """
        if strategy != self.strategy:
            self.strategy = strategy
            self._retriever = None  # 重置检索器
            print(f"检索策略已切换为: {strategy}")


def saveResults(results: list[dict[str, Any]], outputFile: str) -> None:
    """
    保存结果到 JSONL 文件

    Args:
        results: 结果列表
        outputFile: 输出文件路径
    """
    dirname = os.path.dirname(outputFile)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(outputFile, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"结果已保存: {outputFile}")


def loadQueries(filepath: str) -> list[str]:
    """
    从文件加载查询

    Args:
        filepath: 查询文件路径（每行一个查询，或 JSONL 格式）

    Returns:
        查询列表
    """
    queries = []
    skippedCount = 0

    with open(filepath, encoding="utf-8") as f:
        for lineNum, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # 尝试解析为 JSON
            try:
                data = json.loads(line)
                if isinstance(data, dict) and "query" in data:
                    queries.append(data["query"])
                elif isinstance(data, str):
                    queries.append(data)
                else:
                    # JSON 对象但缺少 query 字段
                    skippedCount += 1
                    print(f"⚠️ 第 {lineNum} 行: JSON 对象缺少 'query' 字段，已跳过")
            except json.JSONDecodeError:
                # 纯文本格式
                queries.append(line)

    if skippedCount > 0:
        print(f"⚠️ 共跳过 {skippedCount} 行格式不正确的记录")

    return queries


# ---- 命令行测试 ----


def _testPipeline() -> None:
    """模块级测试"""
    print("=" * 60)
    print("RAG Pipeline 测试")
    print("=" * 60)

    # 初始化
    pipeline = RagPipeline(strategy="hybrid", topK=3)

    # 测试查询
    testQuery = "什么是一致收敛？"
    print(f"\n测试查询: {testQuery}")
    print("-" * 40)

    result = pipeline.query(testQuery)

    print(f"\n检索到 {len(result['retrieved_terms'])} 个术语:")
    for t in result["retrieved_terms"]:
        print(f"  - {t['term']} ({t['subject']}) [score: {t['score']:.4f}]")

    print(f"\n回答:\n{result['answer']}")

    print("\n耗时:")
    print(f"  检索: {result['latency']['retrieval_ms']} ms")
    print(f"  生成: {result['latency']['generation_ms']} ms")
    print(f"  总计: {result['latency']['total_ms']} ms")


if __name__ == "__main__":
    _testPipeline()
