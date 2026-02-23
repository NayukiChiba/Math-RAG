"""
高级检索 - 多路召回 + 重排序

功能：
1. 多路召回（BM25 + 向量 + 查询改写）
2. Cross-Encoder 重排序
3. 查询改写扩展
4. 支持配置化策略

使用方法：
    # 单次查询
    python retrieval/retrievalAdvanced.py --query "泰勒展开" --topk 10

    # 启用重排序
    python retrieval/retrievalAdvanced.py --query "泰勒展开" --topk 10 --use-reranker

    # 启用查询改写
    python retrieval/retrievalAdvanced.py --query "泰勒展开" --topk 10 --rewrite-query
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# 路径调整
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from retrieval.queryRewrite import QueryRewriter


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

        # 延迟加载，避免不必要的导入
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
        with open(self.corpusFile, encoding="utf-8") as f:
            for line in f:
                self._corpus.append(json.loads(line.strip()))
        print(f"✅ 已加载 {len(self._corpus)} 条语料")

    def _loadBM25(self):
        """懒加载 BM25"""
        if self._bm25 is not None:
            return

        import pickle

        print("📂 加载 BM25 索引...")
        with open(self.bm25IndexFile, "rb") as f:
            indexData = pickle.load(f)

        self._bm25 = indexData["bm25"]
        print("✅ BM25 索引加载完成")

    def _loadVectorIndex(self):
        """懒加载向量索引"""
        if self._vectorIndex is not None:
            return

        import faiss
        from sentence_transformers import SentenceTransformer

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
        import faiss

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
            self._loadQueryRewriter()
            rewrittenQueries = self._queryRewriter.rewrite(query)
            print(f"🔄 查询改写：{query} -> {rewrittenQueries}")
        else:
            rewrittenQueries = [query]

        # 2. 多路召回
        allCandidates = {}  # doc_idx -> (doc_idx, max_score, text)

        # BM25 召回
        bm25Results = self._bm25Search(query, recallTopK)
        for idx, score in bm25Results:
            allCandidates[idx] = {"bm25_score": score, "vector_score": 0.0}

        # 向量召回
        vectorResults = self._vectorSearch(query, recallTopK)
        for idx, score in vectorResults:
            if idx in allCandidates:
                allCandidates[idx]["vector_score"] = score
            else:
                allCandidates[idx] = {"bm25_score": 0.0, "vector_score": score}

        # 查询改写召回
        if rewriteQuery and len(rewrittenQueries) > 1:
            for rewrittenQuery in rewrittenQueries[1:4]:  # 用前 3 个改写查询
                rewriteBm25 = self._bm25Search(rewrittenQuery, recallTopK // 3)
                for idx, score in rewriteBm25:
                    if idx in allCandidates:
                        allCandidates[idx]["bm25_score"] = max(
                            allCandidates[idx]["bm25_score"], score
                        )
                    else:
                        allCandidates[idx] = {"bm25_score": score, "vector_score": 0.0}

        print(f"✅ 召回 {len(allCandidates)} 个候选文档")

        # 3. 计算融合分数
        if not allCandidates:
            return []

        # 使用百分位数归一化（更鲁棒，与 Hybrid+ 一致）
        def percentileNorm(scores: list[float]) -> list[float]:
            if not scores:
                return []
            sortedScores = sorted(scores)
            n = len(sortedScores)
            result = []
            for s in scores:
                rank = sum(1 for x in sortedScores if x <= s)
                result.append(rank / n)
            return result

        bm25Scores = [c["bm25_score"] for c in allCandidates.values()]
        vectorScores = [c["vector_score"] for c in allCandidates.values()]

        bm25NormScores = percentileNorm(bm25Scores)
        vectorNormScores = percentileNorm(vectorScores)

        # 构建 doc_id 到归一化分数的映射
        docIds = list(allCandidates.keys())
        bm25ScoreMap = {docIds[i]: bm25NormScores[i] for i in range(len(docIds))}
        vectorScoreMap = {docIds[i]: vectorNormScores[i] for i in range(len(docIds))}

        # 自适应权重调整（与 Hybrid+ 一致）
        import numpy as np

        avgBm25 = np.mean(bm25NormScores) if bm25NormScores else 0
        avgVector = np.mean(vectorNormScores) if vectorNormScores else 0
        total = avgBm25 + avgVector
        if total > 0:
            adaptiveAlpha = avgBm25 / total
            adaptiveBeta = avgVector / total
        else:
            adaptiveAlpha = adaptiveBeta = 0.5

        # 使用自适应权重计算融合分数
        for idx, data in allCandidates.items():
            data["fused_score"] = (
                adaptiveAlpha * bm25ScoreMap[idx] + adaptiveBeta * vectorScoreMap[idx]
            )

        # 4. 重排序
        if useReranker and len(allCandidates) > 0:
            # 先按融合分数排序，取前 50 个进行重排序
            sortedCandidates = sorted(
                allCandidates.items(),
                key=lambda x: x[1]["fused_score"],
                reverse=True,
            )[:50]

            candidates = [(idx, self._getDocText(idx)) for idx, _ in sortedCandidates]
            rerankScores = self._rerankScores(query, candidates)

            if rerankScores is not None:
                # 重排序成功
                for (idx, _), score in zip(sortedCandidates, rerankScores):
                    allCandidates[idx]["reranker_score"] = score

                # 按重排序分数排序
                finalRanking = sorted(
                    allCandidates.items(),
                    key=lambda x: x[1].get("reranker_score", 0),
                    reverse=True,
                )
            else:
                # 重排序不可用，按融合分数排序
                print("⚠️  重排序不可用，使用融合分数排序")
                finalRanking = sorted(
                    allCandidates.items(),
                    key=lambda x: x[1]["fused_score"],
                    reverse=True,
                )
        else:
            # 按融合分数排序
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


def printResults(query: str, results: list[dict[str, Any]]) -> None:
    """打印查询结果"""
    print("\n" + "=" * 80)
    print(f"🔍 查询：{query}")
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
        print(f"     ├─ BM25: {result.get('bm25_score', 0):.4f}")
        print(f"     └─ 向量：{result.get('vector_score', 0):.4f}")
        print(f"  📗 来源：{result['source']}")
        if result.get("page"):
            print(f"  📄 页码：{result['page']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="高级检索 - 多路召回 + 重排序")
    parser.add_argument("--query", type=str, help="单次查询字符串")
    parser.add_argument("--query-file", type=str, help="批量查询文件路径")
    parser.add_argument(
        "--topk", type=int, default=10, help="返回的结果数量（默认 10）"
    )
    parser.add_argument(
        "--recall-topk", type=int, default=100, help="召回候选数量（默认 100）"
    )
    parser.add_argument("--output", type=str, help="输出结果文件路径")
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
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="BAAI/bge-reranker-v2-mixed",
        help="重排序模型名称",
    )
    parser.add_argument("--terms", type=str, help="术语文件路径")
    parser.add_argument("--no-rerank", action="store_true", help="禁用重排序")
    parser.add_argument("--no-rewrite", action="store_true", help="禁用查询改写")
    parser.add_argument(
        "--bm25-weight", type=float, default=0.4, help="BM25 权重（默认 0.4）"
    )
    parser.add_argument(
        "--vector-weight", type=float, default=0.3, help="向量权重（默认 0.3）"
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
    termsFile = args.terms or os.path.join(
        config.PROCESSED_DIR, "terms", "all_terms.json"
    )

    print("=" * 80)
    print("🔍 高级检索 - 多路召回 + 重排序")
    print("=" * 80)
    print(f"📂 语料文件：{corpusFile}")
    print(f"🤖 检索模型：{args.model}")
    print(f"🤖 重排序模型：{args.reranker_model}")
    print(f"🔀 重排序：{'禁用' if args.no_rerank else '启用'}")
    print(f"🔀 查询改写：{'禁用' if args.no_rewrite else '启用'}")
    print()

    # 初始化检索器
    retriever = AdvancedRetriever(
        corpusFile,
        bm25IndexFile,
        vectorIndexFile,
        vectorEmbeddingFile,
        args.model,
        args.reranker_model,
        termsFile,
    )

    # 执行查询
    if args.query:
        results = retriever.search(
            args.query,
            args.topk,
            args.recall_topk,
            not args.no_rerank,
            not args.no_rewrite,
            args.bm25_weight,
            args.vector_weight,
        )
        printResults(args.query, results)

        if args.output:
            saveResults({args.query: results}, args.output)

    elif args.query_file:
        print(f"📂 加载查询：{args.query_file}")
        queries = loadQueriesFromFile(args.query_file)
        print(f"✅ 已加载 {len(queries)} 个查询\n")

        results = retriever.batchSearch(
            queries,
            args.topk,
            recallTopK=args.recall_topk,
            useReranker=not args.no_rerank,
            rewriteQuery=not args.no_rewrite,
            bm25Weight=args.bm25_weight,
            vectorWeight=args.vector_weight,
        )

        for query, queryResults in results.items():
            printResults(query, queryResults)

        if args.output:
            saveResults(results, args.output)
        else:
            defaultOutput = os.path.join(
                config.PROJECT_ROOT, "outputs", "advanced_results.json"
            )
            saveResults(results, defaultOutput)

    else:
        print("⚠️  请提供查询参数")
        parser.print_help()


if __name__ == "__main__":
    main()
