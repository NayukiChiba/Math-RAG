"""
带重排序的检索

功能：
1. 使用 Cross-Encoder 进行重排序
2. 先召回大量候选，再用更精细的模型重排
3. 支持多种重排序策略
4. 可选的重排序模型

使用方法：
    # 单次查询（使用默认重排序）
    python retrieval/retrievalWithReranker.py --query "泰勒展开" --topk 10

    # 指定重排序模型
    python retrieval/retrievalWithReranker.py --query "泰勒展开" --topk 10 --reranker-model bge-reranker-base

    # 调整召回数量
    python retrieval/retrievalWithReranker.py --query "泰勒展开" --topk 10 --recall-topk 50
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# 路径调整
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


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

        import pickle

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

        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("❌ 缺少依赖库")
            sys.exit(1)

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
            print("请安装：pip install sentence-transformers")
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
            # 简单分词
            tokens = query.split()
            scores = self.bm25.get_scores(tokens)

            # 获取 topK 候选
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
            import faiss

            # 生成查询向量
            queryEmbedding = self.vectorModel.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(queryEmbedding)

            # 检索
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
                    # 更新向量分数
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
            # 使用 Cross-Encoder 重排序
            print(f"🔄 使用重排序模型对 {len(candidates)} 个候选进行重排序...")

            # 构建句子对
            pairs = [[query, c["text"]] for c in candidates]

            # 预测分数
            rerankScores = self.reranker.predict(pairs)

            # 添加重排序分数
            for i, candidate in enumerate(candidates):
                candidate["reranker_score"] = float(rerankScores[i])

            # 按重排序分数排序
            sortedCandidates = sorted(
                candidates, key=lambda x: x["reranker_score"], reverse=True
            )

        else:
            # 不使用重排序，按综合分数排序
            print("📊 按综合分数排序...")

            for candidate in candidates:
                bm25Score = candidate.get("bm25_score", 0)
                vectorScore = candidate.get("vector_score", 0)
                # 简单平均
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
        # 检索候选
        print(f"📥 召回候选文档（top{recallTopK}）...")
        candidates = self._retrieveCandidates(query, recallTopK)

        print(f"✅ 召回 {len(candidates)} 个候选文档")

        # 重排序
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
        print(f"  📊 重排序分数：{result['score']:.4f}")
        print(f"     ├─ BM25: {result.get('bm25_score', 0):.4f}")
        print(f"     └─ 向量：{result.get('vector_score', 0):.4f}")
        print(f"  📗 来源：{result['source']}")
        if result.get("page"):
            print(f"  📄 页码：{result['page']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="带重排序的检索")
    parser.add_argument("--query", type=str, help="单次查询字符串")
    parser.add_argument("--query-file", type=str, help="批量查询文件路径")
    parser.add_argument(
        "--topk", type=int, default=10, help="返回的结果数量（默认 10）"
    )
    parser.add_argument(
        "--recall-topk", type=int, default=50, help="召回候选数量（默认 50）"
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
        default="bge-reranker-base",
        help="重排序模型名称",
    )
    parser.add_argument("--no-rerank", action="store_true", help="禁用重排序")

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

    print("=" * 80)
    print("🔍 带重排序的检索")
    print("=" * 80)
    print(f"📂 语料文件：{corpusFile}")
    print(f"📂 BM25 索引：{bm25IndexFile}")
    print(f"📂 向量索引：{vectorIndexFile}")
    print(f"🤖 检索模型：{args.model}")
    print(f"🤖 重排序模型：{args.reranker_model}")
    print(f"📈 召回数量：{args.recall_topk}")
    print(f"🔀 重排序：{'禁用' if args.no_rerank else '启用'}")
    print()

    # 初始化检索器
    retriever = RerankerRetriever(
        corpusFile,
        bm25IndexFile,
        vectorIndexFile,
        vectorEmbeddingFile,
        args.model,
        args.reranker_model,
    )

    # 执行查询
    if args.query:
        results = retriever.search(
            args.query, args.topk, args.recall_topk, not args.no_rerank
        )
        printResults(args.query, results)

        if args.output:
            saveResults({args.query: results}, args.output)

    elif args.query_file:
        print(f"📂 加载查询：{args.query_file}")
        queries = loadQueriesFromFile(args.query_file)
        print(f"✅ 已加载 {len(queries)} 个查询\n")

        results = retriever.batchSearch(
            queries, args.topk, args.recall_topk, not args.no_rerank
        )

        for query, queryResults in results.items():
            printResults(query, queryResults)

        if args.output:
            saveResults(results, args.output)
        else:
            defaultOutput = os.path.join(
                config.PROJECT_ROOT, "outputs", "reranker_results.json"
            )
            saveResults(results, defaultOutput)

    else:
        print("⚠️  请提供查询参数")
        parser.print_help()


if __name__ == "__main__":
    main()
