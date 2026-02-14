"""
向量检索基线

功能：
1. 从语料文件构建向量索引（FAISS）
2. 使用 Sentence Transformers 进行文本嵌入
3. 支持单次查询和批量查询
4. 输出 TopK 结果（doc_id、term、score、rank）
5. 支持索引和嵌入保存加载

使用方法：
    # 单次查询
    python retrieval/retrievalVector.py --query "泰勒展开" --topk 10

    # 批量查询
    python retrieval/retrievalVector.py --query-file queries.txt --output results.json

    # 重新构建索引
    python retrieval/retrievalVector.py --rebuild-index

    # 指定模型
    python retrieval/retrievalVector.py --model paraphrase-multilingual-MiniLM-L12-v2
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# 路径调整
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# 全局变量：GPU 可用性
USE_GPU = False
NUM_GPUS = 0

try:
    import faiss

    # 尝试检测 GPU（faiss-gpu 才有此方法）
    if hasattr(faiss, "get_num_gpus"):
        try:
            NUM_GPUS = faiss.get_num_gpus()
            if NUM_GPUS > 0:
                USE_GPU = True
                print(f"🎮 检测到 {NUM_GPUS} 个 GPU，将使用 GPU 加速")
            else:
                print("💻 使用 CPU 模式（未检测到 GPU）")
        except Exception:
            print("💻 使用 CPU 模式（GPU 初始化失败）")
    else:
        print("💻 使用 CPU 模式（faiss-cpu 版本）")
except ImportError:
    print("❌ 缺少依赖库 faiss")
    print("请安装:")
    print("  CPU 版本: pip install faiss-cpu")
    print("  GPU 版本: conda install -c pytorch -c nvidia faiss-gpu")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ 缺少依赖库 sentence-transformers")
    print("请安装: pip install sentence-transformers")
    sys.exit(1)


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
            embeddingFile: 嵌入向量文件路径（.npy），如果为 None 则不保存
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

        # 构建 FAISS 索引（使用内积，因为向量已标准化，等价于余弦相似度）
        dimension = self.embeddings.shape[1]
        cpuIndex = faiss.IndexFlatIP(dimension)  # Inner Product (余弦相似度)

        # 如果有 GPU，将索引迁移到 GPU
        if USE_GPU:
            res = faiss.StandardGpuResources()  # 使用默认 GPU 资源
            self.index = faiss.index_cpu_to_gpu(res, 0, cpuIndex)  # 迁移到 GPU 0
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
            if dirname:  # 只有当目录名非空时才创建目录
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
            if dirname:  # 只有当目录名非空时才创建目录
                os.makedirs(dirname, exist_ok=True)

            # 保存嵌入和语料
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

            if abs(currentCorpusModTime - savedCorpusModTime) > 1:  # 允许1秒误差
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
                # 如果嵌入文件不存在，重新加载语料
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
        faiss.normalize_L2(queryEmbedding)  # 标准化

        # 执行搜索
        scores, indices = self.index.search(queryEmbedding, topK)

        # 构建结果
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            if idx == -1:  # FAISS 返回 -1 表示无效结果
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
    # 确保输出目录存在
    dirname = os.path.dirname(outputFile)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(outputFile, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 结果已保存: {outputFile}")


def printResults(query: str, results: list[dict[str, Any]]) -> None:
    """
    打印查询结果

    Args:
        query: 查询字符串
        results: 结果列表
    """
    print("\n" + "=" * 80)
    print(f"🔍 查询: {query}")
    print("=" * 80)

    if not results:
        print("❌ 未找到相关结果")
        return

    for result in results:
        print(f"\n🏆 Rank {result['rank']}")
        print(f"  📄 Doc ID: {result['doc_id']}")
        print(f"  📚 术语: {result['term']}")
        print(f"  📖 学科: {result['subject']}")
        print(f"  📊 分数: {result['score']:.4f}")
        print(f"  📗 来源: {result['source']}")
        if result.get("page"):
            print(f"  📄 页码: {result['page']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="向量检索基线")
    parser.add_argument("--query", type=str, help="单次查询字符串")
    parser.add_argument("--query-file", type=str, help="批量查询文件路径")
    parser.add_argument(
        "--topk", type=int, default=10, help="返回的结果数量（默认 10）"
    )
    parser.add_argument("--output", type=str, help="输出结果文件路径（JSON 格式）")
    parser.add_argument("--rebuild-index", action="store_true", help="强制重新构建索引")
    parser.add_argument("--corpus", type=str, help="语料文件路径")
    parser.add_argument("--index", type=str, help="FAISS 索引文件路径")
    parser.add_argument("--embedding", type=str, help="嵌入向量文件路径")
    parser.add_argument(
        "--model",
        type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence Transformer 模型名称",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="嵌入计算的批次大小（默认 32）"
    )

    args = parser.parse_args()

    # 默认路径
    corpusFile = args.corpus or os.path.join(
        config.PROCESSED_DIR, "retrieval", "corpus.jsonl"
    )
    indexFile = args.index or os.path.join(
        config.PROCESSED_DIR, "retrieval", "vector_index.faiss"
    )
    embeddingFile = args.embedding or os.path.join(
        config.PROCESSED_DIR, "retrieval", "vector_embeddings.npz"
    )

    print("=" * 80)
    print("🔍 向量检索基线")
    print("=" * 80)
    print(f"📂 语料文件: {corpusFile}")
    print(f"📂 索引文件: {indexFile}")
    print(f"📂 嵌入文件: {embeddingFile}")
    print(f"🤖 模型: {args.model}")
    print()

    # 初始化检索器
    retriever = VectorRetriever(corpusFile, args.model, indexFile, embeddingFile)

    # 加载或构建索引
    if args.rebuild_index or not retriever.loadIndex():
        retriever.buildIndex(batchSize=args.batch_size)
        retriever.saveIndex()

    # 执行查询
    if args.query:
        # 单次查询
        results = retriever.search(args.query, args.topk)
        printResults(args.query, results)

        if args.output:
            saveResults({args.query: results}, args.output)

    elif args.query_file:
        # 批量查询
        print(f"📂 加载查询: {args.query_file}")
        queries = loadQueriesFromFile(args.query_file)
        print(f"✅ 已加载 {len(queries)} 个查询\n")

        results = retriever.batchSearch(queries, args.topk)

        # 打印每个查询的结果
        for query, queryResults in results.items():
            printResults(query, queryResults)

        # 保存结果
        if args.output:
            saveResults(results, args.output)
        else:
            # 默认输出文件
            defaultOutput = os.path.join(
                config.PROJECT_ROOT, "outputs", "vector_results.json"
            )
            saveResults(results, defaultOutput)

    else:
        print("⚠️  请提供查询参数：")
        print("  --query 'your query'  # 单次查询")
        print("  --query-file queries.txt  # 批量查询")
        parser.print_help()


if __name__ == "__main__":
    main()
