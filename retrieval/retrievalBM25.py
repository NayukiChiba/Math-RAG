"""
BM25 基线检索

功能：
1. 从语料文件构建 BM25 索引
2. 支持单次查询和批量查询
3. 输出 TopK 结果（doc_id、term、score、rank）
4. 支持索引保存和加载

使用方法：
    # 单次查询
    python retrieval/retrievalBM25.py --query "泰勒展开" --topk 10

    # 批量查询
    python retrieval/retrievalBM25.py --query-file queries.txt --output results.json

    # 重新构建索引
    python retrieval/retrievalBM25.py --rebuild-index
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

# 路径调整
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("❌ 缺少依赖库 rank-bm25")
    print("请安装: pip install rank-bm25")
    sys.exit(1)


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
        未来可以替换为更复杂的分词器。

        Args:
            text: 待分词文本

        Returns:
            分词结果列表
        """
        # 简单的字符级分词，去除空格和换行
        # 保留数学符号和标点
        tokens = []
        for char in text:
            if char.strip():  # 跳过空白字符
                tokens.append(char)

        # 也可以按照空格分词（词级别）
        # tokens = text.split()

        # 或者混合策略：提取关键词
        # 这里使用简单策略：按字符分词
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

            if abs(currentCorpusModTime - savedCorpusModTime) > 1:  # 允许1秒误差
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
    os.makedirs(os.path.dirname(outputFile), exist_ok=True)

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
    parser = argparse.ArgumentParser(description="BM25 基线检索")
    parser.add_argument("--query", type=str, help="单次查询字符串")
    parser.add_argument("--query-file", type=str, help="批量查询文件路径")
    parser.add_argument(
        "--topk", type=int, default=10, help="返回的结果数量（默认 10）"
    )
    parser.add_argument("--output", type=str, help="输出结果文件路径（JSON 格式）")
    parser.add_argument("--rebuild-index", action="store_true", help="强制重新构建索引")
    parser.add_argument("--corpus", type=str, help="语料文件路径")
    parser.add_argument("--index", type=str, help="索引文件路径")

    args = parser.parse_args()

    # 默认路径
    corpusFile = args.corpus or os.path.join(
        config.PROCESSED_DIR, "retrieval", "corpus.jsonl"
    )
    indexFile = args.index or os.path.join(
        config.PROCESSED_DIR, "retrieval", "bm25_index.pkl"
    )

    print("=" * 80)
    print("🔍 BM25 基线检索")
    print("=" * 80)
    print(f"📂 语料文件: {corpusFile}")
    print(f"📂 索引文件: {indexFile}")
    print()

    # 初始化检索器
    retriever = BM25Retriever(corpusFile, indexFile)

    # 加载或构建索引
    if args.rebuild_index or not retriever.loadIndex():
        retriever.buildIndex()
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
                config.PROJECT_ROOT, "outputs", "bm25_results.json"
            )
            os.makedirs(os.path.dirname(defaultOutput), exist_ok=True)
            saveResults(results, defaultOutput)

    else:
        print("⚠️  请提供查询参数：")
        print("  --query 'your query'  # 单次查询")
        print("  --query-file queries.txt  # 批量查询")
        parser.print_help()


if __name__ == "__main__":
    main()
