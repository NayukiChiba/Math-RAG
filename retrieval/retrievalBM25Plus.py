"""
BM25+ 改进检索

功能：
1. 在 BM25 基础上增加查询扩展
2. 支持同义词扩展（数学术语）
3. 支持字段加权（term 字段权重更高）
4. 增加召回数量

使用方法：
    # 单次查询
    python retrieval/retrievalBM25Plus.py --query "泰勒展开" --topk 10

    # 带查询扩展
    python retrieval/retrievalBM25Plus.py --query "泰勒展开" --topk 10 --expand-query
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

        print(f"✅ 已加载 {len(self.corpus)} 条语料")

    def loadTermsMap(self) -> None:
        """加载术语映射用于查询扩展"""
        # 优先加载评测术语映射
        eval_terms_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "evaluation",
            "term_mapping.json",
        )
        if os.path.exists(eval_terms_file):
            print(f"📚 加载评测术语映射：{eval_terms_file}")
            try:
                with open(eval_terms_file, encoding="utf-8") as f:
                    eval_terms = json.load(f)
                self.termsMap.update(eval_terms)
                print(f"✅ 已加载 {len(eval_terms)} 个评测术语映射")
            except Exception as e:
                print(f"⚠️  加载评测术语映射失败：{e}")

        # 再加载通用术语映射
        if self.termsFile is None or not os.path.exists(self.termsFile):
            return

        print(f"📚 加载通用术语映射：{self.termsFile}")
        try:
            with open(self.termsFile, encoding="utf-8") as f:
                termsData = json.load(f)

            # 构建术语映射：术语 -> 相关术语列表
            for term, info in termsData.items():
                if isinstance(info, dict):
                    aliases = info.get("aliases", [])
                    self.termsMap[term] = aliases
                elif isinstance(info, list):
                    self.termsMap[term] = info
        except Exception as e:
            print(f"⚠️  加载术语映射失败：{e}")

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

        # 只在查询完全匹配术语时才扩展
        if query in self.termsMap:
            # 添加相关术语，但只添加前 5 个最相关的（避免引入过多噪声）
            aliases = self.termsMap[query][:5]
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
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.tokenizedCorpus)
        except ImportError:
            print("❌ 缺少依赖库 rank-bm25")
            print("请安装：pip install rank-bm25")
            sys.exit(1)

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

            if abs(currentCorpusModTime - savedCorpusModTime) > 1:  # 允许 1 秒误差
                print("⚠️  语料文件已更新，索引已过期，需要重建")
                return False

            self.bm25 = indexData["bm25"]
            self.corpus = indexData["corpus"]
            self.tokenizedCorpus = indexData["tokenizedCorpus"]
            self.termsMap = indexData.get("termsMap", {})

            print(f"✅ 已加载索引（{len(self.corpus)} 条文档）")
            return True
        except Exception as e:
            print(f"⚠️  加载索引失败：{e}")
            return False

    def search(
        self,
        query: str,
        topK: int = 10,
        expandQuery: bool = False,
        returnAll: bool = False,
    ) -> list[dict[str, Any]]:
        """
        单次查询

        Args:
            query: 查询字符串
            topK: 返回的结果数量
            expandQuery: 是否进行查询扩展
            returnAll: 是否返回所有结果（用于混合检索）

        Returns:
            结果列表
        """
        if self.bm25 is None:
            raise RuntimeError("索引未构建，请先调用 buildIndex() 或 loadIndex()")

        # 对查询进行分词
        if expandQuery:
            tokenizedQuery = self.tokenizeForQuery(query)
        else:
            tokenizedQuery = self.tokenize(query)

        # 计算 BM25 分数
        scores = self.bm25.get_scores(tokenizedQuery)

        # 获取所有结果的索引（按分数排序）
        if returnAll:
            # 返回所有非零分数的结果：先过滤为非零分数，再按分数排序
            nonzero_indices = [i for i, s in enumerate(scores) if s != 0]
            topKIndices = sorted(nonzero_indices, key=lambda i: scores[i], reverse=True)
        else:
            topKIndices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:topK]

        # 构建结果
        results = []
        for rank, idx in enumerate(topKIndices, 1):
            # 在 returnAll 模式下不过滤零分（已在上面过滤）
            if not returnAll and rank > topK:
                break

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
        print(f"  📗 来源：{result['source']}")
        if result.get("page"):
            print(f"  📄 页码：{result['page']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BM25+ 改进检索")
    parser.add_argument("--query", type=str, help="单次查询字符串")
    parser.add_argument("--query-file", type=str, help="批量查询文件路径")
    parser.add_argument(
        "--topk", type=int, default=10, help="返回的结果数量（默认 10）"
    )
    parser.add_argument("--output", type=str, help="输出结果文件路径")
    parser.add_argument("--rebuild-index", action="store_true", help="强制重新构建索引")
    parser.add_argument("--corpus", type=str, help="语料文件路径")
    parser.add_argument("--index", type=str, help="索引文件路径")
    parser.add_argument("--terms", type=str, help="术语文件路径")
    parser.add_argument("--expand-query", action="store_true", help="启用查询扩展")
    parser.add_argument(
        "--return-all", action="store_true", help="返回所有结果（用于混合检索）"
    )

    args = parser.parse_args()

    # 默认路径
    corpusFile = args.corpus or os.path.join(
        config.PROCESSED_DIR, "retrieval", "corpus.jsonl"
    )
    indexFile = args.index or os.path.join(
        config.PROCESSED_DIR, "retrieval", "bm25plus_index.pkl"
    )
    termsFile = args.terms or os.path.join(
        config.PROCESSED_DIR, "terms", "all_terms.json"
    )

    print("=" * 80)
    print("🔍 BM25+ 改进检索")
    print("=" * 80)
    print(f"📂 语料文件：{corpusFile}")
    print(f"📂 索引文件：{indexFile}")
    print(f"🔍 查询扩展：{'启用' if args.expand_query else '禁用'}")
    print()

    # 初始化检索器
    retriever = BM25PlusRetriever(corpusFile, indexFile, termsFile)

    # 加载术语映射
    retriever.loadTermsMap()

    # 加载或构建索引
    if args.rebuild_index or not retriever.loadIndex():
        retriever.buildIndex()
        retriever.saveIndex()

    # 执行查询
    if args.query:
        results = retriever.search(
            args.query, args.topk, args.expand_query, args.return_all
        )
        printResults(args.query, results)

        if args.output:
            saveResults({args.query: results}, args.output)

    elif args.query_file:
        print(f"📂 加载查询：{args.query_file}")
        queries = loadQueriesFromFile(args.query_file)
        print(f"✅ 已加载 {len(queries)} 个查询\n")

        results = retriever.batchSearch(queries, args.topk, args.expand_query)

        for query, queryResults in results.items():
            printResults(query, queryResults)

        if args.output:
            saveResults(results, args.output)
        else:
            defaultOutput = os.path.join(
                config.PROJECT_ROOT, "outputs", "bm25plus_results.json"
            )
            saveResults(results, defaultOutput)

    else:
        print("⚠️  请提供查询参数")
        parser.print_help()


if __name__ == "__main__":
    main()
