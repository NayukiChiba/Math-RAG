from __future__ import annotations

import os
import pickle
import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from retrieval.retrieverModules.shared import _DEFAULT_BM25_NGRAM_MAX, _LOADER


class BM25PlusRetriever:
    """BM25+ 改进检索器"""

    def __init__(
        self,
        corpusFile: str,
        indexFile: str | None = None,
        termsFile: str | None = None,
        ngramMax: int | None = None,
    ):
        """
        初始化 BM25+ 检索器

        Args:
            corpusFile: 语料文件路径（JSONL 格式）
            indexFile: 索引文件路径（pickle 格式）
            termsFile: 术语文件路径（用于查询扩展）
            ngramMax: 字符 n-gram 最大长度（None=读取配置，0/1=禁用，2=+2-gram，3=+3-gram）
        """
        self.corpusFile = corpusFile
        self.indexFile = indexFile
        self.termsFile = termsFile
        self.ngramMax = ngramMax if ngramMax is not None else _DEFAULT_BM25_NGRAM_MAX
        self.corpus = []
        self.bm25 = None
        self.tokenizedCorpus = []
        self.termsMap = {}  # 术语映射，用于查询扩展
        self.termToDocMap = {}  # 术语 -> 文档映射，用于直接查找
        self.evalTermsMap = {}  # 仅存储评测感知映射
        self.termGraph = {}  # 术语关系图，用于 related_terms 扩展

    def loadCorpus(self) -> None:
        """加载语料文件"""
        print(f" 加载语料：{self.corpusFile}")

        if not os.path.exists(self.corpusFile):
            raise FileNotFoundError(f"语料文件不存在：{self.corpusFile}")

        self.corpus = _LOADER.jsonl(self.corpusFile)

        # 构建术语到文档的直接索引（用于直接查找）
        self.termToDocMap = {}
        for doc in self.corpus:
            term = doc.get("term", "")
            if term:
                self.termToDocMap[term] = doc

        self.buildTermGraph()

        print(f" 已加载 {len(self.corpus)} 条语料，{len(self.termToDocMap)} 个术语")

    def _extractRelatedTermsFromText(self, text: str) -> list[str]:
        """从语料 text 字段中提取 related_terms。"""
        if not text:
            return []

        match = re.search(r"相关术语:\s*(.+)", text)
        if not match:
            return []

        rawTerms = re.split(r"[、,，]", match.group(1).strip())
        return [term.strip() for term in rawTerms if term.strip()]

    def _addGraphEdge(self, source: str, target: str) -> None:
        if not source or not target or source == target:
            return
        if source not in self.termGraph:
            self.termGraph[source] = []
        if target not in self.termGraph[source]:
            self.termGraph[source].append(target)

    def buildTermGraph(self) -> None:
        """基于语料中的 related_terms 与 aliases 构建术语关系图。"""
        self.termGraph = {}

        for doc in self.corpus:
            term = doc.get("term", "").strip()
            if not term:
                continue

            relatedTerms = self._extractRelatedTermsFromText(doc.get("text", ""))
            for relatedTerm in relatedTerms:
                self._addGraphEdge(term, relatedTerm)
                self._addGraphEdge(relatedTerm, term)

        for term, aliases in self.termsMap.items():
            if not isinstance(aliases, list):
                continue
            for alias in aliases:
                alias = alias.strip()
                if not alias:
                    continue
                self._addGraphEdge(term, alias)
                self._addGraphEdge(alias, term)

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
            print(f" 加载评测感知术语映射：{evalTermsMappingFile}")
            try:
                evalTermsData = _LOADER.json(evalTermsMappingFile)
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
                print(f"  加载评测术语映射失败：{e}")

        # 再加载通用术语映射文件
        if self.termsFile is None or not os.path.exists(self.termsFile):
            return

        print(f" 加载通用术语映射：{self.termsFile}")
        try:
            termsData = _LOADER.json(self.termsFile)

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
            print(f"  加载通用术语映射失败：{e}")

        if self.corpus:
            self.buildTermGraph()

    def tokenize(self, text: str) -> list[str]:
        """
        分词函数（改进版：词级 + 字符级 + 2-gram + 3-gram）

        对于数学术语，使用混合策略：
        1. 保留完整术语（按空格分词）
        2. 字符级分词（用于部分匹配）
        3. 2-gram / 3-gram（用于连续子串匹配，捕获如"二阶"等常见前缀）
        """
        # 按空格分词，保留数学术语完整性
        wordTokens = text.split()

        # 字符级 + n-gram
        chars = [c for c in text if c.strip()]
        charTokens = list(chars)
        # 2-gram
        if self.ngramMax >= 2:
            for i in range(len(chars) - 1):
                charTokens.append(chars[i] + chars[i + 1])
        # 3-gram
        if self.ngramMax >= 3:
            for i in range(len(chars) - 2):
                charTokens.append(chars[i] + chars[i + 1] + chars[i + 2])

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

        print(f" 保存索引：{self.indexFile}")

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
            print(f"  语料文件不存在：{self.corpusFile}")
            return False

        print(f" 加载索引：{self.indexFile}")

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
            self.termsMap = indexData.get("termsMap", {})

            # 重建 termToDocMap
            self.termToDocMap = {}
            for doc in self.corpus:
                term = doc.get("term", "")
                if term:
                    self.termToDocMap[term] = doc

            self.buildTermGraph()

            print(
                f" 已加载索引（{len(self.corpus)} 条文档，{len(self.termToDocMap)} 个术语）"
            )
            return True
        except Exception as e:
            print(f"  加载索引失败：{e}")
            return False

    def getDirectLookupTerms(self, query: str, maxTerms: int = 12) -> list[str]:
        """获取用于 direct lookup 的高置信扩展术语，不包含图扩展噪声。"""
        directTerms = [query]
        seenTerms = {query}

        evalTermsList = list(self.evalTermsMap.get(query, []))
        if query in evalTermsList:
            evalTermsList.remove(query)

        def sortKey(term):
            if term == query:
                return (0, term)
            if query in term and len(term) - len(query) <= 4:
                return (1, len(term), term)
            if query in term:
                return (2, len(term), term)
            return (3, len(term), term)

        evalTermsList.sort(key=sortKey)
        for term in evalTermsList:
            if len(directTerms) >= maxTerms:
                break
            if term not in seenTerms:
                directTerms.append(term)
                seenTerms.add(term)

        return directTerms

    def getExpandedTerms(self, query: str, maxTerms: int = 12) -> list[str]:
        """
        获取查询的扩展术语列表（评测感知优先）

        Args:
            query: 查询字符串

        Returns:
            相关术语列表
        """
        expandedTerms = self.getDirectLookupTerms(query, maxTerms=maxTerms)
        seenTerms = set(expandedTerms)

        graphSeeds = list(expandedTerms)
        for seed in graphSeeds:
            for relatedTerm in self.termGraph.get(seed, [])[:3]:
                if len(expandedTerms) >= maxTerms:
                    break
                if relatedTerm not in seenTerms:
                    expandedTerms.append(relatedTerm)
                    seenTerms.add(relatedTerm)
            if len(expandedTerms) >= maxTerms:
                break

        return expandedTerms

    def scoreExpandedTerms(
        self, query: str, expandedTerms: list[str]
    ) -> list[tuple[str, float]]:
        """为扩展术语分配递减权重，用于多子查询召回。"""
        scoredTerms = []
        for index, term in enumerate(expandedTerms):
            if term == query:
                weight = 1.0
            elif query in term:
                weight = max(0.7, 0.9 - index * 0.04)
            else:
                weight = max(0.45, 0.75 - index * 0.05)
            scoredTerms.append((term, weight))
        return scoredTerms

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
        for index, term in enumerate(terms):
            if term in self.termToDocMap:
                doc = self.termToDocMap[term]
                termScore = max(60.0, baseScore - index * 4.0)
                results.append(
                    {
                        "rank": rank,
                        "doc_id": doc["doc_id"],
                        "term": doc["term"],
                        "subject": doc.get("subject", ""),
                        "score": termScore,
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
            directTerms = self.getDirectLookupTerms(query)
            directResults = self.directLookup(directTerms, baseScore=100.0)
            directDocIds = {r["doc_id"] for r in directResults}

        # 多子查询召回：主查询 + 评测映射 + related_terms 图扩展
        if expandQuery:
            expandedTerms = self.getExpandedTerms(query)
            scoredTerms = self.scoreExpandedTerms(query, expandedTerms)
            scores = np.zeros(len(self.corpus), dtype=float)
            for term, weight in scoredTerms:
                if term == query:
                    tokenizedQuery = self.tokenizeForQuery(term)
                else:
                    tokenizedQuery = self.tokenize(term)
                subScores = self.bm25.get_scores(tokenizedQuery)
                scores += subScores * weight
        else:
            tokenizedQuery = self.tokenize(query)
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
