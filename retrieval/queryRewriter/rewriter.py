"""查询改写器：基于同义词典对数学查询进行术语扩展。"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from retrieval.queryRewriter.synonyms import MATH_SYNONYMS
from utils import getFileLoader

_LOADER = getFileLoader()


class QueryRewriter:
    """查询改写器"""

    def __init__(self, termsFile: str | None = None):
        """
        初始化查询改写器。

        Args:
            termsFile: 术语文件路径（可选，用于加载额外的术语映射）
        """
        self.termsMap = dict(MATH_SYNONYMS)
        self.termsFile = termsFile
        if termsFile and os.path.exists(termsFile):
            self._loadTermsFromFile(termsFile)

    def _loadTermsFromFile(self, filepath: str) -> None:
        """从文件加载额外的术语映射。"""
        try:
            data = _LOADER.json(filepath)
            for term, info in data.items():
                if isinstance(info, dict):
                    aliases = info.get("aliases", [])
                    if aliases:
                        self.termsMap[term] = aliases
                elif isinstance(info, list) and info:
                    self.termsMap[term] = info
        except Exception as e:
            print(f"  加载术语文件失败：{e}")

    def rewrite(self, query: str, maxTerms: int = 10) -> list[str]:
        """
        改写查询，生成扩展术语列表。

        Args:
            query: 原始查询
            maxTerms: 返回的最大术语数量（最小为 1，至少保留原始查询）

        Returns:
            扩展后的术语列表
        """
        maxTerms = max(1, maxTerms)
        if not query.strip():
            return [query]

        seen = {query}
        uniqueTerms = [query]

        for term, synonyms in self.termsMap.items():
            if len(uniqueTerms) >= maxTerms:
                break
            if term in query or query in term:
                for syn in synonyms:
                    if len(uniqueTerms) >= maxTerms:
                        break
                    if syn not in seen:
                        seen.add(syn)
                        uniqueTerms.append(syn)

        return uniqueTerms

    def rewriteBatch(
        self, queries: list[str], maxTerms: int = 10
    ) -> dict[str, list[str]]:
        """
        批量改写查询。

        Args:
            queries: 查询列表
            maxTerms: 每个查询的最大术语数量

        Returns:
            字典，键为原始查询，值为扩展术语列表
        """
        return {query: self.rewrite(query, maxTerms) for query in queries}
