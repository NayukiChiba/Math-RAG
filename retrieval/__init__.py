"""
检索模块

功能：
- 构建检索语料（buildCorpus.py）
- BM25 / BM25+ 检索
- 向量检索
- 混合检索 / 改进混合检索
- 重排序检索
- 高级多路召回检索
"""

__version__ = "1.0.0"

__all__ = [
    "MATH_SYNONYMS",
    "QueryRewriter",
    "AdvancedRetriever",
    "BM25PlusRetriever",
    "BM25Retriever",
    "HybridPlusRetriever",
    "HybridRetriever",
    "RerankerRetriever",
    "VectorRetriever",
    "loadQueriesFromFile",
    "printResults",
    "saveResults",
]

from retrieval.queryRewrite import MATH_SYNONYMS, QueryRewriter
from retrieval.retrievers import (
    AdvancedRetriever,
    BM25PlusRetriever,
    BM25Retriever,
    HybridPlusRetriever,
    HybridRetriever,
    RerankerRetriever,
    VectorRetriever,
    loadQueriesFromFile,
    printResults,
    saveResults,
)
