"""查询改写子包：基于数学同义词典对查询进行术语扩展。"""

from retrieval.queryRewriter.rewriter import QueryRewriter
from retrieval.queryRewriter.synonyms import MATH_SYNONYMS

__all__ = ["MATH_SYNONYMS", "QueryRewriter"]
