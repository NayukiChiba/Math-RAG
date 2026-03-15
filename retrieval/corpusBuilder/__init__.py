"""语料构建子包：从术语 JSON 文件生成检索语料 JSONL。"""

from retrieval.corpusBuilder.bridge import buildBridgeCorpusItems
from retrieval.corpusBuilder.builder import buildCorpus, validateCorpusFile
from retrieval.corpusBuilder.io import loadJsonFile, loadQueriesFile
from retrieval.corpusBuilder.text import buildTextFromTerm, extractCorpusItem

__all__ = [
    "buildBridgeCorpusItems",
    "buildCorpus",
    "buildTextFromTerm",
    "extractCorpusItem",
    "loadJsonFile",
    "loadQueriesFile",
    "validateCorpusFile",
]
