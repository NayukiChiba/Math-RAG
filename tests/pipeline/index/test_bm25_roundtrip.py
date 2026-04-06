"""BM25：建索引、落盘、加载、检索。"""

import pytest

from core.retrieval.retrieverModules.bm25 import BM25Retriever

rank_bm25 = pytest.importorskip("rank_bm25")


def test_bm25_build_load_search(tmp_corpus_and_bm25):
    corpus_path, index_path = tmp_corpus_and_bm25
    r2 = BM25Retriever(str(corpus_path), str(index_path))
    assert r2.loadIndex() is True
    hits = r2.search("\u6781\u9650", topK=3)
    assert hits
    assert hits[0].get("doc_id") == "fixture-limit-001"
