"""RagPipeline 查询—术语重叠判断。"""

from core.answerGeneration.ragPipeline import RagPipeline


def test_has_lexical_overlap_with_term():
    p = RagPipeline(strategy="bm25", topK=3)
    results = [{"term": "极限", "doc_id": "x"}]
    assert p._hasLexicalOverlap("极限的定义", results) is True


def test_no_overlap_different_vocab():
    p = RagPipeline(strategy="bm25", topK=3)
    results = [{"term": "矩阵", "doc_id": "x"}]
    assert p._hasLexicalOverlap("仅英文 unrelated tokens", results) is False
