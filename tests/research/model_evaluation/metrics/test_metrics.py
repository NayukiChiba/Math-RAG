"""研究线检索指标纯函数。"""

from research.modelEvaluation.common.metrics import (
    calculateMRR,
    calculateNDCG,
    calculateRecallAtK,
)


def test_recall_at_k():
    results = [{"term": "a"}, {"term": "b"}, {"term": "c"}]
    assert calculateRecallAtK(results, ["a", "b"], k=2) == 1.0
    assert calculateRecallAtK(results, ["x"], k=5) == 0.0


def test_mrr():
    results = [{"term": "x"}, {"term": "hit"}]
    assert calculateMRR(results, ["hit"]) == 0.5


def test_ndcg():
    results = [{"term": "a"}, {"term": "b"}]
    assert 0.0 <= calculateNDCG(results, ["a", "b"], k=2) <= 1.0
