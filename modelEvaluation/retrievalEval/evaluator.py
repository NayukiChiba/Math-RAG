"""Method-level evaluator for retrieval benchmark."""

from __future__ import annotations

import time
from typing import Any

from modelEvaluation.common.metrics import (
    calculateAP,
    calculateMRR,
    calculateNDCG,
    calculateRecallAtK,
)


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluateMethod(
    method: str,
    retriever: Any,
    queries: list[dict[str, Any]],
    top_k: int = 10,
    alpha: float = 0.7,
    beta: float = 0.3,
    recall_factor: int = 10,
) -> dict[str, Any]:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    if alpha + beta <= 0.0:
        raise ValueError(f"alpha + beta must be > 0, got alpha={alpha}, beta={beta}")
    if recall_factor <= 0:
        raise ValueError(f"recallFactor must be > 0, got {recall_factor}")

    print(f"\n{'=' * 60}")
    print(f"📊 评测方法: {method}")
    print(f"{'=' * 60}")

    metrics: dict[str, Any] = {
        "method": method,
        "total_queries": len(queries),
        "recall@1": [],
        "recall@3": [],
        "recall@5": [],
        "recall@10": [],
        "mrr": [],
        "map": [],
        "ndcg@3": [],
        "ndcg@5": [],
        "ndcg@10": [],
        "avg_query_time": 0.0,
    }

    query_times: list[float] = []
    for idx, query in enumerate(queries, 1):
        query_text = query["query"]
        relevant_terms = query["relevant_terms"]

        print(
            f"  查询 {idx}/{len(queries)}: {query_text} (相关术语: {len(relevant_terms)})"
        )

        start = time.time()
        try:
            if method.startswith("Hybrid+"):
                strategy = "weighted" if method == "Hybrid+-Weighted" else "rrf"
                results = retriever.search(
                    query_text,
                    topK=top_k,
                    strategy=strategy,
                    alpha=alpha,
                    beta=beta,
                    recallFactor=recall_factor,
                    expandQuery=True,
                    useDirectLookup=True,
                )
            elif method == "Hybrid-Weighted":
                results = retriever.search(
                    query_text,
                    topK=top_k,
                    strategy="weighted",
                    alpha=alpha,
                    beta=beta,
                    verbose=False,
                )
            elif method == "BM25+":
                results = retriever.search(
                    query_text,
                    topK=top_k,
                    expandQuery=True,
                    injectDirectLookup=True,
                )
            else:
                results = retriever.search(query_text, topK=top_k)
        except Exception as exc:
            print(f"    ❌ 检索失败: {exc}")
            continue

        query_time = time.time() - start
        query_times.append(query_time)

        metrics["recall@1"].append(calculateRecallAtK(results, relevant_terms, 1))
        metrics["recall@3"].append(calculateRecallAtK(results, relevant_terms, 3))
        metrics["recall@5"].append(calculateRecallAtK(results, relevant_terms, 5))
        metrics["recall@10"].append(calculateRecallAtK(results, relevant_terms, 10))
        metrics["mrr"].append(calculateMRR(results, relevant_terms))
        metrics["map"].append(calculateAP(results, relevant_terms))
        metrics["ndcg@3"].append(calculateNDCG(results, relevant_terms, 3))
        metrics["ndcg@5"].append(calculateNDCG(results, relevant_terms, 5))
        metrics["ndcg@10"].append(calculateNDCG(results, relevant_terms, 10))

        print(f"    ⏱️  查询时间: {query_time * 1000:.2f}ms")

    metrics["avg_query_time"] = _avg(query_times)
    avg_metrics = {
        "recall@1": _avg(metrics["recall@1"]),
        "recall@3": _avg(metrics["recall@3"]),
        "recall@5": _avg(metrics["recall@5"]),
        "recall@10": _avg(metrics["recall@10"]),
        "mrr": _avg(metrics["mrr"]),
        "map": _avg(metrics["map"]),
        "ndcg@3": _avg(metrics["ndcg@3"]),
        "ndcg@5": _avg(metrics["ndcg@5"]),
        "ndcg@10": _avg(metrics["ndcg@10"]),
    }

    print("\n📈 平均指标:")
    print(f"  Recall@1:  {avg_metrics['recall@1']:.4f}")
    print(f"  Recall@3:  {avg_metrics['recall@3']:.4f}")
    print(f"  Recall@5:  {avg_metrics['recall@5']:.4f}")
    print(f"  Recall@10: {avg_metrics['recall@10']:.4f}")
    print(f"  MRR:       {avg_metrics['mrr']:.4f}")
    print(f"  MAP:       {avg_metrics['map']:.4f}")
    print(f"  nDCG@3:    {avg_metrics['ndcg@3']:.4f}")
    print(f"  nDCG@5:    {avg_metrics['ndcg@5']:.4f}")
    print(f"  nDCG@10:   {avg_metrics['ndcg@10']:.4f}")
    print(f"  平均查询时间: {metrics['avg_query_time'] * 1000:.2f}ms")

    metrics["avg_metrics"] = avg_metrics
    return metrics
