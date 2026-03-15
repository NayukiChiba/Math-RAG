"""Method evaluator for quick modelEvaluation."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from modelEvaluation.common.metrics import (
    calculateMAP,
    calculateMRR,
    calculateNDCG,
    calculateRecallAtK,
)


def evaluateMethod(
    method: str,
    retriever: Any,
    queries: list[dict[str, Any]],
    top_k: int = 10,
    search_func: str | Callable = "search",
    **search_kwargs,
) -> dict[str, Any]:
    print(f"\n{'=' * 60}")
    print(f" 评测方法：{method}")
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

    for i, query in enumerate(queries, 1):
        query_text = query["query"]
        relevant_terms = query["relevant_terms"]
        start_time = time.time()

        if callable(search_func):
            results = search_func(query_text, top_k * 2, **search_kwargs)
        elif search_func == "search":
            results = retriever.search(query_text, top_k * 2, **search_kwargs)
        elif search_func == "batchSearch":
            results = retriever.batchSearch(
                [query_text], top_k * 2, **search_kwargs
            ).get(query_text, [])
        else:
            results = getattr(retriever, search_func)(
                query_text, top_k * 2, **search_kwargs
            )

        query_time = time.time() - start_time
        query_times.append(query_time)

        metrics["recall@1"].append(calculateRecallAtK(results, relevant_terms, 1))
        metrics["recall@3"].append(calculateRecallAtK(results, relevant_terms, 3))
        metrics["recall@5"].append(calculateRecallAtK(results, relevant_terms, 5))
        metrics["recall@10"].append(calculateRecallAtK(results, relevant_terms, 10))
        metrics["mrr"].append(calculateMRR(results, relevant_terms))
        metrics["map"].append(calculateMAP(results, relevant_terms))
        metrics["ndcg@3"].append(calculateNDCG(results, relevant_terms, 3))
        metrics["ndcg@5"].append(calculateNDCG(results, relevant_terms, 5))
        metrics["ndcg@10"].append(calculateNDCG(results, relevant_terms, 10))

        if i % 10 == 0 or i == len(queries):
            print(f"  进度：{i}/{len(queries)} ({i / len(queries) * 100:.1f}%)")

    metrics["avg_query_time"] = (
        sum(query_times) / len(query_times) if query_times else 0.0
    )
    metrics["avg_metrics"] = {
        "recall@1": sum(metrics["recall@1"]) / len(metrics["recall@1"]),
        "recall@3": sum(metrics["recall@3"]) / len(metrics["recall@3"]),
        "recall@5": sum(metrics["recall@5"]) / len(metrics["recall@5"]),
        "recall@10": sum(metrics["recall@10"]) / len(metrics["recall@10"]),
        "mrr": sum(metrics["mrr"]) / len(metrics["mrr"]),
        "map": sum(metrics["map"]) / len(metrics["map"]),
        "ndcg@3": sum(metrics["ndcg@3"]) / len(metrics["ndcg@3"]),
        "ndcg@5": sum(metrics["ndcg@5"]) / len(metrics["ndcg@5"]),
        "ndcg@10": sum(metrics["ndcg@10"]) / len(metrics["ndcg@10"]),
    }

    print(f"\n {method} 评测摘要:")
    print(f"  Recall@5: {metrics['avg_metrics']['recall@5']:.2%}")
    print(f"  Recall@10: {metrics['avg_metrics']['recall@10']:.2%}")
    print(f"  MRR: {metrics['avg_metrics']['mrr']:.4f}")
    print(f"  nDCG@5: {metrics['avg_metrics']['ndcg@5']:.4f}")
    print(f"  平均查询时间：{metrics['avg_query_time']:.4f}s")

    return metrics
