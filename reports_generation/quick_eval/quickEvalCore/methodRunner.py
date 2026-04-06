"""Method dispatcher for quick evaluation strategies."""

from __future__ import annotations

from typing import Any

from reports_generation.quick_eval.quickEvalCore.evaluator import evaluateMethod
from reports_generation.quick_eval.quickEvalCore.retrievers import (
    createAdvancedRetriever,
    createBM25PlusRetriever,
    createBM25Retriever,
    createHybridPlusRetriever,
    createVectorRetriever,
)


def runMethod(
    method: str, queries: list[dict[str, Any]], top_k: int, assets: Any
) -> dict[str, Any] | None:
    if method == "bm25":
        return evaluateMethod("BM25", createBM25Retriever(assets), queries, top_k)

    if method == "bm25plus":
        return evaluateMethod(
            "BM25+", createBM25PlusRetriever(assets), queries, top_k, expandQuery=True
        )

    if method == "vector":
        return evaluateMethod("Vector", createVectorRetriever(assets), queries, top_k)

    if method == "hybrid_plus":
        return evaluateMethod(
            "Hybrid+",
            createHybridPlusRetriever(assets),
            queries,
            top_k,
            strategy="weighted",
            alpha=0.85,
            beta=0.15,
            recallFactor=8,
        )

    if method == "hybrid_rrf":
        return evaluateMethod(
            "Hybrid+RRF",
            createHybridPlusRetriever(assets),
            queries,
            top_k,
            strategy="rrf",
            recallFactor=8,
        )

    if method == "advanced":
        retriever = createHybridPlusRetriever(assets)

        def advancedSearch(query: str, tk: int):
            return retriever.search(
                query, tk, strategy="rrf", recallFactor=8, expandQuery=True
            )

        return evaluateMethod(
            "Advanced", retriever, queries, top_k, search_func=advancedSearch
        )

    if method == "optimized_hybrid":
        return evaluateMethod(
            "Hybrid-Original",
            createHybridPlusRetriever(assets),
            queries,
            top_k,
            strategy="weighted",
            alpha=0.85,
            beta=0.15,
            recallFactor=8,
            expandQuery=True,
            normalization="percentile",
        )

    if method == "hybrid_more_recall":
        return evaluateMethod(
            "Hybrid-MoreRecall",
            createHybridPlusRetriever(assets),
            queries,
            top_k,
            strategy="weighted",
            alpha=0.80,
            beta=0.20,
            recallFactor=15,
            expandQuery=True,
            normalization="percentile",
        )

    if method == "bm25_heavy":
        return evaluateMethod(
            "BM25-Heavy",
            createHybridPlusRetriever(assets),
            queries,
            top_k,
            strategy="weighted",
            alpha=0.90,
            beta=0.10,
            recallFactor=10,
            expandQuery=True,
            normalization="percentile",
        )

    if method == "bm25_ultra":
        return evaluateMethod(
            "BM25-Ultra",
            createHybridPlusRetriever(assets),
            queries,
            top_k,
            strategy="weighted",
            alpha=0.95,
            beta=0.05,
            recallFactor=6,
            expandQuery=True,
            normalization="percentile",
        )

    if method == "optimized_rrf":
        return evaluateMethod(
            "Optimized-RRF",
            createHybridPlusRetriever(assets),
            queries,
            top_k,
            strategy="rrf",
            rrfK=50,
            recallFactor=8,
            expandQuery=True,
        )

    if method == "extreme_rrf":
        return evaluateMethod(
            "Extreme-RRF",
            createHybridPlusRetriever(assets),
            queries,
            top_k,
            strategy="rrf",
            rrfK=30,
            recallFactor=35,
            expandQuery=True,
        )

    if method == "optimized_advanced":
        return evaluateMethod(
            "Optimized-Advanced",
            createAdvancedRetriever(assets),
            queries,
            top_k,
            useReranker=True,
            rewriteQuery=True,
            recallTopK=150,
            bm25Weight=0.4,
            vectorWeight=0.3,
        )

    if method == "advanced_no_rerank":
        return evaluateMethod(
            "Advanced-NoRerank",
            createAdvancedRetriever(assets),
            queries,
            top_k,
            useReranker=False,
            rewriteQuery=True,
            recallTopK=200,
            bm25Weight=0.5,
            vectorWeight=0.3,
        )

    if method == "advanced_more_rewrite":
        return evaluateMethod(
            "Advanced-MoreRewrite",
            createAdvancedRetriever(assets),
            queries,
            top_k,
            useReranker=True,
            rewriteQuery=True,
            recallTopK=200,
            bm25Weight=0.4,
            vectorWeight=0.3,
            rewriteWeight=0.4,
        )

    if method == "bm25plus_only":
        retriever = createBM25PlusRetriever(assets)
        return evaluateMethod("BM25+-Only", retriever, queries, top_k, expandQuery=True)

    if method == "bm25plus_aggressive":
        retriever = createBM25PlusRetriever(assets)
        return evaluateMethod(
            "BM25+-Aggressive", retriever, queries, top_k * 2, expandQuery=True
        )

    if method == "vector_only":
        return evaluateMethod(
            "Vector-Only", createVectorRetriever(assets), queries, top_k
        )

    if method == "direct_lookup_hybrid":
        return evaluateMethod(
            "DirectLookup-Hybrid",
            createHybridPlusRetriever(assets),
            queries,
            top_k,
            strategy="weighted",
            alpha=0.85,
            beta=0.15,
            recallFactor=8,
            expandQuery=True,
            normalization="percentile",
            useDirectLookup=True,
        )

    if method == "direct_lookup_rrf":
        return evaluateMethod(
            "DirectLookup-RRF",
            createHybridPlusRetriever(assets),
            queries,
            top_k,
            strategy="rrf",
            rrfK=50,
            recallFactor=8,
            expandQuery=True,
            useDirectLookup=True,
        )

    if method == "direct_lookup_bm25_only":
        retriever = createBM25PlusRetriever(assets)
        return evaluateMethod(
            "DirectLookup-BM25",
            retriever,
            queries,
            top_k,
            expandQuery=True,
            injectDirectLookup=True,
        )

    print(f"  未知方法：{method}，跳过")
    return None
