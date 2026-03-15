"""Method constants for quick evaluation."""

BASIC_METHODS = ["bm25", "bm25plus", "hybrid_plus"]

OPTIMIZED_METHODS = [
    "bm25_heavy",
    "hybrid_more_recall",
    "optimized_hybrid",
    "optimized_rrf",
    "optimized_advanced",
    "extreme_rrf",
]

ALL_METHODS = [
    "bm25",
    "bm25plus",
    "vector",
    "hybrid_plus",
    "hybrid_rrf",
    "advanced",
    "optimized_hybrid",
    "hybrid_more_recall",
    "bm25_heavy",
    "bm25_ultra",
    "optimized_rrf",
    "extreme_rrf",
    "optimized_advanced",
    "advanced_no_rerank",
    "advanced_more_rewrite",
    "bm25plus_only",
    "bm25plus_aggressive",
    "vector_only",
    "direct_lookup_hybrid",
    "direct_lookup_rrf",
    "direct_lookup_bm25_only",
]
