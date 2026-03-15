"""Retriever factory functions for quick modelEvaluation."""

from __future__ import annotations

from modelEvaluation.common.paths import RetrievalAssets


def createBM25Retriever(assets: RetrievalAssets):
    from retrieval.retrieverModules import BM25Retriever

    retriever = BM25Retriever(assets.corpus_file, assets.bm25_index_file)
    if not retriever.loadIndex():
        print("  BM25 索引不存在，正在构建...")
        retriever.buildIndex()
        retriever.saveIndex()
    return retriever


def createBM25PlusRetriever(assets: RetrievalAssets):
    from retrieval.retrieverModules import BM25PlusRetriever

    retriever = BM25PlusRetriever(
        assets.corpus_file, assets.bm25plus_index_file, assets.terms_file
    )
    if not retriever.loadIndex():
        print("  BM25+ 索引不存在，正在构建...")
        retriever.buildIndex()
        retriever.saveIndex()
    retriever.loadTermsMap()
    return retriever


def createVectorRetriever(assets: RetrievalAssets):
    from retrieval.retrieverModules import VectorRetriever

    retriever = VectorRetriever(
        assets.corpus_file,
        assets.embedding_model,
        assets.vector_index_file,
        assets.vector_embedding_file,
    )
    if not retriever.loadIndex():
        print("  向量索引不存在，正在构建...")
        retriever.buildIndex()
        retriever.saveIndex()
    return retriever


def createHybridPlusRetriever(assets: RetrievalAssets):
    from retrieval.retrieverModules import HybridPlusRetriever

    return HybridPlusRetriever(
        assets.corpus_file,
        assets.bm25plus_index_file,
        assets.vector_index_file,
        assets.vector_embedding_file,
        assets.embedding_model,
        assets.terms_file,
    )


def createAdvancedRetriever(assets: RetrievalAssets):
    from retrieval.retrieverModules import AdvancedRetriever

    return AdvancedRetriever(
        assets.corpus_file,
        assets.bm25plus_index_file,
        assets.vector_index_file,
        assets.vector_embedding_file,
        modelName=assets.embedding_model,
        termsFile=assets.terms_file,
    )
