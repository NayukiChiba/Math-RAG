"""Retriever initialization for retrieval evaluation."""

from __future__ import annotations

from typing import Any

from evaluation.common.paths import RetrievalAssets


def initRetrievers(methods: list[str], assets: RetrievalAssets) -> dict[str, Any]:
    retrievers: dict[str, Any] = {}

    for method in methods:
        print(f"\n🔄 初始化检索器: {method.upper()}")
        try:
            if method == "bm25":
                from retrieval.retrievers import BM25Retriever

                retriever = BM25Retriever(assets.corpus_file, assets.bm25_index_file)
                if not retriever.loadIndex():
                    print("  索引不存在，开始构建...")
                    retriever.buildIndex()
                    retriever.saveIndex()
                retrievers["BM25"] = retriever
            elif method == "bm25plus":
                from retrieval.retrievers import BM25PlusRetriever

                retriever = BM25PlusRetriever(
                    assets.corpus_file,
                    assets.bm25plus_index_file,
                    assets.terms_file,
                )
                if not retriever.loadIndex():
                    print("  索引不存在，开始构建...")
                    retriever.buildIndex()
                    retriever.saveIndex()
                retriever.loadTermsMap()
                retrievers["BM25+"] = retriever
            elif method == "vector":
                from retrieval.retrievers import VectorRetriever

                retriever = VectorRetriever(
                    assets.corpus_file,
                    assets.embedding_model,
                    indexFile=assets.vector_index_file,
                    embeddingFile=assets.vector_embedding_file,
                )
                if not retriever.loadIndex():
                    print("  索引不存在，开始构建...")
                    retriever.buildIndex()
                    retriever.saveIndex()
                retrievers["Vector"] = retriever
            elif method == "hybrid-plus-weighted":
                from retrieval.retrievers import HybridPlusRetriever

                retrievers["Hybrid+-Weighted"] = HybridPlusRetriever(
                    assets.corpus_file,
                    assets.bm25plus_index_file,
                    assets.vector_index_file,
                    assets.vector_embedding_file,
                    assets.embedding_model,
                    assets.terms_file,
                )
            elif method == "hybrid-plus-rrf":
                from retrieval.retrievers import HybridPlusRetriever

                retrievers["Hybrid+-RRF"] = HybridPlusRetriever(
                    assets.corpus_file,
                    assets.bm25plus_index_file,
                    assets.vector_index_file,
                    assets.vector_embedding_file,
                    assets.embedding_model,
                    assets.terms_file,
                )
            elif method == "hybrid-weighted":
                from retrieval.retrievers import HybridRetriever

                retrievers["Hybrid-Weighted"] = HybridRetriever(
                    assets.corpus_file,
                    assets.bm25_index_file,
                    assets.vector_index_file,
                    assets.vector_embedding_file,
                    assets.embedding_model,
                )
        except (ImportError, SystemExit) as exc:
            print(f"❌ 初始化失败（缺少依赖）: {exc}")
            print(f"💡 提示: 请检查 {method.upper()} 所需的依赖库是否已安装")
        except Exception as exc:
            print(f"❌ 初始化失败: {exc}")
            import traceback

            traceback.print_exc()

    return retrievers
