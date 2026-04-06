"""本地门控 e2e：BM25 检索 + Mock 生成器，跑通 RagPipeline 全链路（无真实大模型）。"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from core.answerGeneration.ragPipeline import RagPipeline

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("MATHRAG_RUN_E2E") != "1",
        reason="设置 MATHRAG_RUN_E2E=1 后运行本地最小 RAG 全链路；见 tests/README.md",
    ),
]


def test_minimal_rag_chain_bm25_and_mock_generator(tmp_corpus_and_bm25):
    pytest.importorskip("rank_bm25")

    corpus_path, index_path = tmp_corpus_and_bm25
    mock_gen = MagicMock()
    mock_gen.generateFromMessages.return_value = "e2e fixture answer"

    with (
        patch(
            "core.answerGeneration.ragPipeline.createGenerator", return_value=mock_gen
        ),
        patch.object(
            RagPipeline,
            "_shouldRefuseOutOfScope",
            return_value=False,
        ),
    ):
        pipe = RagPipeline(
            strategy="bm25",
            topK=3,
            corpusFile=str(corpus_path),
            bm25IndexFile=str(index_path),
        )

        q = "极限的定义是什么？"
        out = pipe.query(q)

    assert out["query"] == q
    assert out["answer"] == "e2e fixture answer"
    terms_blob = str([t.get("term") for t in out.get("retrieved_terms", [])])
    assert "极限" in terms_blob
    assert "latency" in out and "total_ms" in out["latency"]
    assert isinstance(out.get("sources"), list)
    mock_gen.generateFromMessages.assert_called_once()
