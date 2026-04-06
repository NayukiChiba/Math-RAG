"""RagPipeline：BM25 检索 + Mock 生成器，不加载本地大模型。"""

from unittest.mock import MagicMock, patch

import pytest

from core.answerGeneration.ragPipeline import RagPipeline

rank_bm25 = pytest.importorskip("rank_bm25")


def test_query_with_mock_generator(tmp_corpus_and_bm25):
    corpus_path, index_path = tmp_corpus_and_bm25
    mock_gen = MagicMock()
    mock_gen.generateFromMessages.return_value = "fixture answer"

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

        out = pipe.query("极限的定义是什么？")
        terms_blob = str([t.get("term") for t in out.get("retrieved_terms", [])])
        assert "极限" in terms_blob
        assert out["answer"] == "fixture answer"
        mock_gen.generateFromMessages.assert_called_once()
