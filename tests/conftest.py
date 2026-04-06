"""Pytest 根配置：路径、可选环境变量与共享 fixture。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).resolve().parent
_FIXTURE_CHUNK = _TESTS_ROOT / "fixtures" / "chunk_snapshot"


@pytest.fixture
def tests_root() -> Path:
    return _TESTS_ROOT


@pytest.fixture
def chunk_snapshot_dir() -> Path:
    """与 `data/processed/chunk` 同形的最小快照目录。"""
    override = os.environ.get("MATHRAG_TEST_CHUNK_DIR")
    if override:
        return Path(override).resolve()
    return _FIXTURE_CHUNK


@pytest.fixture
def tmp_corpus_and_bm25(
    tmp_path: Path, chunk_snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """由 chunk 快照构建语料 JSONL 与 BM25 索引（临时目录）。"""
    from core.retrieval.corpusBuilder import bridge as bridge_mod
    from core.retrieval.corpusBuilder.builder import buildCorpus
    from core.retrieval.retrieverModules.bm25 import BM25Retriever

    empty_eval = tmp_path / "empty_evaluation"
    empty_eval.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(bridge_mod.config, "EVALUATION_DIR", str(empty_eval))

    corpus_path = tmp_path / "corpus.jsonl"
    buildCorpus(str(chunk_snapshot_dir), str(corpus_path))
    index_path = tmp_path / "bm25_index.pkl"
    r = BM25Retriever(str(corpus_path), str(index_path))
    r.buildIndex()
    r.saveIndex()
    return corpus_path, index_path
