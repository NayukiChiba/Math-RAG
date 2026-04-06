"""从 chunk 快照构建语料 JSONL。"""

from __future__ import annotations

from pathlib import Path

from core.retrieval.corpusBuilder.builder import buildCorpus, validateCorpusFile


def test_build_corpus_from_fixture_chunk(tmp_path: Path, chunk_snapshot_dir: Path):
    out = tmp_path / "corpus.jsonl"
    stats = buildCorpus(str(chunk_snapshot_dir), str(out))
    assert stats["validFiles"] >= 1
    assert stats["corpusItems"] >= 1
    assert out.is_file()

    validation = validateCorpusFile(str(out))
    assert validation["valid"] is True
    assert validation["validLines"] >= 1
