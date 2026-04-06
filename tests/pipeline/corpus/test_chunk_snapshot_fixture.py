"""chunk_snapshot_dir fixture 与环境变量 MATHRAG_TEST_CHUNK_DIR 的约定。"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_chunk_snapshot_dir_respects_mathrag_test_chunk_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    custom = tmp_path / "custom_chunk_snapshot"
    custom.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MATHRAG_TEST_CHUNK_DIR", str(custom))
    resolved: Path = request.getfixturevalue("chunk_snapshot_dir")
    assert resolved.resolve() == custom.resolve()
