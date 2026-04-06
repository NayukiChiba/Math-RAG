"""ingest 前置：materialize_pdf 对用户错误的异常类型。"""

from __future__ import annotations

from pathlib import Path

import pytest

from core import config
from core.cli.errors import CliUserError
from core.cli.handlers import materialize_pdf


def test_materialize_pdf_rejects_non_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(config, "RAW_DIR", str(tmp_path))
    f = tmp_path / "note.txt"
    f.write_text("x")
    with pytest.raises(CliUserError, match="仅支持 PDF"):
        materialize_pdf(str(f))
