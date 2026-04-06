"""双 CLI 入口 `--help` 可执行（子进程，不跑业务逻辑）。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _pythonpath_with_repo() -> str:
    """在保留现有 PYTHONPATH 的前提下，将 src 与仓库根置于靠前搜索顺序。"""
    src = str(_REPO_ROOT / "src")
    root = str(_REPO_ROOT)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [src, root]
    if existing.strip():
        parts.append(existing)
    return os.pathsep.join(parts)


def _run_help(module: str) -> None:
    env = {**os.environ, "PYTHONPATH": _pythonpath_with_repo()}
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_math_rag_cli_help():
    _run_help("core.mathRag")


def test_math_rag_research_cli_help():
    _run_help("research.researchMain")
