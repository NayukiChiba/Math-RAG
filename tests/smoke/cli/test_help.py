"""双 CLI 入口 `--help` 可执行（子进程，不跑业务逻辑）。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_help(module: str) -> None:
    src = str(_REPO_ROOT / "src")
    env = {**os.environ, "PYTHONPATH": f"{src}{os.pathsep}{_REPO_ROOT}"}
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
