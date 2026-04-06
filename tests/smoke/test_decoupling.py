"""core 不得依赖 research / reports_generation 的 Python 模块导入。"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORE_SRC = _REPO_ROOT / "src" / "core"


def _py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def _forbidden_modules(line: str) -> list[str]:
    line = line.split("#", 1)[0].strip()
    if not line or line.startswith("__"):
        return []
    bad: list[str] = []
    try:
        tree = ast.parse(line)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = (alias.name or "").split(".", 1)[0]
                if base in ("research", "reports_generation"):
                    bad.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base = node.module.split(".", 1)[0]
                if base in ("research", "reports_generation"):
                    bad.append(node.module)
    return bad


def test_core_sources_do_not_import_research_or_reports():
    offenders: list[str] = []
    for path in _py_files(_CORE_SRC):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            for mod in _forbidden_modules(line):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}: {mod}")
    assert not offenders, "禁止的导入：\n" + "\n".join(offenders)
