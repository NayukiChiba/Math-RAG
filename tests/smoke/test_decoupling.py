"""core 不得依赖 research / reports_generation 的 Python 模块导入。"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORE_SRC = _REPO_ROOT / "src" / "core"


def _py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def _forbidden_imports_in_module(tree: ast.AST) -> list[tuple[int, str]]:
    """整文件 AST 遍历，避免漏检括号续行等多行 import。"""
    bad: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = (alias.name or "").split(".", 1)[0]
                if base in ("research", "reports_generation"):
                    bad.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base = node.module.split(".", 1)[0]
                if base in ("research", "reports_generation"):
                    bad.append((node.lineno, node.module))
    return bad


def test_core_sources_do_not_import_research_or_reports():
    offenders: list[str] = []
    for path in _py_files(_CORE_SRC):
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as e:
            offenders.append(
                f"{path.relative_to(_REPO_ROOT)}:syntax:{e.lineno}: {e.msg}"
            )
            continue
        for lineno, mod in _forbidden_imports_in_module(tree):
            offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}: {mod}")
    assert not offenders, "禁止的导入：\n" + "\n".join(offenders)
