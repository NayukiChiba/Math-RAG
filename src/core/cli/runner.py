"""CLI 透传：在隔离的 sys.argv 下调用其它模块的 main()。"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from importlib import import_module


@contextmanager
def temporary_argv(argv: list[str]):
    original = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = original


def run_module_main(module_name: str, argv: list[str] | None = None) -> None:
    module = import_module(module_name)
    if not hasattr(module, "main"):
        raise AttributeError(f"模块 {module_name} 未导出 main()")

    cli_argv = [module_name]
    if argv:
        cli_argv.extend(argv)

    with temporary_argv(cli_argv):
        module.main()
