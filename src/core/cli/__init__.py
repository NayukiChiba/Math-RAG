"""Math-RAG 命令行门面：子命令注册见 parser，处理逻辑见 handlers。"""

from __future__ import annotations

import sys

from core.cli.errors import CliUserError
from core.cli.parser import build_parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except CliUserError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1) from e


__all__ = ["build_parser", "main"]
