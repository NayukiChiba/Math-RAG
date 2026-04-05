"""Math-RAG 命令行门面：子命令注册见 parser，处理逻辑见 handlers。"""

from __future__ import annotations

from core.cli.parser import build_parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


__all__ = ["build_parser", "main"]
