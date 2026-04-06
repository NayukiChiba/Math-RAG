"""研究线 CLI 门面。"""

from __future__ import annotations

from research.cli.parser import build_parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


__all__ = ["build_parser", "main"]
