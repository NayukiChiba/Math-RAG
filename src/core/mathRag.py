"""Math-RAG 统一 CLI 可执行入口（实现位于 core.cli）。"""

from core.cli import build_parser, main

__all__ = ["build_parser", "main"]

if __name__ == "__main__":
    main()
