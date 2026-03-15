"""CLI parser for quick evaluation."""

from __future__ import annotations

import argparse

from evaluation.quick_eval.constants import ALL_METHODS


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="快速检索评测系统")
    parser.add_argument(
        "--mode",
        type=str,
        default="basic",
        choices=["basic", "optimized"],
        help="评测模式：basic（默认）或 optimized（多策略对比，目标 Recall@5 > 60%%）",
    )
    parser.add_argument(
        "--num-queries", type=int, default=20, help="抽样查询数量（默认 20）"
    )
    parser.add_argument(
        "--all-queries", action="store_true", help="使用全部查询（不抽样）"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        choices=ALL_METHODS,
        help="手动指定评测方法列表（覆盖 --mode 的默认集合）",
    )
    parser.add_argument(
        "--topk", type=int, default=10, help="评估的 TopK 值（默认 10）"
    )
    parser.add_argument("--output", type=str, help="输出报告文件路径")
    return parser
