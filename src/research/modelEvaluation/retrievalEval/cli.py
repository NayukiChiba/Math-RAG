"""CLI parser for retrieval modelEvaluation."""

from __future__ import annotations

import argparse

from core import config


def buildParser(default_output: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检索评测脚本")
    parser.add_argument(
        "--queries",
        type=str,
        default=f"{config.EVALUATION_DIR}/queries.jsonl",
        help="查询集文件路径",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bm25plus", "vector", "hybrid-plus-weighted", "hybrid-plus-rrf"],
        choices=[
            "bm25",
            "bm25plus",
            "vector",
            "hybrid-weighted",
            "hybrid-plus-weighted",
            "hybrid-plus-rrf",
        ],
        help="要评测的检索方法",
    )
    parser.add_argument("--topk", type=int, default=10, help="TopK 阈值")
    parser.add_argument("--visualize", action="store_true", help="生成对比图表")
    parser.add_argument(
        "--output", type=str, default=default_output, help="输出报告路径"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.7, help="混合检索 BM25 权重（默认 0.7）"
    )
    parser.add_argument(
        "--beta", type=float, default=0.3, help="混合检索向量权重（默认 0.3）"
    )
    parser.add_argument(
        "--recall-factor",
        type=int,
        default=10,
        dest="recallFactor",
        help="混合检索召回因子（默认 10）",
    )
    return parser
