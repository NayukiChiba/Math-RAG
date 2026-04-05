"""CLI parser for generation modelEvaluation."""

from __future__ import annotations

import argparse
import os

from core import config


def buildParser(default_output: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成质量评测脚本")
    parser.add_argument(
        "--results",
        type=str,
        default=config.RAG_RESULTS_FILE,
        help="RAG 问答结果文件路径",
    )
    parser.add_argument(
        "--gold",
        type=str,
        default=os.path.join(config.EVALUATION_DIR, "queries.jsonl"),
        help="黄金测试集文件路径",
    )
    parser.add_argument(
        "--output", type=str, default=default_output, help="输出报告路径"
    )
    parser.add_argument(
        "--bleu", action="store_true", help="计算 BLEU 分数（需要 nltk）"
    )
    parser.add_argument(
        "--rouge", action="store_true", help="计算 ROUGE 分数（需要 rouge-score）"
    )
    parser.add_argument("--examples", type=int, default=3, help="最好/最差示例数量")
    return parser
