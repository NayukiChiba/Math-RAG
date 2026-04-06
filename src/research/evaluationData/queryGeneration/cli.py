"""Argument parser for query generation CLI."""

from __future__ import annotations

import argparse


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="自动生成评测查询数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 默认：按固定数量生成
        python evaluationData/generateQueries.py

  # 生成所有符合条件的术语
        python evaluationData/generateQueries.py --all

  # 按50%比例采样
        python evaluationData/generateQueries.py --ratio 0.5

  # 自定义各学科数量
        python evaluationData/generateQueries.py --num-ma 50 --num-gd 30 --num-gl 30
        """,
    )
    parser.add_argument(
        "--all", action="store_true", help="使用所有符合条件的术语（忽略数量限制）"
    )
    parser.add_argument(
        "--ratio", type=float, help="采样比例 (0-1)，如 0.5 表示使用 50%% 的术语"
    )
    parser.add_argument(
        "--num-ma", type=int, default=35, help="数学分析生成数量（默认35）"
    )
    parser.add_argument(
        "--num-gd", type=int, default=20, help="高等代数生成数量（默认20）"
    )
    parser.add_argument(
        "--num-gl", type=int, default=20, help="概率论生成数量（默认20）"
    )
    parser.add_argument(
        "--min-related", type=int, default=1, help="最少相关术语数量阈值（默认1）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（默认：data/evaluation/queries.jsonl）",
    )
    parser.add_argument(
        "--no-merge", action="store_true", help="不与现有数据合并，直接覆盖"
    )
    return parser
