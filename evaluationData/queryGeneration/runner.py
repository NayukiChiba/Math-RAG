"""CLI runner for query generation."""

from __future__ import annotations

import os
import random
from typing import Any

import config
from evaluationData.queryGeneration.generator import generateQueries
from evaluationData.queryGeneration.ioOps import (
    loadAllTerms,
    loadExistingQueries,
    mergeQueries,
    saveQueries,
)


def runGenerateQueries(args: Any) -> int:
    print("=" * 60)
    print(" 自动生成评测查询数据")
    print("=" * 60)

    chunk_dir = config.CHUNK_DIR
    output_file = args.output or os.path.join(config.EVALUATION_DIR, "queries.jsonl")
    random.seed(42)

    terms = loadAllTerms(chunk_dir)
    if not terms:
        print(" 未找到术语数据")
        return 1

    print("\n" + "=" * 60)
    print(" 生成新查询")
    print("=" * 60)

    if args.all:
        print(" 模式: 使用所有符合条件的术语")
        generated = generateQueries(
            terms, min_related_terms=args.min_related, use_all=True
        )
    elif args.ratio is not None:
        print(f" 模式: 按比例采样 ({args.ratio * 100:.0f}%)")
        generated = generateQueries(
            terms,
            min_related_terms=args.min_related,
            sample_ratio=args.ratio,
        )
    else:
        print(" 模式: 固定数量")
        print(f"  - 数学分析: {args.num_ma} 条")
        print(f"  - 高等代数: {args.num_gd} 条")
        print(f"  - 概率论: {args.num_gl} 条")
        generated = generateQueries(
            terms,
            num_per_subject={
                "数学分析": args.num_ma,
                "高等代数": args.num_gd,
                "概率论": args.num_gl,
            },
            min_related_terms=args.min_related,
        )

    if not args.no_merge:
        print("\n" + "=" * 60)
        print(" 合并现有查询")
        print("=" * 60)
        merged = mergeQueries(loadExistingQueries(output_file), generated)
    else:
        print("\n" + "=" * 60)
        print("  直接覆盖模式（不保留现有数据）")
        print("=" * 60)
        merged = generated

    print("\n" + "=" * 60)
    print(" 保存结果")
    print("=" * 60)
    saveQueries(merged, output_file)

    print("\n" + "=" * 60)
    print(" 完成！")
    print("=" * 60)
    print(f"总查询数: {len(merged)}")
    if not args.no_merge:
        print(f"建议人工审核: {output_file}")
    return 0
