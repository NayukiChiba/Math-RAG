"""Runner and report writer for quick modelEvaluation."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from core import config
from core.modelEvaluation.common.paths import buildRetrievalAssets
from reports_generation.quick_eval.quickEvalCore.constants import (
    BASIC_METHODS,
    OPTIMIZED_METHODS,
)
from reports_generation.quick_eval.quickEvalCore.dataOps import loadCorpus, loadQueries
from reports_generation.quick_eval.quickEvalCore.methodRunner import runMethod


def runEval(
    methods: list[str] | None = None,
    mode: str = "basic",
    num_queries: int = 20,
    all_queries: bool = False,
    top_k: int = 10,
) -> dict[str, Any]:
    is_optimized = mode == "optimized"

    print("=" * 60)
    print(
        " 优化版检索评测系统 - 目标 Recall@5 > 60%"
        if is_optimized
        else " 快速检索评测系统"
    )
    print("=" * 60)

    if methods is None:
        methods = OPTIMIZED_METHODS if is_optimized else BASIC_METHODS

    queries_dir = config.EVALUATION_DIR
    if not os.path.exists(queries_dir):
        queries_dir = os.path.join(config.PROCESSED_DIR, "evaluation")

    rg = config.getReportsGenerationConfig()
    queries_file = os.path.join(queries_dir, rg["queries_basename"])
    assets = buildRetrievalAssets()

    print(f"\n 查询文件：{queries_file}")
    print(f" 语料文件：{assets.corpus_file}")

    queries = loadQueries(
        queries_file, num_queries if not all_queries else None, all_queries
    )
    if not queries:
        print(" 没有可用的查询，退出评测")
        return {}

    loadCorpus(assets.corpus_file)

    all_metrics: dict[str, Any] = {}
    for method in methods:
        metrics = runMethod(method, queries, top_k, assets)
        if metrics is not None:
            all_metrics[method] = metrics

    print("\n" + "=" * 60)
    print(" 优化版评测对比报告" if is_optimized else " 评测对比报告")
    print("=" * 60)

    col_width = 20 if is_optimized else 15
    print(
        f"\n{'方法':<{col_width}} {'R@1':>8} {'R@3':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'nDCG@5':>8} {'时间 (s)':>8}"
    )
    print("-" * (col_width + 60))

    for _, metric in all_metrics.items():
        avg = metric["avg_metrics"]
        time_str = (
            f"{metric.get('avg_query_time', 0.0):.3f}"
            if "avg_query_time" in metric
            else "N/A"
        )
        print(
            f"{metric['method']:<{col_width}} "
            f"{avg['recall@1']:.2%}  "
            f"{avg['recall@3']:.2%}  "
            f"{avg['recall@5']:.2%}  "
            f"{avg['recall@10']:.2%}  "
            f"{avg['mrr']:.4f}  "
            f"{avg['ndcg@5']:.4f}  "
            f"{time_str:>8}"
        )

    if not all_metrics:
        print("\n  没有有效的评测结果")
        return all_metrics

    best_method = max(
        all_metrics.keys(),
        key=lambda name: all_metrics[name]["avg_metrics"]["recall@5"],
    )
    best_r5 = all_metrics[best_method]["avg_metrics"]["recall@5"]

    if is_optimized:
        target = 0.60
        status = "" if best_r5 >= target else ""
        tail = (
            "  - 达到目标!"
            if best_r5 >= target
            else f"  - 距离 60% 还差 {((target - best_r5) * 100):.1f}%"
        )
        print(
            f"\n{status} Recall@5 最佳方法：{all_metrics[best_method]['method']} ({best_r5:.2%}){tail}"
        )
    else:
        print(
            f"\n Recall@5 最佳方法：{all_metrics[best_method]['method']} ({best_r5:.2%})"
        )

    return all_metrics


def saveReport(metrics: dict[str, Any], output_file: str) -> None:
    dirname = os.path.dirname(output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f" 评测报告已保存：{output_file}")
