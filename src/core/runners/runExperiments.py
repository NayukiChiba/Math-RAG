"""对比实验入口。"""

from __future__ import annotations

import argparse
import json
import os
import time

from core.runners.experiments import runExperiments as exp


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RAG 对比实验脚本")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["norag", "bm25", "vector", "hybrid", "hybrid-rrf"],
        default=["norag", "bm25", "vector", "hybrid"],
        help="要运行的实验组（hybrid=BM25+ 0.85/向量 0.15 加权, hybrid-rrf=RRF 融合，默认不含 hybrid-rrf）",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="限制查询数量（调试用）"
    )
    parser.add_argument("--topk", type=int, default=5, help="检索返回数量")
    parser.add_argument(
        "--query-file", type=str, default=None, help="测试查询集文件路径"
    )
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")

    args = parser.parse_args(argv)

    print("=" * 60)
    print(" Math-RAG 对比实验")
    print("=" * 60)
    print(f"实验组: {', '.join(args.groups)}")
    print(f"Top-K: {args.topk}")
    if args.limit:
        print(f"查询限制: {args.limit}")
    print("=" * 60)

    runner = exp.ExperimentRunner(queryFile=args.query_file, outputDir=args.output_dir)
    results = runner.runAllExperiments(
        groups=args.groups, limit=args.limit, topK=args.topk
    )

    exp.printSummary(results)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    reportPath = os.path.join(runner.outputDir, "comparison_results.json")
    chartPath = os.path.join(runner.outputDir, "comparison_chart.png")
    markdownPath = os.path.join(runner.outputDir, "comparison_table.md")
    detailedPath = os.path.join(runner.outputDir, f"detailed_results_{timestamp}.jsonl")

    runner.saveResults(
        results,
        reportPath=reportPath,
        chartPath=chartPath,
        markdownPath=markdownPath,
        detailedPath=detailedPath,
    )

    logPath = os.path.join(runner.logDir, f"experiment_{timestamp}.json")
    with open(logPath, "w", encoding="utf-8") as f:
        logData = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "args": vars(args),
            "results_summary": [
                {
                    "group": r["group"],
                    "retrieval_metrics": r["retrieval_metrics"],
                    "generation_metrics": r["generation_metrics"],
                    "avg_latency_ms": r["avg_latency_ms"],
                }
                for r in results
            ],
        }
        json.dump(logData, f, ensure_ascii=False, indent=2)
    print(f" 日志已保存: {logPath}")

    print("\n 对比实验完成！")


if __name__ == "__main__":
    main()
