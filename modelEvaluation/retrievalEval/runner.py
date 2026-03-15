"""检索评测 CLI 运行器。

职责：
1. 组织完整评测流程（加载查询 -> 初始化检索器 -> 执行评测 -> 保存报告）。
2. 输出面向人类阅读的控制台摘要。
3. 以退出码表达成功/失败状态，便于脚本化调用。
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

from modelEvaluation.common.ioUtils import saveJsonFile
from modelEvaluation.common.paths import buildRetrievalAssets
from modelEvaluation.retrievalEval.charting import generateComparisonChart
from modelEvaluation.retrievalEval.evaluator import evaluateMethod
from modelEvaluation.retrievalEval.ioOps import loadQueries
from modelEvaluation.retrievalEval.retrievers import initRetrievers


def _printSummaryTable(all_metrics: list[dict[str, Any]]) -> None:
    """以表格形式打印多方法评测汇总。

    输入数据要求：
    - 每个元素需包含 `method`、`avg_metrics`、`avg_query_time`。
    """
    print(f"\n{'=' * 60}")
    print("📊 评测结果汇总")
    print(f"{'=' * 60}")
    print(
        f"{'方法':<20} {'Recall@1':<10} {'Recall@3':<10} {'Recall@5':<10} {'Recall@10':<10} {'MRR':<10} {'MAP':<10} {'nDCG@10':<10} {'查询时间':<10}"
    )
    print("-" * 110)

    for row in all_metrics:
        avg = row["avg_metrics"]
        print(
            f"{row['method']:<20} "
            f"{avg['recall@1']:<10.4f} "
            f"{avg['recall@3']:<10.4f} "
            f"{avg['recall@5']:<10.4f} "
            f"{avg['recall@10']:<10.4f} "
            f"{avg['mrr']:<10.4f} "
            f"{avg['map']:<10.4f} "
            f"{avg['ndcg@10']:<10.4f} "
            f"{row['avg_query_time'] * 1000:<10.2f}ms"
        )


def runEvalRetrieval(args: Any) -> int:
    """执行检索评测主流程。

    参数：
    - args: CLI 解析后的参数对象。

    返回：
    - 0: 评测成功。
    - 1: 评测失败（例如无有效查询、检索器初始化失败、无可用结果）。
    """
    print("=" * 60)
    print("📊 Math-RAG 检索评测")
    print("=" * 60)
    print(f"查询集: {args.queries}")
    print(f"评测方法: {', '.join(args.methods)}")
    print(f"TopK: {args.topk}")
    print(
        f"Hybrid alpha/beta: {args.alpha}/{args.beta}  recallFactor: {args.recallFactor}"
    )
    print("=" * 60)

    # Step 1: 加载并校验查询集。
    queries = loadQueries(args.queries)
    if not queries:
        print("❌ 无有效查询，退出")
        return 1

    subject_count: dict[str, int] = defaultdict(int)
    for query in queries:
        subject_count[query["subject"]] += 1
    print("\n📚 学科分布:")
    for subject, count in sorted(subject_count.items()):
        print(f"  {subject}: {count} 条")

    # Step 2: 初始化用户指定的检索器集合。
    retrievers = initRetrievers(args.methods, buildRetrievalAssets())
    if not retrievers:
        print("❌ 没有可用的检索器，退出")
        return 1

    # Step 3: 逐方法执行评测并聚合结果。
    all_metrics: list[dict[str, Any]] = []
    for method_name, retriever in retrievers.items():
        try:
            metrics = evaluateMethod(
                method_name,
                retriever,
                queries,
                args.topk,
                alpha=args.alpha,
                beta=args.beta,
                recall_factor=args.recallFactor,
            )
            all_metrics.append(metrics)
        except Exception as exc:
            print(f"❌ 评测 {method_name} 失败: {exc}")

    if not all_metrics:
        print("❌ 评测失败，无结果")
        return 1

    # Step 4: 结构化报告持久化，便于后续脚本分析与可视化。
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "queries_file": args.queries,
        "total_queries": len(queries),
        "subject_distribution": dict(subject_count),
        "topk": args.topk,
        "results": all_metrics,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    saveJsonFile(report, args.output)
    print(f"\n✅ 评测报告已保存: {args.output}")

    # Step 5: 控制台汇总 + 可选图表输出。
    _printSummaryTable(all_metrics)
    if args.visualize and len(all_metrics) > 1:
        output_dir = os.path.dirname(args.output)
        generateComparisonChart(all_metrics, output_dir)

    print("\n✅ 评测完成！")
    return 0
