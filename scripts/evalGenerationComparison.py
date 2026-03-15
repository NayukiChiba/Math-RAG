"""生成质量对比评测入口。"""

from __future__ import annotations

import argparse
import json
import os
import sys

import config
from scripts.evaluation import evalGenerationComparison as egc


def main(argv: list[str] | None = None) -> None:
    outputController = config.getOutputController()
    parser = argparse.ArgumentParser(description="生成质量对比：RAG vs 无检索")
    parser.add_argument(
        "--rag-results",
        default=config.RAG_RESULTS_FILE,
        help="RAG 生成结果文件（JSONL）",
    )
    parser.add_argument(
        "--queries",
        default=os.path.join(config.EVALUATION_DIR, "queries.jsonl"),
        help="查询集文件（含 relevant_terms）",
    )
    parser.add_argument(
        "--all-methods",
        default=os.path.join(
            outputController.get_json_dir(), "full_eval", "all_methods.json"
        ),
        help="全量检索评测结果（用于填充检索指标）",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            outputController.get_json_dir(), "comparison_results.json"
        ),
        help="输出对比结果 JSON 路径",
    )
    parser.add_argument(
        "--norag-limit",
        type=int,
        default=15,
        help="无检索基线推理的查询数量（0 = skip）",
    )
    parser.add_argument(
        "--skip-norag",
        action="store_true",
        help="跳过无检索基线推理，仅评估 RAG",
    )
    args = parser.parse_args(argv)
    args.output = outputController.normalize_json_path(
        args.output, "comparison_results.json"
    )

    print("=" * 60)
    print(" Math-RAG 生成质量对比实验")
    print("=" * 60)

    for label, path in [("RAG结果", args.rag_results), ("查询集", args.queries)]:
        if not os.path.isfile(path):
            print(f"[错误] {label}文件不存在: {path}")
            sys.exit(1)

    rag_results = egc._load_rag_results(args.rag_results)
    gold_map = egc._load_queries(args.queries)
    retrieval_metrics = egc._load_retrieval_metrics(args.all_methods, method="BM25+")
    print(f"\n[数据] RAG 结果: {len(rag_results)} 条")
    print(f"[数据] 查询集: {len(gold_map)} 条")
    print(f"[数据] BM25+ Recall@5: {retrieval_metrics['recall@5']:.4f}")

    groups = []

    rag_group = egc.evaluate_rag_results(rag_results, gold_map, retrieval_metrics)
    groups.append(rag_group)

    run_norag = not args.skip_norag and args.norag_limit > 0
    if run_norag:
        all_queries = sorted(gold_map.values(), key=lambda q: q["query"])
        norag_queries = all_queries[: args.norag_limit]
        norag_group = egc.run_norag_baseline(norag_queries, gold_map)
        groups.append(norag_group)
    else:
        print("\n[跳过] 无检索基线推理")

    report = egc.build_comparison_report(groups)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 对比结果已保存: {args.output}")

    egc.print_markdown_table(groups)


if __name__ == "__main__":
    main()
