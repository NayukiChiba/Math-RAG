"""Runner for generation evaluation CLI."""

from __future__ import annotations

import os
import time
from typing import Any

from modelEvaluation.common.ioUtils import saveJsonFile, saveJsonlFile
from modelEvaluation.generationEval.evaluator import (
    evaluateGeneration,
    findBestWorstExamples,
)
from modelEvaluation.generationEval.ioOps import loadGoldQueries, loadRagResults
from modelEvaluation.generationEval.reporting import printExamples, printSummary


def runEvalGeneration(args: Any) -> int:
    print("=" * 60)
    print("📊 Math-RAG 生成质量评测")
    print("=" * 60)
    print(f"RAG 结果: {args.results}")
    print(f"黄金测试集: {args.gold}")
    print(f"BLEU: {'启用' if args.bleu else '禁用'}")
    print(f"ROUGE: {'启用' if args.rouge else '禁用'}")
    print("=" * 60)

    rag_results = loadRagResults(args.results)
    if not rag_results:
        print("❌ 无 RAG 结果，退出")
        return 1

    gold_map = loadGoldQueries(args.gold)
    eval_result = evaluateGeneration(
        rag_results,
        gold_map,
        calculate_bleu=args.bleu,
        calculate_rouge=args.rouge,
    )

    examples = findBestWorstExamples(
        eval_result["detailed_results"], rag_results, n=args.examples
    )

    printSummary(eval_result["summary"])
    printExamples(examples)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results_file": args.results,
        "gold_file": args.gold,
        "summary": eval_result["summary"],
        "examples": examples,
        "detailed_results": eval_result["detailed_results"],
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    saveJsonFile(report, args.output)
    print(f"\n✅ 评测报告已保存: {args.output}")

    jsonl_output = args.output.replace(".json", "_detailed.jsonl")
    saveJsonlFile(eval_result["detailed_results"], jsonl_output)
    print(f"✅ 逐条结果已保存: {jsonl_output}")
    print("\n✅ 评测完成！")
    return 0
