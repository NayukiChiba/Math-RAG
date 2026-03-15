"""评测报告生成入口。"""

from __future__ import annotations

import argparse
import os

from scripts.evaluation import generateReport as report


def main(argv: list[str] | None = None) -> None:
    outputController = report.config.getOutputController()
    parser = argparse.ArgumentParser(description="Math-RAG 评测报告生成脚本")
    parser.add_argument(
        "--results",
        type=str,
        default=os.path.join(
            outputController.get_json_dir(), "full_eval", "all_methods.json"
        ),
        help="全量方法对比报告路径",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=os.path.join(outputController.get_json_dir(), "ablation_study.json"),
        help="消融实验汇总报告路径",
    )
    parser.add_argument(
        "--significance",
        type=str,
        default=os.path.join(outputController.get_json_dir(), "significance_test.json"),
        help="显著性检验报告路径",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=os.path.join(report.config.EVALUATION_DIR, "queries_full.jsonl"),
        help="查询集路径（含 subject 字段）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(outputController.get_json_dir(), "final_report.md"),
        help="输出 Markdown 报告路径",
    )
    parser.add_argument(
        "--figures",
        type=str,
        default=report.config.FIGURES_DIR,
        help="输出图表目录",
    )
    parser.add_argument(
        "--comparison",
        type=str,
        default=os.path.join(
            outputController.get_json_dir(), "comparison_results.json"
        ),
        help="生成质量对比结果路径（来自 evalGenerationComparison.py）",
    )
    args = parser.parse_args(argv)
    report.generate_report(
        results_path=args.results,
        ablation_path=args.ablation,
        significance_path=args.significance,
        queries_path=args.queries,
        output_path=args.output,
        figures_dir=args.figures,
        comparison_path=args.comparison,
    )


if __name__ == "__main__":
    main()
