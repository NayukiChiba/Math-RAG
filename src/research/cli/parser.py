"""研究线 CLI：注册全部实验/评测/报告子命令。"""

from __future__ import annotations

import argparse

from core.cli.runner import run_module_main


def _handle_passthrough(module_name: str, passthrough_args: list[str]) -> None:
    run_module_main(module_name, passthrough_args)


def _handle_serve_experiment(args: argparse.Namespace) -> None:
    module_name = "research.runners.experimentWebUI"
    passthrough = ["--port", str(args.port)]
    if args.share:
        passthrough.append("--share")
    run_module_main(module_name, passthrough)


def _handle_stats(_: argparse.Namespace) -> None:
    from research.dataStat import run_statistics

    run_statistics()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="math-rag-research",
        description="Math-RAG 论文研究线 CLI（评测/实验/报告）",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, module_name, help_text in [
        (
            "generate-queries",
            "research.evaluationData.generateQueries",
            "生成评测查询集",
        ),
        ("build-term-mapping", "research.runners.buildTermMapping", "构建评测术语映射"),
        (
            "eval-retrieval",
            "research.modelEvaluation.evalRetrieval",
            "运行正式检索评测",
        ),
        ("experiments", "research.runners.runExperiments", "运行端到端对比实验"),
        (
            "eval-generation",
            "research.modelEvaluation.evalGeneration",
            "运行生成质量评测",
        ),
        (
            "eval-generation-comparison",
            "research.runners.evalGenerationComparison",
            "生成质量对比评测",
        ),
        (
            "significance-test",
            "research.runners.significanceTest",
            "统计显著性检验",
        ),
        (
            "report",
            "reports_generation.reports.generateReport",
            "生成最终评测报告",
        ),
        (
            "quick-eval",
            "reports_generation.quick_eval.quickEval",
            "快速检索评测",
        ),
        (
            "defense-figures",
            "reports_generation.reports.generateDefenseFigures",
            "生成答辩图表",
        ),
        (
            "add-missing-terms",
            "research.runners.addMissingTerms",
            "补充缺失术语",
        ),
    ]:
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument(
            "args", nargs=argparse.REMAINDER, help="透传给底层脚本的参数"
        )
        subparser.set_defaults(
            handler=lambda parsed, target=module_name: _handle_passthrough(
                target, parsed.args
            )
        )

    stats = subparsers.add_parser("stats", help="运行术语数据统计")
    stats.set_defaults(handler=_handle_stats)

    serve = subparsers.add_parser("serve", help="启动实验 WebUI")
    serve.add_argument("--port", type=int, default=7861, help="监听端口")
    serve.add_argument("--share", action="store_true", help="生成公网分享链接")
    serve.set_defaults(handler=_handle_serve_experiment)

    return parser
