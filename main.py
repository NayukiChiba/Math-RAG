"""Math-RAG 统一启动入口。

直接使用 python main.py，无需调用内部模块入口。

用法
----
    python main.py cli      <子命令> [参数]  # 产品线 CLI
    python main.py research <子命令> [参数]  # 研究线 CLI
    python main.py ui       [--port N] [--share] [--research]

快速示例
--------
    python main.py cli ingest data/raw/数学分析.pdf
    python main.py cli rag --query "什么是一致收敛？"
    python main.py ui
    python main.py research full-reports --retrieval-only
"""

from __future__ import annotations

import argparse
import sys
import textwrap

# ── 产品线 CLI 子命令说明（与 src/core/cli/parser.py 对应） ───────────────
_CLI_COMMAND_HELP = """\
产品线子命令（运行各命令可加 --help 查看详细参数）:
  ingest         PDF 入库流水线
                   OCR → 术语抽取 → 结构化生成 → 检索索引构建
                   示例: python main.py cli ingest data/raw/数学分析.pdf
                         python main.py cli ingest 数学分析.pdf --skip-generation

  build-index    仅构建/重建检索语料与索引
                   示例: python main.py cli build-index
                         python main.py cli build-index --rebuild --skip-bm25

  rag            RAG 问答（单条或批量）
                   示例: python main.py cli rag --query "什么是一致收敛？"
                         python main.py cli rag --input data/evaluation/queries.jsonl

  serve          启动产品线 Gradio WebUI（同 python main.py ui）
                   示例: python main.py cli serve --port 7860 --share
"""

# ── 研究线 CLI 子命令说明（与 src/research/cli/parser.py 对应） ──────────
_RESEARCH_COMMAND_HELP = """\
研究线子命令（运行各命令可加 --help 查看详细参数）:
  eval-retrieval              正式检索评测（BM25/Vector/Hybrid 等方法对比）
                                示例: python main.py research eval-retrieval --visualize
                                      python main.py research eval-retrieval \\
                                        --queries data/evaluation/queries_full.jsonl

  full-reports                全量评测总控（检索→消融→显著性→报告），
                                日志写入 outputs/log/<run_id>/，定稿写入 outputs/reports/
                                示例: python main.py research full-reports
                                      python main.py research full-reports \\
                                        --retrieval-only --continue-on-error

  publish-reports             从已有 log 跑次重新发布定稿到 outputs/reports/
                                示例: python main.py research publish-reports \\
                                        --run-id 20260406_164049

  experiments                 端到端对比实验（norag/bm25/vector/hybrid）
                                示例: python main.py research experiments --limit 10

  eval-generation             生成质量评测
  eval-generation-comparison  生成质量对比评测
  significance-test           统计显著性检验（配对 t 检验 / Bootstrap）
  report                      生成最终 Markdown 评测报告与图表
  quick-eval                  快速检索评测（调试用）
  defense-figures             生成答辩演示图表（写入 outputs/figures/defense/）
  add-missing-terms           分析并补充语料缺失术语
  generate-queries            生成评测查询集
  build-term-mapping          构建评测术语映射
  stats                       术语与语料统计可视化
  serve                       启动研究线实验对比 Gradio WebUI
                                示例: python main.py research serve --port 7861
"""

# ── 顶层 epilog ───────────────────────────────────────────────────────────
_ROOT_EPILOG = textwrap.dedent("""\
WebUI 快速启动:
  python main.py ui                          产品线 RAG 问答界面（端口 7860）
  python main.py ui --port 7861 --share      自定义端口并生成公网链接
  python main.py ui --research               研究线实验对比界面（端口 7861）

产品线 CLI 快速示例:
  python main.py cli ingest data/raw/数学分析.pdf
  python main.py cli rag --query "什么是一致收敛？"
  python main.py cli build-index --rebuild

研究线 CLI 快速示例:
  python main.py research eval-retrieval --visualize
  python main.py research full-reports --retrieval-only
  python main.py research publish-reports --run-id 20260406_164049

运行子命令 --help 查看详细参数:
  python main.py cli --help
  python main.py research --help
  python main.py ui --help
""")


def _cli(passthrough: list[str]) -> None:
    from core.cli import main as core_main

    sys.argv = ["math-rag"] + passthrough
    core_main()


def _research(passthrough: list[str]) -> None:
    from research.researchMain import main as research_main

    sys.argv = ["math-rag-research"] + passthrough
    research_main()


def _ui(port: int, share: bool, research: bool) -> None:
    from core.cli.runner import run_module_main

    if research:
        argv = ["--port", str(port)]
        if share:
            argv.append("--share")
        run_module_main("research.runners.experimentWebUI", argv)
    else:
        argv = ["--port", str(port)]
        if share:
            argv.append("--share")
        run_module_main("core.answerGeneration.webui", argv)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Math-RAG 统一启动入口\n\n"
            "  python main.py cli      — 产品线 CLI（入库/问答/索引）\n"
            "  python main.py research — 研究线 CLI（评测/实验/报告）\n"
            "  python main.py ui       — 启动 Gradio WebUI\n\n"
            "运行 `python main.py <模式> --help` 查看该模式的详细说明。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_ROOT_EPILOG,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ── cli：产品线 ──────────────────────────────────────────────────────
    cli_sub = subparsers.add_parser(
        "cli",
        help="运行产品线 CLI 子命令（ingest / rag / build-index / serve ...）",
        description="产品线 CLI — 将参数透传给 math-rag 内部解析器。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_COMMAND_HELP,
    )
    cli_sub.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        metavar="<子命令> [参数]",
        help="产品线子命令及其参数（见下方命令列表）",
    )

    # ── research：研究线 ─────────────────────────────────────────────────
    research_sub = subparsers.add_parser(
        "research",
        help="运行研究线 CLI 子命令（评测/实验/报告 ...）",
        description="研究线 CLI — 将参数透传给 math-rag-research 内部解析器。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_RESEARCH_COMMAND_HELP,
    )
    research_sub.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        metavar="<子命令> [参数]",
        help="研究线子命令及其参数（见下方命令列表）",
    )

    # ── ui：Gradio WebUI ─────────────────────────────────────────────────
    ui_sub = subparsers.add_parser(
        "ui",
        help="启动 Gradio WebUI（产品线 RAG 界面或研究线实验界面）",
        description=(
            "启动 Gradio WebUI。\n\n"
            "  默认（不加 --research）：产品线 RAG 问答界面，端口 7860。\n"
            "  --research：研究线多方法对比实验界面，端口 7861。\n\n"
            "示例:\n"
            "  python main.py ui\n"
            "  python main.py ui --port 7861 --share\n"
            "  python main.py ui --research --port 7861"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ui_sub.add_argument(
        "--port",
        type=int,
        default=None,
        help="监听端口（产品线默认 7860，研究线默认 7861）",
    )
    ui_sub.add_argument(
        "--share",
        action="store_true",
        help="生成 Gradio 公网分享链接（需要网络访问）",
    )
    ui_sub.add_argument(
        "--research",
        action="store_true",
        help="启动研究线实验对比 WebUI（默认启动产品线 RAG WebUI）",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode == "cli":
        _cli(args.args)
    elif args.mode == "research":
        _research(args.args)
    elif args.mode == "ui":
        default_port = 7861 if args.research else 7860
        port = args.port if args.port is not None else default_port
        _ui(port=port, share=args.share, research=args.research)


if __name__ == "__main__":
    main()
