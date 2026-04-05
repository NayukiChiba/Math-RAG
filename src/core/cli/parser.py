"""集中注册 math-rag 全部子命令与参数。"""

from __future__ import annotations

import argparse

from core import config
from core.cli import handlers


def build_parser() -> argparse.ArgumentParser:
    retrieval_cfg = config.getRetrievalConfig()

    parser = argparse.ArgumentParser(
        prog="math-rag",
        description="Math-RAG 统一命令行入口",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="执行 PDF 入库流水线")
    ingest.add_argument("pdf", help="PDF 路径，或 raw 目录中的 PDF 文件名")
    ingest.add_argument("--ocr-start-page", type=int, help="OCR 起始页码（1-based）")
    ingest.add_argument(
        "--extract-start-page", type=int, help="术语抽取起始页码（1-based）"
    )
    ingest.add_argument(
        "--generate-start-page", type=int, help="结构化生成起始页码（1-based）"
    )
    ingest.add_argument(
        "--skip-generation", action="store_true", help="跳过 data_gen 阶段"
    )
    ingest.add_argument(
        "--skip-index", action="store_true", help="跳过检索语料与索引构建"
    )
    ingest.add_argument("--rebuild-index", action="store_true", help="强制重建检索索引")
    ingest.add_argument("--skip-bm25", action="store_true", help="不构建 BM25 索引")
    ingest.add_argument(
        "--skip-bm25plus", action="store_true", help="不构建 BM25+ 索引"
    )
    ingest.add_argument("--skip-vector", action="store_true", help="不构建向量索引")
    ingest.add_argument(
        "--vector-model",
        default=retrieval_cfg.get("default_vector_model", "BAAI/bge-base-zh-v1.5"),
        help="向量检索模型名称",
    )
    ingest.add_argument(
        "--batch-size", type=int, default=32, help="向量索引构建批次大小"
    )
    ingest.set_defaults(handler=handlers.handle_ingest)

    build_index = subparsers.add_parser("build-index", help="构建检索语料与索引")
    build_index.add_argument("--rebuild", action="store_true", help="强制重建全部索引")
    build_index.add_argument(
        "--skip-bm25", action="store_true", help="不构建 BM25 索引"
    )
    build_index.add_argument(
        "--skip-bm25plus", action="store_true", help="不构建 BM25+ 索引"
    )
    build_index.add_argument(
        "--skip-vector", action="store_true", help="不构建向量索引"
    )
    build_index.add_argument(
        "--vector-model",
        default=retrieval_cfg.get("default_vector_model", "BAAI/bge-base-zh-v1.5"),
        help="向量检索模型名称",
    )
    build_index.add_argument(
        "--batch-size", type=int, default=32, help="向量索引构建批次大小"
    )
    build_index.set_defaults(handler=handlers.handle_build_index)

    for command, module_name, help_text in [
        ("generate-queries", "core.evaluationData.generateQueries", "生成评测查询集"),
        ("build-term-mapping", "core.runners.buildTermMapping", "构建评测术语映射"),
        ("eval-retrieval", "core.modelEvaluation.evalRetrieval", "运行正式检索评测"),
        ("rag", "core.runners.runRag", "运行 RAG 问答"),
        ("experiments", "core.runners.runExperiments", "运行端到端对比实验"),
        ("eval-generation", "core.modelEvaluation.evalGeneration", "运行生成质量评测"),
        (
            "report",
            "reports_generation.reports.generateReport",
            "生成最终评测报告",
        ),
    ]:
        subparser = subparsers.add_parser(command, help=help_text)
        subparser.add_argument(
            "args", nargs=argparse.REMAINDER, help="透传给底层脚本的参数"
        )
        subparser.set_defaults(
            handler=lambda parsed, target=module_name: handlers.handle_passthrough(
                target, parsed.args
            )
        )

    stats = subparsers.add_parser("stats", help="运行术语数据统计")
    stats.set_defaults(handler=handlers.handle_stats)

    serve = subparsers.add_parser("serve", help="启动 WebUI")
    serve.add_argument(
        "--target",
        choices=["webui", "experiment-webui"],
        default="webui",
        help="服务目标",
    )
    serve.add_argument("--port", type=int, default=7860, help="监听端口")
    serve.add_argument("--share", action="store_true", help="生成公网分享链接")
    serve.set_defaults(handler=handlers.handle_serve)

    return parser
