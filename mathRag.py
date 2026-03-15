"""Math-RAG 统一 CLI 实现。"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import config
from dataStat import run_statistics
from retrieval import (
    BM25PlusRetriever,
    BM25Retriever,
    VectorRetriever,
    run_build_corpus,
)


@contextmanager
def _temporary_argv(argv: list[str]):
    original = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = original


def _run_module_main(module_name: str, argv: list[str] | None = None) -> None:
    module = import_module(module_name)
    if not hasattr(module, "main"):
        raise AttributeError(f"模块 {module_name} 未导出 main()")

    cli_argv = [module_name]
    if argv:
        cli_argv.extend(argv)

    with _temporary_argv(cli_argv):
        module.main()


def _materialize_pdf(pdf_arg: str) -> tuple[str, str]:
    source = Path(pdf_arg).expanduser()
    raw_dir = Path(config.RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if source.exists():
        if source.suffix.lower() != ".pdf":
            raise SystemExit(f"仅支持 PDF 文件: {source}")
        target = raw_dir / source.name
        if source.resolve() != target.resolve():
            if target.exists():
                raise SystemExit(
                    f"目标文件已存在，拒绝覆盖: {target}。"
                    "请改名后重试，或直接传入 raw 目录中的现有文件名。"
                )
            shutil.copy2(source, target)
            print(f"📥 已复制 PDF 到原始目录: {target}")
        else:
            print(f"📂 使用 raw 目录中的 PDF: {target}")
    else:
        target_name = pdf_arg if pdf_arg.lower().endswith(".pdf") else f"{pdf_arg}.pdf"
        target = raw_dir / target_name
        if not target.exists():
            raise SystemExit(f"未找到 PDF: {pdf_arg}")
        print(f"📂 使用 raw 目录中的 PDF: {target}")

    return target.name, target.stem


def _ensure_corpus(*, rebuild: bool) -> str:
    corpus_file = os.path.join(config.PROCESSED_DIR, "retrieval", "corpus.jsonl")
    if rebuild or not os.path.exists(corpus_file):
        run_build_corpus()
    else:
        print(f"✅ 复用已有语料文件: {corpus_file}")
    return corpus_file


def _build_indexes(
    *,
    rebuild: bool,
    skip_bm25: bool,
    skip_bm25plus: bool,
    skip_vector: bool,
    vector_model: str,
    batch_size: int,
) -> None:
    corpus_file = _ensure_corpus(rebuild=rebuild)
    retrieval_dir = os.path.join(config.PROCESSED_DIR, "retrieval")
    terms_file = os.path.join(config.TERMS_DIR, "all_terms.json")
    terms_path = terms_file if os.path.exists(terms_file) else None

    if not skip_bm25:
        bm25_index = os.path.join(retrieval_dir, "bm25_index.pkl")
        retriever = BM25Retriever(corpus_file, bm25_index)
        if rebuild or not retriever.loadIndex():
            retriever.buildIndex()
            retriever.saveIndex()
        else:
            print(f"✅ 复用已有 BM25 索引: {bm25_index}")

    if not skip_bm25plus:
        bm25plus_index = os.path.join(retrieval_dir, "bm25plus_index.pkl")
        retriever = BM25PlusRetriever(corpus_file, bm25plus_index, terms_path)
        if terms_path is not None:
            retriever.loadTermsMap()
        if rebuild or not retriever.loadIndex():
            retriever.buildIndex()
            retriever.saveIndex()
        else:
            print(f"✅ 复用已有 BM25+ 索引: {bm25plus_index}")

    if not skip_vector:
        vector_index = os.path.join(retrieval_dir, "vector_index.faiss")
        vector_embeddings = os.path.join(retrieval_dir, "vector_embeddings.npz")
        retriever = VectorRetriever(
            corpus_file,
            vector_model,
            indexFile=vector_index,
            embeddingFile=vector_embeddings,
        )
        if rebuild or not retriever.loadIndex():
            retriever.buildIndex(batchSize=batch_size)
            retriever.saveIndex()
        else:
            print(f"✅ 复用已有向量索引: {vector_index}")


def _handle_ingest(args: argparse.Namespace) -> None:
    pdf_name, book_name = _materialize_pdf(args.pdf)

    ocr_argv = [pdf_name]
    if args.ocr_start_page is not None:
        ocr_argv.append(str(args.ocr_start_page))
    _run_module_main("dataGen.pix2text_ocr", ocr_argv)

    extract_argv = [book_name]
    if args.extract_start_page is not None:
        extract_argv.append(str(args.extract_start_page))
    _run_module_main("dataGen.extract_terms_from_ocr", extract_argv)

    if not args.skip_generation:
        generate_argv = [book_name]
        if args.generate_start_page is not None:
            generate_argv.append(str(args.generate_start_page))
        _run_module_main("dataGen.data_gen", generate_argv)

    if not args.skip_index:
        _build_indexes(
            rebuild=args.rebuild_index,
            skip_bm25=args.skip_bm25,
            skip_bm25plus=args.skip_bm25plus,
            skip_vector=args.skip_vector,
            vector_model=args.vector_model,
            batch_size=args.batch_size,
        )


def _handle_build_index(args: argparse.Namespace) -> None:
    _build_indexes(
        rebuild=args.rebuild,
        skip_bm25=args.skip_bm25,
        skip_bm25plus=args.skip_bm25plus,
        skip_vector=args.skip_vector,
        vector_model=args.vector_model,
        batch_size=args.batch_size,
    )


def _handle_passthrough(module_name: str, passthrough_args: list[str]) -> None:
    _run_module_main(module_name, passthrough_args)


def _handle_serve(args: argparse.Namespace) -> None:
    if args.target == "webui":
        module_name = "generation.webui"
    elif args.target == "experiment-webui":
        module_name = "scripts.experimentWebUI"
    else:
        raise SystemExit(f"暂不支持的服务目标: {args.target}")

    passthrough = ["--port", str(args.port)]
    if args.share:
        passthrough.append("--share")
    _run_module_main(module_name, passthrough)


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
    ingest.set_defaults(handler=_handle_ingest)

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
    build_index.set_defaults(handler=_handle_build_index)

    for command, module_name, help_text in [
        ("generate-queries", "generation.generateQueries", "生成评测查询集"),
        ("build-term-mapping", "scripts.buildEvalTermMapping", "构建评测术语映射"),
        ("quick-eval", "evaluation.quickEval", "运行快速检索评测"),
        ("eval-retrieval", "evaluation.evalRetrieval", "运行正式检索评测"),
        ("rag", "scripts.runRag", "运行 RAG 问答"),
        ("experiments", "scripts.runExperiments", "运行端到端对比实验"),
        ("eval-generation", "evaluation.evalGeneration", "运行生成质量评测"),
        ("report", "scripts.generateReport", "生成最终评测报告"),
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
    stats.set_defaults(handler=lambda parsed: run_statistics())

    serve = subparsers.add_parser("serve", help="启动 WebUI")
    serve.add_argument(
        "--target",
        choices=["webui", "experiment-webui"],
        default="webui",
        help="服务目标",
    )
    serve.add_argument("--port", type=int, default=7860, help="监听端口")
    serve.add_argument("--share", action="store_true", help="生成公网分享链接")
    serve.set_defaults(handler=_handle_serve)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
