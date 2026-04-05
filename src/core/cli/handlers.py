"""产品线各子命令的处理逻辑。"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from core import config
from core.cli.runner import run_module_main


def materialize_pdf(pdf_arg: str) -> tuple[str, str]:
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
            print(f" 已复制 PDF 到原始目录: {target}")
        else:
            print(f" 使用 raw 目录中的 PDF: {target}")
    else:
        target_name = pdf_arg if pdf_arg.lower().endswith(".pdf") else f"{pdf_arg}.pdf"
        target = raw_dir / target_name
        if not target.exists():
            raise SystemExit(f"未找到 PDF: {pdf_arg}")
        print(f" 使用 raw 目录中的 PDF: {target}")

    return target.name, target.stem


def ensure_corpus(*, rebuild: bool) -> str:
    from core.retrieval import run_build_corpus

    corpus_file = os.path.join(config.PROCESSED_DIR, "retrieval", "corpus.jsonl")
    if rebuild or not os.path.exists(corpus_file):
        run_build_corpus()
    else:
        print(f" 复用已有语料文件: {corpus_file}")
    return corpus_file


def build_indexes(
    *,
    rebuild: bool,
    skip_bm25: bool,
    skip_bm25plus: bool,
    skip_vector: bool,
    vector_model: str,
    batch_size: int,
) -> None:
    from core.retrieval import BM25PlusRetriever, BM25Retriever, VectorRetriever

    corpus_file = ensure_corpus(rebuild=rebuild)
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
            print(f" 复用已有 BM25 索引: {bm25_index}")

    if not skip_bm25plus:
        bm25plus_index = os.path.join(retrieval_dir, "bm25plus_index.pkl")
        retriever = BM25PlusRetriever(corpus_file, bm25plus_index, terms_path)
        if terms_path is not None:
            retriever.loadTermsMap()
        if rebuild or not retriever.loadIndex():
            retriever.buildIndex()
            retriever.saveIndex()
        else:
            print(f" 复用已有 BM25+ 索引: {bm25plus_index}")

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
            print(f" 复用已有向量索引: {vector_index}")


def handle_ingest(args: argparse.Namespace) -> None:
    pdf_name, book_name = materialize_pdf(args.pdf)

    ocr_argv = [pdf_name]
    if args.ocr_start_page is not None:
        ocr_argv.append(str(args.ocr_start_page))
    run_module_main("core.dataGen.pix2text_ocr", ocr_argv)

    extract_argv = [book_name]
    if args.extract_start_page is not None:
        extract_argv.append(str(args.extract_start_page))
    run_module_main("core.dataGen.extract_terms_from_ocr", extract_argv)

    if not args.skip_generation:
        generate_argv = [book_name]
        if args.generate_start_page is not None:
            generate_argv.append(str(args.generate_start_page))
        run_module_main("core.dataGen.data_gen", generate_argv)

    if not args.skip_index:
        build_indexes(
            rebuild=args.rebuild_index,
            skip_bm25=args.skip_bm25,
            skip_bm25plus=args.skip_bm25plus,
            skip_vector=args.skip_vector,
            vector_model=args.vector_model,
            batch_size=args.batch_size,
        )


def handle_build_index(args: argparse.Namespace) -> None:
    build_indexes(
        rebuild=args.rebuild,
        skip_bm25=args.skip_bm25,
        skip_bm25plus=args.skip_bm25plus,
        skip_vector=args.skip_vector,
        vector_model=args.vector_model,
        batch_size=args.batch_size,
    )


def handle_passthrough(module_name: str, passthrough_args: list[str]) -> None:
    run_module_main(module_name, passthrough_args)


def handle_serve(args: argparse.Namespace) -> None:
    module_name = "core.answerGeneration.webui"
    passthrough = ["--port", str(args.port)]
    if args.share:
        passthrough.append("--share")
    run_module_main(module_name, passthrough)
