"""RAG 问答入口。"""

from __future__ import annotations

import argparse
import json
import sys

from runners.pipelines import runRag as rag


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="RAG 问答命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单条查询
  python scripts/runRag.py --query "什么是一致收敛？"

  # 批量查询
  python scripts/runRag.py --query-file queries.txt

  # 使用 BM25 检索
  python scripts/runRag.py --query "什么是极限？" --strategy bm25

  # 使用向量检索
  python scripts/runRag.py --query "什么是极限？" --strategy vector
        """,
    )

    parser.add_argument("--query", "-q", type=str, help="单条查询文本")
    parser.add_argument(
        "--query-file", "-f", type=str, help="批量查询文件路径（每行一个查询）"
    )

    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        default="hybrid",
        choices=["bm25", "vector", "hybrid"],
        help="检索策略（默认: hybrid）",
    )
    parser.add_argument(
        "--topk", "-k", type=int, default=5, help="检索返回的结果数量（默认: 5）"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="混合检索 BM25 权重（默认: 0.5）"
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, help="混合检索向量权重（默认: 0.5）"
    )

    parser.add_argument(
        "--temperature", "-t", type=float, default=None, help="采样温度"
    )
    parser.add_argument("--top-p", type=float, default=None, help="top-p 采样参数")
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="最大生成 token 数"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出文件路径（JSONL 格式）",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="静默模式（仅输出结果，不打印详情）"
    )

    args = parser.parse_args(argv)

    if not args.query and not args.query_file:
        print("请提供查询参数：")
        print("  --query '你的问题'  # 单条查询")
        print("  --query-file queries.txt  # 批量查询")
        parser.print_help()
        sys.exit(1)

    if not args.quiet:
        print("=" * 80)
        print("RAG 问答系统")
        print("=" * 80)
        print(f"检索策略: {args.strategy}")
        print(f"TopK: {args.topk}")
        if args.strategy == "hybrid":
            print(f"权重: BM25={args.alpha}, 向量={args.beta}")
        print()

    pipeline = rag.RagPipeline(
        strategy=args.strategy,
        topK=args.topk,
        hybridAlpha=args.alpha,
        hybridBeta=args.beta,
    )

    if args.query:
        result = pipeline.query(
            args.query,
            temperature=args.temperature,
            topP=args.top_p,
            maxNewTokens=args.max_tokens,
        )

        if args.quiet:
            print(json.dumps(result, ensure_ascii=False))
        else:
            rag.printResult(result)

        if args.output:
            rag.saveResults([result], args.output)

    elif args.query_file:
        if not args.quiet:
            print(f"加载查询文件: {args.query_file}")

        queries = rag.loadQueries(args.query_file)

        if not args.quiet:
            print(f"已加载 {len(queries)} 个查询\n")

        results = pipeline.batchQuery(
            queries,
            temperature=args.temperature,
            topP=args.top_p,
            maxNewTokens=args.max_tokens,
            showProgress=not args.quiet,
        )

        if not args.quiet:
            for result in results:
                rag.printResult(result)

        outputFile = args.output or rag.config.RAG_RESULTS_FILE
        rag.saveResults(results, outputFile)

        if not args.quiet:
            print("\n" + "=" * 80)
            print("统计信息")
            print("=" * 80)
            if results:
                totalRetrieval = sum(r["latency"]["retrieval_ms"] for r in results)
                totalGeneration = sum(r["latency"]["generation_ms"] for r in results)
                totalTime = sum(r["latency"]["total_ms"] for r in results)
                print(f"  查询数量: {len(results)}")
                print(f"  总检索耗时: {totalRetrieval} ms")
                print(f"  总生成耗时: {totalGeneration} ms")
                print(f"  总耗时: {totalTime} ms")
                print(f"  平均每条: {totalTime // len(results)} ms")
            else:
                print("  无查询结果")


if __name__ == "__main__":
    main()
