"""
RAG 问答命令行入口

功能：
1. 单条查询：输入问题，输出包含来源的结构化回答
2. 批量查询：从文件读取查询列表，输出 JSONL 结果文件
3. 检索策略可切换（BM25 / 向量 / 混合）

使用方法：
    # 单条查询
    python scripts/runRag.py --query "什么是一致收敛？"

    # 批量查询
    python scripts/runRag.py --query-file queries.txt --output outputs/rag_results.jsonl

    # 指定检索策略
    python scripts/runRag.py --query "什么是极限？" --strategy bm25

    # 调整参数
    python scripts/runRag.py --query "什么是导数？" --topk 5 --temperature 0.1
"""

import argparse
import json
import os
import sys
from pathlib import Path

# 路径调整
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from generation.ragPipeline import RagPipeline, loadQueries, saveResults


def printResult(result: dict) -> None:
    """
    打印单条查询结果

    Args:
        result: 查询结果字典
    """
    print("\n" + "=" * 80)
    print(f"查询: {result['query']}")
    print("=" * 80)

    # 检索结果
    terms = result.get("retrieved_terms", [])
    if terms:
        print(f"\n检索到 {len(terms)} 个相关术语:")
        for t in terms:
            source = t.get("source", "")
            page = t.get("page")
            sourceStr = f"【{source} 第{page}页】" if source and page else ""
            print(f"  - {t['term']} ({t.get('subject', '')}) {sourceStr}")
            print(f"    分数: {t.get('score', 0):.4f}")
    else:
        print("\n未检索到相关术语")

    # 回答
    print("\n回答:")
    print("-" * 40)
    print(result.get("answer", "（无回答）"))
    print("-" * 40)

    # 来源
    sources = result.get("sources", [])
    if sources:
        print("\n来源:")
        for s in sources:
            if s.get("source"):
                pageStr = f" 第{s['page']}页" if s.get("page") else ""
                print(f"  - {s['source']}{pageStr}")

    # 耗时
    latency = result.get("latency", {})
    print("\n⏱耗时:")
    print(f"  检索: {latency.get('retrieval_ms', 0)} ms")
    print(f"  生成: {latency.get('generation_ms', 0)} ms")
    print(f"  总计: {latency.get('total_ms', 0)} ms")


def main():
    """主函数"""
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

    # 查询参数
    parser.add_argument("--query", "-q", type=str, help="单条查询文本")
    parser.add_argument(
        "--query-file", "-f", type=str, help="批量查询文件路径（每行一个查询）"
    )

    # 检索参数
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

    # 生成参数
    parser.add_argument(
        "--temperature", "-t", type=float, default=None, help="采样温度"
    )
    parser.add_argument("--top-p", type=float, default=None, help="top-p 采样参数")
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="最大生成 token 数"
    )

    # 输出参数
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

    args = parser.parse_args()

    # 检查参数
    if not args.query and not args.query_file:
        print("请提供查询参数：")
        print("  --query '你的问题'  # 单条查询")
        print("  --query-file queries.txt  # 批量查询")
        parser.print_help()
        sys.exit(1)

    # 打印配置
    if not args.quiet:
        print("=" * 80)
        print("RAG 问答系统")
        print("=" * 80)
        print(f"检索策略: {args.strategy}")
        print(f"TopK: {args.topk}")
        if args.strategy == "hybrid":
            print(f"权重: BM25={args.alpha}, 向量={args.beta}")
        print()

    # 初始化 Pipeline
    pipeline = RagPipeline(
        strategy=args.strategy,
        topK=args.topk,
        hybridAlpha=args.alpha,
        hybridBeta=args.beta,
    )

    # 执行查询
    if args.query:
        # 单条查询
        result = pipeline.query(
            args.query,
            temperature=args.temperature,
            topP=args.top_p,
            maxNewTokens=args.max_tokens,
        )

        if args.quiet:
            print(json.dumps(result, ensure_ascii=False))
        else:
            printResult(result)

        # 保存结果
        if args.output:
            saveResults([result], args.output)

    elif args.query_file:
        # 批量查询
        if not args.quiet:
            print(f"加载查询文件: {args.query_file}")

        queries = loadQueries(args.query_file)

        if not args.quiet:
            print(f"已加载 {len(queries)} 个查询\n")

        results = pipeline.batchQuery(
            queries,
            temperature=args.temperature,
            topP=args.top_p,
            maxNewTokens=args.max_tokens,
            showProgress=not args.quiet,
        )

        # 打印结果
        if not args.quiet:
            for result in results:
                printResult(result)

        # 保存结果
        outputFile = args.output or os.path.join(
            config.PROJECT_ROOT, "outputs", "rag_results.jsonl"
        )
        saveResults(results, outputFile)

        # 统计信息
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
