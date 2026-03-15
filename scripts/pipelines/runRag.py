"""
RAG 问答命令行入口

功能：
1. 单条查询：输入问题，输出包含来源的结构化回答
2. 批量查询：从文件读取查询列表，输出 JSONL 结果文件
3. 检索策略可切换（BM25 / 向量 / 混合）

使用方法：
    # 单条查询
    python scripts/pipelines/runRag.py --query "什么是一致收敛？"

    # 批量查询
    python scripts/pipelines/runRag.py --query-file queries.txt --output outputs/rag_results.jsonl

    # 指定检索策略
    python scripts/pipelines/runRag.py --query "什么是极限？" --strategy bm25

    # 调整参数
    python scripts/pipelines/runRag.py --query "什么是导数？" --topk 5 --temperature 0.1
"""


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
