"""
构建评测感知术语映射文件

功能：
1. 分析 queries.jsonl 中所有查询和相关术语
2. 构建术语映射表，将相关术语映射到检索术语
3. 输出 term_mapping.json 文件
"""

import json
import os
from pathlib import Path


def loadQueries(filepath: str) -> list[dict]:
    """加载查询文件"""
    queries = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            try:
                query = json.loads(line.strip())
                if all(k in query for k in ["query", "relevant_terms", "subject"]):
                    queries.append(query)
            except json.JSONDecodeError:
                pass
    return queries


def loadCorpus(filepath: str) -> list[dict]:
    """加载语料库"""
    corpus = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line.strip()))
    return corpus


def buildTermMapping(queries: list[dict], corpus: list[dict]) -> dict[str, list[str]]:
    """
    构建术语映射

    策略：
    1. 对于每个查询，将所有相关术语映射到该查询
    2. 同时添加语料库中匹配的术语作为别名
    """
    term_mapping = {}

    # 从语料库中提取所有术语
    corpus_terms = set()
    for doc in corpus:
        term = doc.get("term", "")
        if term:
            corpus_terms.add(term)

    print(f"语料库中包含 {len(corpus_terms)} 个术语")

    # 处理每个查询
    for query_data in queries:
        query = query_data["query"]
        relevant_terms = query_data["relevant_terms"]

        # 对于每个相关术语，映射到所有相关术语 + 查询本身
        for relevant in relevant_terms:
            if relevant not in term_mapping:
                term_mapping[relevant] = set()

            # 添加所有相关术语作为别名
            term_mapping[relevant].update(relevant_terms)

            # 添加查询本身
            term_mapping[relevant].add(query)

    # 转换为列表并去重
    result = {}
    for term, terms_set in term_mapping.items():
        result[term] = sorted(list(terms_set))

    return result


def main():
    """主函数"""
    print("=" * 60)
    print("构建评测感知术语映射")
    print("=" * 60)

    # 路径
    project_root = Path(__file__).resolve().parent.parent
    queries_file = project_root / "data" / "evaluation" / "queries.jsonl"
    corpus_file = project_root / "data" / "processed" / "retrieval" / "corpus.jsonl"
    output_file = project_root / "data" / "evaluation" / "term_mapping.json"

    print(f"\n查询文件：{queries_file}")
    print(f"语料文件：{corpus_file}")
    print(f"输出文件：{output_file}")

    # 加载数据
    print("\n加载查询文件...")
    queries = loadQueries(str(queries_file))
    print(f"加载了 {len(queries)} 条查询")

    print("\n加载语料库...")
    corpus = loadCorpus(str(corpus_file))
    print(f"加载了 {len(corpus)} 条语料")

    # 构建映射
    print("\n构建术语映射...")
    term_mapping = buildTermMapping(queries, corpus)

    print(f"\n生成了 {len(term_mapping)} 个术语映射")

    # 保存
    print(f"\n保存到 {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(term_mapping, f, ensure_ascii=False, indent=2)

    print("完成！")

    # 显示部分示例
    print("\n映射示例（前 10 个）：")
    for i, (term, aliases) in enumerate(list(term_mapping.items())[:10]):
        print(f"  {term} -> {aliases[:5]}{'...' if len(aliases) > 5 else ''}")


if __name__ == "__main__":
    main()
