"""
自动生成评测查询数据

功能：
1. 从术语库中智能采样高频术语
2. 自动提取相关术语（aliases + related_terms）
3. 分学科生成 queries.jsonl
4. 与现有数据合并，去重

使用方法：
    # 默认：按固定数量生成（数学分析35，高等代数20，概率论20）
    python evaluation/generateQueries.py

    # 生成所有符合条件的术语
    python evaluation/generateQueries.py --all

    # 按比例采样（如50%）
    python evaluation/generateQueries.py --ratio 0.5

    # 指定每个学科的数量
    python evaluation/generateQueries.py --num-ma 50 --num-gd 30 --num-gl 30

    # 调整质量阈值（最少相关术语数）
    python evaluation/generateQueries.py --min-related 2
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# 路径调整
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from utils import getFileLoader

_LOADER = getFileLoader()


def loadJsonFile(filepath: str) -> dict[str, Any]:
    """加载 JSON 文件"""
    try:
        return _LOADER.json(filepath)
    except Exception as e:
        print(f"❌ 加载文件失败: {filepath}, 错误: {e}")
        return None


def loadAllTerms(chunkDir: str) -> list[dict[str, Any]]:
    """
    加载所有术语数据

    Args:
        chunkDir: 术语数据目录

    Returns:
        术语列表
    """
    terms = []

    print("📚 加载术语库...")
    # 排序保证加载顺序固定，确保可复现性
    for bookName in sorted(os.listdir(chunkDir)):
        bookPath = os.path.join(chunkDir, bookName)

        if not os.path.isdir(bookPath):
            continue

        # 排序保证文件加载顺序固定
        jsonFiles = sorted([f for f in os.listdir(bookPath) if f.endswith(".json")])
        for jsonFile in jsonFiles:
            filepath = os.path.join(bookPath, jsonFile)
            data = loadJsonFile(filepath)

            if data and "term" in data and "subject" in data:
                terms.append(
                    {
                        "term": data["term"],
                        "aliases": data.get("aliases", []),
                        "related_terms": data.get("related_terms", []),
                        "subject": data["subject"],
                        "book": bookName,
                    }
                )

    print(f"✅ 加载 {len(terms)} 个术语")
    return terms


def normalizeSubject(subject: str) -> str:
    """标准化学科名称"""
    if "数学分析" in subject:
        return "数学分析"
    elif "高等代数" in subject:
        return "高等代数"
    elif "概率" in subject or "统计" in subject:
        return "概率论"
    else:
        return subject


def generateQueries(
    terms: list[dict[str, Any]],
    numPerSubject: dict[str, int] = None,
    minRelatedTerms: int = 1,
    useAll: bool = False,
    sampleRatio: float = None,
) -> list[dict[str, Any]]:
    """
    生成查询数据

    Args:
        terms: 术语列表
        numPerSubject: 每个学科生成数量，默认 {'数学分析': 30, '高等代数': 20, '概率论': 20}
        minRelatedTerms: 最少相关术语数量
        useAll: 是否使用所有符合条件的术语（忽略 numPerSubject）
        sampleRatio: 采样比例 (0-1)，如 0.5 表示使用 50% 的术语

    Returns:
        查询列表
    """
    if numPerSubject is None and not useAll and sampleRatio is None:
        numPerSubject = {"数学分析": 30, "高等代数": 20, "概率论": 20}

    # 按学科分组
    termsBySubject = defaultdict(list)
    for term in terms:
        subject = normalizeSubject(term["subject"])
        termsBySubject[subject].append(term)

    queries = []

    for subject, subjectTerms in termsBySubject.items():
        print(f"\n📊 处理学科: {subject} (共 {len(subjectTerms)} 个术语)")

        # 过滤出有相关术语的高质量术语
        candidateTerms = []
        for term in subjectTerms:
            relatedCount = len(term["aliases"]) + len(term["related_terms"])
            if relatedCount >= minRelatedTerms:
                candidateTerms.append(
                    {
                        "term": term,
                        "score": relatedCount,  # 按相关术语数量排序
                    }
                )

        # 按质量排序
        candidateTerms.sort(key=lambda x: x["score"], reverse=True)

        print(f"  - 符合条件的术语: {len(candidateTerms)} 个")

        # 选择术语策略
        if useAll:
            # 使用所有符合条件的术语
            selectedTerms = [c["term"] for c in candidateTerms]
            print("  - 生成策略: 使用全部符合条件的术语")
        elif sampleRatio is not None:
            # 按比例采样
            targetNum = int(len(candidateTerms) * sampleRatio)
            # 80% 高质量 + 20% 随机
            numHigh = int(targetNum * 0.8)
            numRandom = targetNum - numHigh
            highQuality = [c["term"] for c in candidateTerms[:numHigh]]
            remainingTerms = [c["term"] for c in candidateTerms[numHigh:]]
            randomTerms = random.sample(
                remainingTerms, min(numRandom, len(remainingTerms))
            )
            selectedTerms = highQuality + randomTerms
            print(
                f"  - 生成策略: 按比例采样 {sampleRatio * 100:.0f}% = {len(selectedTerms)} 条"
            )
        else:
            # 按固定数量采样
            targetNum = numPerSubject.get(subject, 20)
            numHigh = int(targetNum * 0.8)
            numRandom = targetNum - numHigh
            highQuality = [c["term"] for c in candidateTerms[:numHigh]]
            remainingTerms = [c["term"] for c in candidateTerms[numHigh:]]
            randomTerms = random.sample(
                remainingTerms, min(numRandom, len(remainingTerms))
            )
            selectedTerms = highQuality + randomTerms
            print(f"  - 生成策略: 固定数量 {targetNum} 条")

        # 生成查询
        for term in selectedTerms:
            # 构建 relevant_terms：term + aliases + 部分 related_terms
            relevantTerms = [term["term"]]
            relevantTerms.extend(term["aliases"][:3])  # 最多3个别名

            # 从 related_terms 中选择相关性高的
            if term["related_terms"]:
                # 优先选择包含查询词的相关术语
                relatedWithQuery = [
                    rt for rt in term["related_terms"] if term["term"][:2] in rt
                ]
                relatedOthers = [
                    rt for rt in term["related_terms"] if term["term"][:2] not in rt
                ]

                selectedRelated = relatedWithQuery[:2] + relatedOthers[:1]
                relevantTerms.extend(selectedRelated)

            # 去重并保持顺序
            uniqueRelevant = []
            seen = set()
            for rt in relevantTerms:
                if rt not in seen:
                    uniqueRelevant.append(rt)
                    seen.add(rt)

            query = {
                "query": term["term"],
                "relevant_terms": uniqueRelevant,
                "subject": subject,
            }
            queries.append(query)

        print(f"✅ 生成 {len(selectedTerms)} 条查询")

    return queries


def loadExistingQueries(filepath: str) -> list[dict[str, Any]]:
    """加载现有查询数据"""
    if not os.path.exists(filepath):
        return []

    try:
        queries = _LOADER.jsonl(filepath)
        print(f"📋 加载现有查询: {len(queries)} 条")
    except Exception as e:
        print(f"❌ 加载现有查询失败: {e}")
        return []

    return queries


def mergeQueries(
    existing: list[dict[str, Any]], generated: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    合并查询，去重

    优先保留人工标注的数据
    """
    # 使用 query 作为唯一键
    seenQueries = {q["query"]: q for q in existing}

    merged = list(seenQueries.values())
    newCount = 0

    # 遍历生成的查询，去重（包括内部重复）
    for gq in generated:
        if gq["query"] not in seenQueries:
            merged.append(gq)
            seenQueries[gq["query"]] = gq  # 更新已见集合，避免内部重复
            newCount += 1

    print("\n📊 合并结果:")
    print(f"  - 现有: {len(existing)} 条")
    print(f"  - 新增: {newCount} 条")
    print(f"  - 总计: {len(merged)} 条")

    return merged


def saveQueries(queries: list[dict[str, Any]], filepath: str):
    """保存查询数据到 JSONL"""
    # 按学科分组统计
    bySubject = defaultdict(int)
    for q in queries:
        bySubject[q["subject"]] += 1

    print("\n📊 学科分布:")
    for subject, count in sorted(bySubject.items()):
        print(f"  - {subject}: {count} 条")

    # 保存
    with open(filepath, "w", encoding="utf-8") as f:
        for query in queries:
            f.write(json.dumps(query, ensure_ascii=False) + "\n")

    print(f"\n✅ 保存到: {filepath}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="自动生成评测查询数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 默认：按固定数量生成
  python evaluation/generateQueries.py
  
  # 生成所有符合条件的术语
  python evaluation/generateQueries.py --all
  
  # 按50%比例采样
  python evaluation/generateQueries.py --ratio 0.5
  
  # 自定义各学科数量
  python evaluation/generateQueries.py --num-ma 50 --num-gd 30 --num-gl 30
        """,
    )

    parser.add_argument(
        "--all", action="store_true", help="使用所有符合条件的术语（忽略数量限制）"
    )

    parser.add_argument(
        "--ratio", type=float, help="采样比例 (0-1)，如 0.5 表示使用 50%% 的术语"
    )

    parser.add_argument(
        "--num-ma", type=int, default=35, help="数学分析生成数量（默认35）"
    )

    parser.add_argument(
        "--num-gd", type=int, default=20, help="高等代数生成数量（默认20）"
    )

    parser.add_argument(
        "--num-gl", type=int, default=20, help="概率论生成数量（默认20）"
    )

    parser.add_argument(
        "--min-related", type=int, default=1, help="最少相关术语数量阈值（默认1）"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（默认：data/evaluation/queries.jsonl）",
    )

    parser.add_argument(
        "--no-merge", action="store_true", help="不与现有数据合并，直接覆盖"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🤖 自动生成评测查询数据")
    print("=" * 60)

    # 配置
    chunkDir = config.CHUNK_DIR
    outputFile = args.output or os.path.join(config.EVALUATION_DIR, "queries.jsonl")

    # 设置随机种子保证可复现
    random.seed(42)

    # Step 1: 加载所有术语
    terms = loadAllTerms(chunkDir)
    if not terms:
        print("❌ 未找到术语数据")
        return

    # Step 2: 生成查询
    print("\n" + "=" * 60)
    print("🔧 生成新查询")
    print("=" * 60)

    if args.all:
        print("📌 模式: 使用所有符合条件的术语")
        generatedQueries = generateQueries(
            terms, minRelatedTerms=args.min_related, useAll=True
        )
    elif args.ratio is not None:
        print(f"📌 模式: 按比例采样 ({args.ratio * 100:.0f}%)")
        generatedQueries = generateQueries(
            terms, minRelatedTerms=args.min_related, sampleRatio=args.ratio
        )
    else:
        print("📌 模式: 固定数量")
        print(f"  - 数学分析: {args.num_ma} 条")
        print(f"  - 高等代数: {args.num_gd} 条")
        print(f"  - 概率论: {args.num_gl} 条")
        generatedQueries = generateQueries(
            terms,
            numPerSubject={
                "数学分析": args.num_ma,
                "高等代数": args.num_gd,
                "概率论": args.num_gl,
            },
            minRelatedTerms=args.min_related,
        )

    # Step 3: 加载现有查询
    if not args.no_merge:
        print("\n" + "=" * 60)
        print("📋 合并现有查询")
        print("=" * 60)

        existingQueries = loadExistingQueries(outputFile)
        mergedQueries = mergeQueries(existingQueries, generatedQueries)
    else:
        print("\n" + "=" * 60)
        print("⚠️  直接覆盖模式（不保留现有数据）")
        print("=" * 60)
        mergedQueries = generatedQueries

    # Step 4: 保存
    print("\n" + "=" * 60)
    print("💾 保存结果")
    print("=" * 60)

    saveQueries(mergedQueries, outputFile)

    print("\n" + "=" * 60)
    print("✅ 完成！")
    print("=" * 60)
    print(f"总查询数: {len(mergedQueries)}")
    if not args.no_merge:
        print(f"建议人工审核: {outputFile}")


if __name__ == "__main__":
    main()
