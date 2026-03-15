"""Console report printers for generation evaluation."""

from __future__ import annotations

from typing import Any


def printSummary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📊 生成质量评测结果")
    print("=" * 60)

    print(f"\n总查询数: {summary['total_queries']}")

    print("\n📈 术语命中率:")
    print(f"  平均: {summary['term_hit_rate']['mean']:.4f}")
    print(f"  最小: {summary['term_hit_rate']['min']:.4f}")
    print(f"  最大: {summary['term_hit_rate']['max']:.4f}")

    print("\n📚 来源引用率:")
    print(f"  平均: {summary['source_citation_rate']['mean']:.4f}")
    print(f"  最小: {summary['source_citation_rate']['min']:.4f}")
    print(f"  最大: {summary['source_citation_rate']['max']:.4f}")

    print("\n✅ 回答有效率:")
    print(f"  有效率: {summary['answer_valid_rate']:.4f}")
    print(f"  有效回答: {summary['valid_answers']}")
    print(f"  无效回答: {summary['invalid_answers']}")

    if "bleu" in summary:
        print("\n📝 BLEU 分数:")
        print(f"  平均: {summary['bleu']['mean']:.4f}")
        print(f"  最小: {summary['bleu']['min']:.4f}")
        print(f"  最大: {summary['bleu']['max']:.4f}")

    if "rouge" in summary:
        print("\n📝 ROUGE 分数:")
        print(f"  ROUGE-1: {summary['rouge']['rouge1_mean']:.4f}")
        print(f"  ROUGE-2: {summary['rouge']['rouge2_mean']:.4f}")
        print(f"  ROUGE-L: {summary['rouge']['rougeL_mean']:.4f}")


def printExamples(examples: dict[str, list[dict[str, Any]]]) -> None:
    print("\n" + "=" * 60)
    print("🏆 最佳示例")
    print("=" * 60)

    for i, ex in enumerate(examples["best"], 1):
        print(f"\n[{i}] 查询: {ex['query']}")
        print(f"    术语命中率: {ex['term_hit_rate']:.4f}")
        print(f"    来源引用率: {ex['source_citation_rate']:.4f}")
        print(f"    综合分数: {ex['composite_score']:.4f}")
        print(f"    回答: {ex['answer'][:200]}...")

    print("\n" + "=" * 60)
    print("⚠️ 最差示例")
    print("=" * 60)

    for i, ex in enumerate(examples["worst"], 1):
        print(f"\n[{i}] 查询: {ex['query']}")
        print(f"    术语命中率: {ex['term_hit_rate']:.4f}")
        print(f"    来源引用率: {ex['source_citation_rate']:.4f}")
        print(f"    综合分数: {ex['composite_score']:.4f}")
        if ex.get("validity_reason"):
            print(f"    无效原因: {ex['validity_reason']}")
        print(f"    回答: {ex['answer'][:200]}...")
