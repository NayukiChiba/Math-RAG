"""
分析诊断报告，找出 recall 低于理论最大值的查询
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

with open(
    os.path.join(config.PROJECT_ROOT, "outputs", "diagnosis_report.json"),
    encoding="utf-8",
) as f:
    report = json.load(f)

print(f"avg_recall5: {report['avg_recall5']:.4f}")
print(f"avg_max_recall5: {report['avg_max_recall5']:.4f}")
print(f"\n{'查询':<25} {'recall5':>8} {'max_r5':>8} {'gap':>8} {'in_corpus':>10}")
print("-" * 75)

for q in report["queries"]:
    gap = q["max_recall5"] - q["recall5"]
    print(
        f"{q['query']:<25} {q['recall5']:>8.2%} {q['max_recall5']:>8.2%} {gap:>8.2%} {len(q['in_corpus']):>10}"
    )

print("\n=== 低于理论最大值的查询详情 ===")
for q in report["queries"]:
    gap = q["max_recall5"] - q["recall5"]
    if gap > 0.05:  # 差距超过 5%
        print(f"\n查询: {q['query']}")
        print(
            f"  recall5 / max_recall5: {q['recall5']:.2%} / {q['max_recall5']:.2%} (gap: {gap:.2%})"
        )
        print(f"  relevant_terms: {q['relevant_terms']}")
        print(f"  in_corpus: {q['in_corpus']}")
        print(f"  eval_terms_map: {q['eval_terms_map']}")
        print(f"  expanded_terms: {q['expanded_terms']}")
        print(f"  direct_top5: {q['direct_top5_terms']}")
        direct5_relevant = [
            t for t in q["relevant_terms"] if t in q["direct_top5_terms"]
        ]
        print(f"  直接查找Top5中的相关术语: {direct5_relevant}")
