"""查询生成策略模块。

该模块负责把术语库数据转换为评测查询集，核心包含：
1. 学科标准化。
2. 候选术语筛选与采样。
3. relevant_terms 构建与去重。

设计关注点：
- 可复现：配合上层固定随机种子可复现实验。
- 可控：支持固定数量、比例采样、全量三种模式。
- 可读：将复杂流程拆成小函数，便于单测与维护。
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def normalizeSubject(subject: str) -> str:
    """标准化学科名称。

    目的：把同一学科的不同描述归并为统一标签，避免统计分桶碎片化。
    """
    if "数学分析" in subject:
        return "数学分析"
    if "高等代数" in subject:
        return "高等代数"
    if "概率" in subject or "统计" in subject:
        return "概率论"
    return subject


def _selectTerms(
    candidate_terms: list[dict[str, Any]],
    use_all: bool,
    sample_ratio: float | None,
    target_num: int,
) -> list[dict[str, Any]]:
    """从候选术语中选择本轮使用的术语集合。

    选择策略：
    1. 全量模式：直接返回全部候选。
    2. 比例/定额模式：80% 高质量 + 20% 随机补充。

    这样可以兼顾：
    - 样本质量（优先高相关术语）；
    - 样本多样性（避免过于集中在头部术语）。
    """
    if use_all:
        print("  - 生成策略: 使用全部符合条件的术语")
        return [item["term"] for item in candidate_terms]

    if sample_ratio is not None:
        target_num = int(len(candidate_terms) * sample_ratio)
        print(f"  - 生成策略: 按比例采样 {sample_ratio * 100:.0f}% = {target_num} 条")
    else:
        print(f"  - 生成策略: 固定数量 {target_num} 条")

    num_high = int(target_num * 0.8)
    num_random = target_num - num_high
    high_quality = [item["term"] for item in candidate_terms[:num_high]]
    remaining = [item["term"] for item in candidate_terms[num_high:]]
    random_terms = random.sample(remaining, min(num_random, len(remaining)))
    return high_quality + random_terms


def _buildRelevantTerms(term_row: dict[str, Any]) -> list[str]:
    """构建单条 query 的 relevant_terms。

    组合来源：
    1. 主术语 `term`（必须包含）。
    2. 前 3 个别名 `aliases`。
    3. 最多 3 个 related_terms（优先与 query 词面更接近的项）。

    最后会做顺序保持去重，确保评测集合稳定且无重复标注。
    """
    relevant_terms = [term_row["term"]]
    relevant_terms.extend(term_row["aliases"][:3])

    if term_row["related_terms"]:
        related_with_query = [
            text for text in term_row["related_terms"] if term_row["term"][:2] in text
        ]
        related_others = [
            text
            for text in term_row["related_terms"]
            if term_row["term"][:2] not in text
        ]
        relevant_terms.extend(related_with_query[:2] + related_others[:1])

    deduplicated: list[str] = []
    seen: set[str] = set()
    for value in relevant_terms:
        if value not in seen:
            deduplicated.append(value)
            seen.add(value)
    return deduplicated


def generateQueries(
    terms: list[dict[str, Any]],
    num_per_subject: dict[str, int] | None = None,
    min_related_terms: int = 1,
    use_all: bool = False,
    sample_ratio: float | None = None,
) -> list[dict[str, Any]]:
    """主流程：根据术语库生成评测查询。

    参数：
    - terms: 术语记录列表。
    - num_per_subject: 每学科固定数量；仅在固定数量模式生效。
    - min_related_terms: 最低相关术语阈值，用于过滤低质量项。
    - use_all: 是否启用全量模式。
    - sample_ratio: 比例采样值（0~1）。

    返回：
    - 标准化后的 query 列表，元素结构为：
      {"query": str, "relevant_terms": list[str], "subject": str}
    """
    if num_per_subject is None and not use_all and sample_ratio is None:
        num_per_subject = {"数学分析": 30, "高等代数": 20, "概率论": 20}

    terms_by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for term in terms:
        terms_by_subject[normalizeSubject(term["subject"])].append(term)

    queries: list[dict[str, Any]] = []
    for subject, subject_terms in terms_by_subject.items():
        print(f"\n 处理学科: {subject} (共 {len(subject_terms)} 个术语)")

        # 依据 aliases + related_terms 的数量过滤低质量样本。
        candidate_terms: list[dict[str, Any]] = []
        for term in subject_terms:
            related_count = len(term["aliases"]) + len(term["related_terms"])
            if related_count >= min_related_terms:
                candidate_terms.append({"term": term, "score": related_count})

        # 分值越高表示可构建的相关词越多，默认优先。
        candidate_terms.sort(key=lambda item: item["score"], reverse=True)
        print(f"  - 符合条件的术语: {len(candidate_terms)} 个")

        target_num = (num_per_subject or {}).get(subject, 20)
        selected_terms = _selectTerms(
            candidate_terms, use_all, sample_ratio, target_num
        )

        for term_row in selected_terms:
            queries.append(
                {
                    "query": term_row["term"],
                    "relevant_terms": _buildRelevantTerms(term_row),
                    "subject": subject,
                }
            )

        print(f" 生成 {len(selected_terms)} 条查询")

    return queries
