"""检索评测指标模块。

本模块提供 Recall、MRR、AP/MAP、DCG/IDCG/nDCG 等通用指标函数，
用于评估排序结果的相关性与排名质量。

约定：
1. `results` 为检索结果列表，每项至少包含 `term` 字段。
2. `relevant_terms` 表示黄金相关术语集合（有序时可体现相关性等级）。
3. 若无相关术语，指标通常返回 0.0，避免除零错误。
"""

from __future__ import annotations

import math
from typing import Any


def calculateRecallAtK(
    results: list[dict[str, Any]], relevant_terms: list[str], k: int
) -> float:
    """计算 Recall@K。

    含义：在前 K 个检索结果里，覆盖了多少比例的相关术语。
    公式：
        Recall@K = 命中的相关术语数 / 相关术语总数
    """
    if not relevant_terms:
        return 0.0
    # 用集合做 membership 判断，降低重复查找成本。
    topk_terms = {item.get("term") for item in results[:k]}
    found = sum(1 for term in relevant_terms if term in topk_terms)
    return found / len(relevant_terms)


def calculateMRR(results: list[dict[str, Any]], relevant_terms: list[str]) -> float:
    """计算 MRR（Mean Reciprocal Rank）。

    只关注“第一个命中的相关结果”位置：
    - 若首个相关结果排名为 r，则该查询得分为 1/r。
    - 若没有命中，则得分为 0。
    """
    for rank, result in enumerate(results, 1):
        if result.get("term") in relevant_terms:
            return 1.0 / rank
    return 0.0


def calculateAP(results: list[dict[str, Any]], relevant_terms: list[str]) -> float:
    """计算 AP（Average Precision）。

    对每个命中的相关结果，计算其命中时刻的 Precision，
    最后对“相关术语总数”做平均。
    """
    if not relevant_terms:
        return 0.0

    precision_sum = 0.0
    hit_count = 0
    for rank, result in enumerate(results, 1):
        if result.get("term") in relevant_terms:
            hit_count += 1
            precision_sum += hit_count / rank
    return precision_sum / len(relevant_terms) if hit_count else 0.0


def calculateMAP(results: list[dict[str, Any]], relevant_terms: list[str]) -> float:
    """计算 MAP 的单查询版本。

    在当前实现中与 AP 等价；若后续支持批量查询，MAP 应是各查询 AP 的平均。
    """
    return calculateAP(results, relevant_terms)


def calculateDCG(
    results: list[dict[str, Any]], relevant_terms: list[str], k: int
) -> float:
    """计算 DCG@K（Discounted Cumulative Gain）。

    相关结果出现得越靠前，折损越小，贡献越大。
    此处将 `relevant_terms` 的顺序映射为相关性分值（前者分值更高）。
    """
    dcg = 0.0
    for idx, result in enumerate(results[:k], 1):
        term = result.get("term")
        relevance = (
            len(relevant_terms) - relevant_terms.index(term)
            if term in relevant_terms
            else 0
        )
        dcg += relevance / math.log2(idx + 1)
    return dcg


def calculateIDCG(relevant_terms: list[str], k: int) -> float:
    """计算理想排序下的 IDCG@K。

    这是 nDCG 归一化的分母，表示“理论最优排序”可获得的 DCG。
    """
    idcg = 0.0
    for i in range(min(k, len(relevant_terms))):
        relevance = len(relevant_terms) - i
        idcg += relevance / math.log2(i + 2)
    return idcg


def calculateNDCG(
    results: list[dict[str, Any]], relevant_terms: list[str], k: int
) -> float:
    """计算 nDCG@K（Normalized DCG）。

    公式：
        nDCG@K = DCG@K / IDCG@K

    返回范围通常在 [0, 1]，越接近 1 表示排序越接近理想。
    """
    idcg = calculateIDCG(relevant_terms, k)
    if idcg <= 0:
        return 0.0
    return calculateDCG(results, relevant_terms, k) / idcg
