"""数据统计格式化逻辑。"""

from typing import Any


def calculatePercentiles(
    values: list[float], percentiles: list[int] = [25, 50, 75, 90, 95, 99]
) -> dict:
    """计算百分位数。"""
    if not values:
        return {}

    sortedValues = sorted(values)
    n = len(sortedValues)

    result = {
        "min": sortedValues[0],
        "max": sortedValues[-1],
        "mean": sum(sortedValues) / n,
    }

    for p in percentiles:
        idx = int(n * p / 100)
        if idx >= n:
            idx = n - 1
        result[f"p{p}"] = sortedValues[idx]

    return result


def formatStatistics(stats: dict[str, Any]) -> dict[str, Any]:
    """格式化统计结果，使其更易读。"""
    formatted = {
        "meta": {
            "generatedAt": "2026-02-14",
            "description": "数学术语 JSON 数据统计报告",
            "version": "2.0",
        },
        "summary": stats["summary"],
        "byBook": {},
        "bySubject": dict(stats["bySubject"]),
        "fieldCoverage": {},
        "fieldLengthDistribution": {},
        "definitionsAnalysis": {},
        "termLengthDistribution": {},
        "duplicates": {
            "count": len(stats["duplicates"]),
            "details": stats["duplicates"],
        },
    }

    for book, bookStats in stats["byBook"].items():
        formatted["byBook"][book] = {
            "count": bookStats["count"],
            "subjects": dict(bookStats["subjects"]),
        }

    total = stats["summary"]["validFiles"]
    for field, fieldStats in stats["fields"].items():
        formatted["fieldCoverage"][field] = {
            "present": fieldStats["present"],
            "missing": fieldStats["missing"],
            "coverageRate": round(fieldStats["present"] / total * 100, 2)
            if total > 0
            else 0,
        }

    for field, fieldStats in stats["fields"].items():
        if fieldStats["lengths"]:
            formatted["fieldLengthDistribution"][field] = calculatePercentiles(
                fieldStats["lengths"]
            )

        if fieldStats["itemCounts"]:
            formatted["fieldLengthDistribution"][f"{field}Count"] = (
                calculatePercentiles(fieldStats["itemCounts"])
            )

    formatted["definitionsAnalysis"] = {
        "totalDefinitions": stats["definitions"]["totalCount"],
        "avgPerTerm": round(stats["definitions"]["totalCount"] / total, 2)
        if total > 0
        else 0,
        "types": dict(stats["definitions"]["types"]),
        "textLength": calculatePercentiles(stats["definitions"]["textLengths"])
        if stats["definitions"]["textLengths"]
        else {},
        "withConditions": stats["definitions"]["hasConditions"],
        "withNotation": stats["definitions"]["hasNotation"],
        "withReference": stats["definitions"]["hasReference"],
    }

    formatted["termLengthDistribution"] = (
        calculatePercentiles(stats["termLengths"]) if stats["termLengths"] else {}
    )

    return formatted
