"""数据统计构建逻辑。"""

import os
from collections import Counter, defaultdict
from typing import Any

try:
    from research.dataStat.loaders import loadJsonFile
except ModuleNotFoundError:
    from loaders import loadJsonFile


def calculateFieldStats(data: dict[str, Any], fieldName: str, fieldStats: dict) -> None:
    """计算单个字段的统计信息。"""
    if fieldName in data and data[fieldName]:
        fieldStats["present"] += 1
        value = data[fieldName]

        if isinstance(value, str):
            fieldStats["lengths"].append(len(value))
        elif isinstance(value, list):
            fieldStats["lengths"].append(len(value))
            if value and isinstance(value[0], str):
                for item in value:
                    fieldStats["itemLengths"].append(len(item))
            elif value and isinstance(value[0], dict):
                fieldStats["itemCounts"].append(len(value))
    else:
        fieldStats["missing"] += 1


def analyzeDefinitions(definitions: list[dict]) -> dict:
    """分析 definitions 字段的详细信息。"""
    if not definitions:
        return {}

    stats = {
        "count": len(definitions),
        "types": Counter(),
        "textLengths": [],
        "hasConditions": 0,
        "hasNotation": 0,
        "hasReference": 0,
    }

    for defn in definitions:
        if isinstance(defn, dict):
            stats["types"][defn.get("type", "unknown")] += 1
            if defn.get("text"):
                stats["textLengths"].append(len(defn["text"]))
            if defn.get("conditions"):
                stats["hasConditions"] += 1
            if defn.get("notation"):
                stats["hasNotation"] += 1
            if defn.get("reference"):
                stats["hasReference"] += 1

    return stats


def buildStatistics(chunkDir: str) -> dict[str, Any]:
    """构建完整的统计信息。"""
    stats = {
        "summary": {
            "totalFiles": 0,
            "validFiles": 0,
            "invalidFiles": 0,
            "totalTerms": 0,
        },
        "byBook": defaultdict(
            lambda: {
                "count": 0,
                "subjects": Counter(),
            }
        ),
        "bySubject": Counter(),
        "fields": {},
        "definitions": {
            "totalCount": 0,
            "types": Counter(),
            "textLengths": [],
            "hasConditions": 0,
            "hasNotation": 0,
            "hasReference": 0,
        },
        "termLengths": [],
        "duplicates": defaultdict(list),
    }

    fieldsToCheck = [
        "id",
        "term",
        "aliases",
        "sense_id",
        "subject",
        "definitions",
        "notation",
        "formula",
        "usage",
        "applications",
        "disambiguation",
        "related_terms",
        "sources",
        "search_keys",
        "lang",
        "confidence",
    ]

    for field in fieldsToCheck:
        stats["fields"][field] = {
            "present": 0,
            "missing": 0,
            "lengths": [],
            "itemLengths": [],
            "itemCounts": [],
        }

    if not os.path.exists(chunkDir):
        print(f" 目录不存在: {chunkDir}")
        return stats

    for bookName in os.listdir(chunkDir):
        bookPath = os.path.join(chunkDir, bookName)
        if not os.path.isdir(bookPath):
            continue

        print(f" 处理书籍: {bookName}")
        jsonFiles = [f for f in os.listdir(bookPath) if f.endswith(".json")]

        for jsonFile in jsonFiles:
            filepath = os.path.join(bookPath, jsonFile)
            stats["summary"]["totalFiles"] += 1

            data = loadJsonFile(filepath)
            if data is None:
                stats["summary"]["invalidFiles"] += 1
                continue

            stats["summary"]["validFiles"] += 1
            stats["summary"]["totalTerms"] += 1
            stats["byBook"][bookName]["count"] += 1

            subject = data.get("subject", "unknown")
            stats["bySubject"][subject] += 1
            stats["byBook"][bookName]["subjects"][subject] += 1

            term = data.get("term", "")
            if term:
                stats["termLengths"].append(len(term))
                stats["duplicates"][term].append(
                    {"book": bookName, "file": jsonFile, "subject": subject}
                )

            for field in fieldsToCheck:
                calculateFieldStats(data, field, stats["fields"][field])

            definitions = data.get("definitions", [])
            if definitions:
                defStats = analyzeDefinitions(definitions)
                if defStats:
                    stats["definitions"]["totalCount"] += defStats["count"]
                    stats["definitions"]["types"].update(defStats["types"])
                    stats["definitions"]["textLengths"].extend(defStats["textLengths"])
                    stats["definitions"]["hasConditions"] += defStats["hasConditions"]
                    stats["definitions"]["hasNotation"] += defStats["hasNotation"]
                    stats["definitions"]["hasReference"] += defStats["hasReference"]

    stats["duplicates"] = {
        term: locations
        for term, locations in stats["duplicates"].items()
        if len(locations) > 1
    }

    return stats
