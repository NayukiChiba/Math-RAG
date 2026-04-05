"""
构建检索语料主逻辑。

使用方法：
    python -m retrieval.corpusBuilder.builder
"""

import json
import os
from typing import Any

from core.retrieval.corpusBuilder.bridge import buildBridgeCorpusItems
from core.retrieval.corpusBuilder.io import loadJsonFile
from core.retrieval.corpusBuilder.text import extractCorpusItem
from core.utils import getFileLoader

_LOADER = getFileLoader()


def buildCorpus(chunkDir: str, outputFile: str) -> dict[str, Any]:
    """
    构建检索语料。

    Args:
        chunkDir: 术语数据目录
        outputFile: 输出 JSONL 文件路径

    Returns:
        统计信息字典
    """
    stats: dict[str, Any] = {
        "totalFiles": 0,
        "validFiles": 0,
        "skippedFiles": 0,
        "corpusItems": 0,
        "bridgeItems": 0,
        "bookStats": {},
    }

    os.makedirs(os.path.dirname(outputFile), exist_ok=True)
    corpusItems: list[dict[str, Any]] = []

    for bookName in os.listdir(chunkDir):
        bookPath = os.path.join(chunkDir, bookName)
        if not os.path.isdir(bookPath):
            continue

        print(f" 处理书籍: {bookName}")
        stats["bookStats"][bookName] = {
            "totalFiles": 0,
            "validItems": 0,
            "skippedItems": 0,
        }

        for jsonFile in [f for f in os.listdir(bookPath) if f.endswith(".json")]:
            filepath = os.path.join(bookPath, jsonFile)
            stats["totalFiles"] += 1
            stats["bookStats"][bookName]["totalFiles"] += 1

            termData = loadJsonFile(filepath)
            if termData is None:
                stats["skippedFiles"] += 1
                stats["bookStats"][bookName]["skippedItems"] += 1
                continue

            stats["validFiles"] += 1
            corpusItem = extractCorpusItem(termData, bookName)
            if corpusItem is None:
                stats["skippedFiles"] += 1
                stats["bookStats"][bookName]["skippedItems"] += 1
                continue

            corpusItems.append(corpusItem)
            stats["corpusItems"] += 1
            stats["bookStats"][bookName]["validItems"] += 1

        print(f"   生成 {stats['bookStats'][bookName]['validItems']} 条语料项")

    bridgeItems = buildBridgeCorpusItems(corpusItems)
    if bridgeItems:
        print(f" 生成桥接语料项: {len(bridgeItems)} 条")
        corpusItems.extend(bridgeItems)
        stats["bridgeItems"] = len(bridgeItems)
        stats["corpusItems"] += len(bridgeItems)

    with open(outputFile, "w", encoding="utf-8") as outFile:
        for item in corpusItems:
            outFile.write(json.dumps(item, ensure_ascii=False) + "\n")

    return stats


def validateCorpusFile(corpusFile: str) -> dict[str, Any]:
    """
    验证语料文件格式。

    Args:
        corpusFile: 语料文件路径

    Returns:
        验证结果字典
    """
    result: dict[str, Any] = {
        "valid": True,
        "totalLines": 0,
        "validLines": 0,
        "errorLines": [],
        "sampleItems": [],
    }

    try:
        requiredFields = ["doc_id", "term", "subject", "text", "source", "page"]
        for lineNum, item in enumerate(_LOADER.jsonl(corpusFile), 1):
            result["totalLines"] += 1
            missingFields = [f for f in requiredFields if f not in item]
            if missingFields:
                result["valid"] = False
                result["errorLines"].append(
                    {"line": lineNum, "error": f"缺少字段: {', '.join(missingFields)}"}
                )
            else:
                result["validLines"] += 1
            if len(result["sampleItems"]) < 3:
                result["sampleItems"].append(item)
    except Exception as e:
        result["valid"] = False
        result["error"] = str(e)

    return result
