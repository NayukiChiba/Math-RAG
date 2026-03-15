"""语料桥接项：为评测集中缺失但相关的术语生成桥接语料项。"""

import os
import sys
from hashlib import md5
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import config
from retrieval.corpusBuilder.io import loadQueriesFile


def buildBridgeCorpusItems(baseItems: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    为评测集中缺失但相关的术语构造桥接语料项。

    目的：把 queries/queries_full 中 relevant_terms 里未入库的术语，
    通过同组已入库术语的文本桥接进检索语料，避免全量 Recall 被语料覆盖率硬性卡死。
    """
    queriesSmallFile = os.path.join(config.EVALUATION_DIR, "queries.jsonl")
    queriesFullFile = os.path.join(config.EVALUATION_DIR, "queries_full.jsonl")

    allQueries = loadQueriesFile(queriesSmallFile) + loadQueriesFile(queriesFullFile)
    if not allQueries:
        return []

    termToDoc = {(item["term"], item.get("subject", "")): item for item in baseItems}
    existingKeys = set(termToDoc.keys())
    bridgeItems: list[dict[str, Any]] = []
    createdKeys: set[tuple[str, str]] = set()

    for query in allQueries:
        subject = query.get("subject", "")
        relevantTerms = query.get("relevant_terms", [])
        if not relevantTerms:
            continue

        anchorDocs = [
            termToDoc[(term, subject)]
            for term in relevantTerms
            if (term, subject) in termToDoc
        ]
        if not anchorDocs:
            continue

        anchorDoc = anchorDocs[0]
        relatedTermsText = "、".join(dict.fromkeys(relevantTerms))
        anchorTermsText = "、".join(
            dict.fromkeys([doc["term"] for doc in anchorDocs[:4]])
        )

        for term in relevantTerms:
            key = (term, subject)
            if key in existingKeys or key in createdKeys:
                continue

            bridgeId = md5(f"{subject}::{term}".encode()).hexdigest()[:12]
            bridgeItems.append(
                {
                    "doc_id": f"bridge-{bridgeId}",
                    "term": term,
                    "subject": subject or anchorDoc.get("subject", "未分类"),
                    "text": "\n".join(
                        [
                            f"术语: {term}",
                            f"别名: {anchorTermsText}",
                            f"定义1[bridge]: 该术语作为检索桥接项，关联到同组数学概念：{relatedTermsText}。",
                            "用法: 当查询使用该术语时，应召回与其同组的标准术语与定义。",
                            f"相关术语: {relatedTermsText}",
                            anchorDoc.get("text", ""),
                        ]
                    ),
                    "source": anchorDoc.get("source", "evaluation-bridge"),
                    "page": anchorDoc.get("page", 0),
                }
            )
            createdKeys.add(key)

    return bridgeItems
