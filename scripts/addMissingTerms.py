"""补充缺失术语入口。"""

from __future__ import annotations

import json
import os

import config
from scripts.tools import addMissingTerms as terms


def main(argv: list[str] | None = None) -> None:
    corpus_file = os.path.join(config.PROCESSED_DIR, "retrieval", "corpus.jsonl")

    existing_doc_ids: set[str] = set()
    existing_terms: set[str] = set()
    for entry in terms._LOADER.jsonl(corpus_file):
        existing_doc_ids.add(entry["doc_id"])
        existing_terms.add(entry["term"])

    print(f"现有语料库：{len(existing_terms)} 个术语，{len(existing_doc_ids)} 条文档")

    to_add = [
        e
        for e in terms.MISSING_TERM_ENTRIES
        if e["doc_id"] not in existing_doc_ids and e["term"] not in existing_terms
    ]
    print(
        f"待添加条目：{len(to_add)} 个（跳过 {len(terms.MISSING_TERM_ENTRIES) - len(to_add)} 个已存在）"
    )

    if not to_add:
        print(" 所有术语已在语料库中，无需添加。")
        return

    with open(corpus_file, "a", encoding="utf-8") as f:
        for entry in to_add:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"   添加：{entry['term']} ({entry['subject']})")

    print(f"\n 已添加 {len(to_add)} 个缺失术语到语料库")
    print(f"语料库路径：{corpus_file}")
    print("\n  请重建 BM25+ 索引：")
    print("   python retrieval/buildCorpus.py  # 如果需要重建 corpus.jsonl")
    print("   python scripts/rebuildIndex.py  # 重建 BM25+ 索引")


if __name__ == "__main__":
    main()
