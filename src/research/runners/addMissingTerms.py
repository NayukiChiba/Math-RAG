"""补充缺失术语入口（支持 analyze/apply/report 模式）。"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime
from typing import Any

from core import config
from research.runners.tools import addMissingTerms as terms


def _load_existing_terms(corpus_file: str) -> tuple[set[str], set[str]]:
    existing_doc_ids: set[str] = set()
    existing_terms: set[str] = set()
    for entry in terms._LOADER.jsonl(corpus_file):
        existing_doc_ids.add(entry["doc_id"])
        existing_terms.add(entry["term"])
    return existing_doc_ids, existing_terms


def _build_auto_candidates(
    queries_file: str, existing_terms: set[str], top_n: int
) -> list[dict[str, Any]]:
    missing_counter: Counter[str] = Counter()
    sample_queries: dict[str, str] = {}
    subject_map: dict[str, str] = {}

    for rec in terms._LOADER.jsonl(queries_file):
        query = rec.get("query", "")
        subject = rec.get("subject", "未知")
        for term_name in rec.get("relevant_terms", []):
            if not isinstance(term_name, str) or not term_name.strip():
                continue
            term_name = term_name.strip()
            if term_name in existing_terms:
                continue
            missing_counter[term_name] += 1
            sample_queries.setdefault(term_name, query)
            subject_map.setdefault(term_name, subject)

    candidates = []
    for term_name, freq in missing_counter.most_common(top_n):
        candidates.append(
            {
                "term": term_name,
                "subject": subject_map.get(term_name, "未知"),
                "frequency": freq,
                "sample_query": sample_queries.get(term_name, ""),
                "status": "needs_definition",
            }
        )
    return candidates


def _save_missing_report(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_entries_file(path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到补全条目文件: {path}")
    data = terms._LOADER.json(path)
    if isinstance(data, dict):
        rows = data.get("entries", [])
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError("补全条目文件格式错误，应为数组或含 entries 字段的对象")
    if not isinstance(rows, list):
        raise ValueError("entries 字段必须是数组")
    return rows


def _append_entries(
    corpus_file: str,
    rows: list[dict[str, Any]],
    existing_doc_ids: set[str],
    existing_terms: set[str],
) -> list[dict[str, Any]]:
    to_add: list[dict[str, Any]] = []
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        term_name = str(entry.get("term", "")).strip()
        doc_id = str(entry.get("doc_id", "")).strip()
        if not term_name:
            continue
        if not doc_id:
            # 与历史脚本兼容：若未给 doc_id 则自动生成占位 id
            subject = str(entry.get("subject", "misc")).strip() or "misc"
            doc_id = f"auto-{subject}-{term_name}"
            entry["doc_id"] = doc_id
        if term_name in existing_terms or doc_id in existing_doc_ids:
            continue
        to_add.append(entry)

    if not to_add:
        return []

    with open(corpus_file, "a", encoding="utf-8") as f:
        for entry in to_add:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return to_add


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="缺失术语补全工具")
    parser.add_argument(
        "--mode",
        choices=["analyze", "apply", "report"],
        default="analyze",
        help="analyze=分析缺口并输出候选；apply=写入补全条目；report=仅打印统计",
    )
    parser.add_argument(
        "--queries",
        default=os.path.join(config.EVALUATION_DIR, "queries.jsonl"),
        help="用于分析缺口的查询集",
    )
    parser.add_argument(
        "--entries-file",
        default=None,
        help="apply 模式下使用的补全条目 JSON 文件；不传则回退到内置 MISSING_TERM_ENTRIES",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(config.EVALUATION_DIR, "missing_terms_candidates.json"),
        help="analyze 模式输出候选报告路径",
    )
    parser.add_argument(
        "--top-n", type=int, default=80, help="analyze 模式输出候选上限"
    )
    args = parser.parse_args(argv)

    corpus_file = os.path.join(config.PROCESSED_DIR, "retrieval", "corpus.jsonl")
    existing_doc_ids, existing_terms = _load_existing_terms(corpus_file)

    print(f"现有语料库：{len(existing_terms)} 个术语，{len(existing_doc_ids)} 条文档")

    if args.mode == "report":
        print("[REPORT] 仅统计模式完成。")
        return

    if args.mode == "analyze":
        candidates = _build_auto_candidates(args.queries, existing_terms, args.top_n)
        payload = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "queries_mtime": time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(os.path.getmtime(args.queries)),
            )
            if os.path.exists(args.queries)
            else None,
            "queries_file": args.queries,
            "corpus_file": corpus_file,
            "candidate_count": len(candidates),
            "entries": candidates,
        }
        _save_missing_report(args.output, payload)
        print(f"[ANALYZE] 缺口候选数：{len(candidates)}")
        print(f"[ANALYZE] 候选文件：{args.output}")
        return

    if args.entries_file:
        source_rows = _load_entries_file(args.entries_file)
    else:
        source_rows = list(terms.MISSING_TERM_ENTRIES)

    to_add = _append_entries(corpus_file, source_rows, existing_doc_ids, existing_terms)
    print(
        f"[APPLY] 待写入条目：{len(to_add)} / 输入条目：{len(source_rows)}（自动跳过已存在）"
    )
    if not to_add:
        print("[APPLY] 没有新增术语，跳过写入。")
        return

    for entry in to_add[:20]:
        print(f"  [ADD] {entry.get('term', '')} ({entry.get('subject', '未知')})")
    if len(to_add) > 20:
        print(f"  ... 其余 {len(to_add) - 20} 条略")

    print(f"\n[APPLY] 已添加 {len(to_add)} 个缺失术语到语料库")
    print(f"[APPLY] 语料库路径：{corpus_file}")
    print("\n[NEXT] 请重建索引：")
    print("  math-rag build-index --rebuild")


if __name__ == "__main__":
    main()
