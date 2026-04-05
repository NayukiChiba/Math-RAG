"""生成质量对比实验：RAG vs 无检索

功能：
1. 评估现有 outputs/rag_results.jsonl 中的 RAG 生成质量
   （术语命中率、来源引用率、回答有效率）
2. 可选：运行无检索基线推理，输出同组指标
3. 合并输出到 log 时间目录下的 JSON 文件

用法：
    python scripts/evaluation/evalGenerationComparison.py
    python scripts/evaluation/evalGenerationComparison.py --norag-limit 15
    python scripts/evaluation/evalGenerationComparison.py --skip-norag
"""

import argparse
import json
import os
import sys
import time
from typing import Any

from core import config
from core.utils import getFileLoader
from research.modelEvaluation.evalGeneration import (
    calculateSourceCitationRate,
    calculateTermHitRate,
    isAnswerValid,
)

_LOADER = getFileLoader()


# ── 数据加载 ────────────────────────────────────────────────────


def _load_rag_results(path: str) -> list[dict[str, Any]]:
    return _LOADER.jsonl(path)


def _load_queries(path: str) -> dict[str, dict[str, Any]]:
    """返回 query -> record 映射。"""
    gold: dict[str, dict[str, Any]] = {}
    for rec in _LOADER.jsonl(path):
        gold[rec["query"]] = rec
    return gold


def _load_retrieval_metrics(
    all_methods_path: str, method: str = "BM25+"
) -> dict[str, float]:
    """从 all_methods.json 中提取指定方法的检索指标。"""
    try:
        data = _LOADER.json(all_methods_path)
        for r in data.get("results", []):
            if r["method"] == method:
                am = r.get("avg_metrics", {})
                return {
                    "recall@5": am.get("recall@5", 0.0),
                    "mrr": am.get("mrr", 0.0),
                    "map": am.get("map", 0.0),
                }
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass
    return {"recall@5": 0.0, "mrr": 0.0, "map": 0.0}


# ── 评估现有 RAG 结果 ──────────────────────────────────────────


def evaluate_rag_results(
    rag_results: list[dict[str, Any]],
    gold_map: dict[str, dict[str, Any]],
    retrieval_metrics: dict[str, float],
) -> dict[str, Any]:
    """评估 rag_results.jsonl 中的 RAG 生成质量。"""
    print(f"\n{'=' * 60}")
    print(" 评估 RAG 生成质量（来自 rag_results.jsonl）")
    print(f"{'=' * 60}")

    term_hits: list[float] = []
    src_cites: list[float] = []
    valid_count = 0
    latencies: list[float] = []

    for i, r in enumerate(rag_results, 1):
        query = r.get("query", "")
        answer = r.get("answer", "")
        sources = r.get("sources", [])
        gold = gold_map.get(query, {})
        rel_terms = gold.get("relevant_terms", [])

        if i % 20 == 0 or i == 1:
            print(f"  [{i}/{len(rag_results)}] {query[:30]}...")

        th = calculateTermHitRate(answer, rel_terms)
        sc = calculateSourceCitationRate(answer, sources)
        av = isAnswerValid(answer)

        term_hits.append(th["rate"])
        src_cites.append(sc["rate"])
        if av["valid"]:
            valid_count += 1

        lat = r.get("latency", {})
        if isinstance(lat, dict):
            latencies.append(lat.get("total_ms", 0.0))

    n = len(rag_results)
    avg_term_hit = sum(term_hits) / n if n else 0.0
    avg_src_cite = sum(src_cites) / n if n else 0.0
    avg_valid = valid_count / n if n else 0.0
    avg_lat = sum(latencies) / n if latencies else 0.0

    print(f"\n  术语命中率:   {avg_term_hit:.4f} ({avg_term_hit * 100:.1f}%)")
    print(f"  来源引用率:   {avg_src_cite:.4f} ({avg_src_cite * 100:.1f}%)")
    print(f"  回答有效率:   {avg_valid:.4f} ({avg_valid * 100:.1f}%)")
    print(f"  平均延迟(ms): {avg_lat:.0f}")

    return {
        "group": "rag-bm25plus",
        "strategy": "BM25+（Qwen2.5-Math-7B）",
        "total_queries": n,
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": {
            "term_hit_rate": avg_term_hit,
            "source_citation_rate": avg_src_cite,
            "answer_valid_rate": avg_valid,
        },
        "avg_latency_ms": avg_lat,
    }


# ── 无检索基线推理 ─────────────────────────────────────────────


def run_norag_baseline(
    queries: list[dict[str, Any]],
    gold_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """运行无检索基线：Qwen 直接回答，不注入任何上下文。"""
    print(f"\n{'=' * 60}")
    print(f" 无检索基线推理（{len(queries)} 条查询）")
    print(f"{'=' * 60}")

    # 延迟导入，避免未安装 transformers 时直接报错
    from core.answerGeneration.localInference import LocalInference  # noqa: PLC0415

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    print("  加载模型中……")
    model = LocalInference(modelDir=config.LOCAL_MODEL_DIR)

    term_hits: list[float] = []
    src_cites: list[float] = []
    valid_count = 0
    latencies: list[float] = []
    results: list[dict[str, Any]] = []

    for i, q in enumerate(queries, 1):
        query_text = q["query"]
        gold = gold_map.get(query_text, q)
        rel_terms = gold.get("relevant_terms", [])

        print(f"  [{i}/{len(queries)}] {query_text[:40]}...")

        messages = [
            {
                "role": "system",
                "content": "你是一位专业的数学教学助手。请直接回答用户的数学问题。",
            },
            {"role": "user", "content": query_text},
        ]

        t0 = time.time()
        try:
            answer = model.generateFromMessages(messages)
        except Exception as exc:
            print(f"     生成失败: {exc}")
            answer = ""
        lat = (time.time() - t0) * 1000

        th = calculateTermHitRate(answer, rel_terms)
        sc = calculateSourceCitationRate(answer, [])
        av = isAnswerValid(answer)

        term_hits.append(th["rate"])
        src_cites.append(sc["rate"])  # 无来源，恒为 0
        if av["valid"]:
            valid_count += 1
        latencies.append(lat)

        results.append(
            {
                "query": query_text,
                "answer": answer,
                "latency_ms": lat,
                "term_hit_rate": th["rate"],
            }
        )

    n = len(queries)
    avg_term_hit = sum(term_hits) / n if n else 0.0
    avg_valid = valid_count / n if n else 0.0
    avg_lat = sum(latencies) / n if latencies else 0.0

    print(f"\n  术语命中率:   {avg_term_hit:.4f} ({avg_term_hit * 100:.1f}%)")
    print("  来源引用率:   0.0000 (0.0%)  [无检索，固定为 0]")
    print(f"  回答有效率:   {avg_valid:.4f} ({avg_valid * 100:.1f}%)")
    print(f"  平均延迟(ms): {avg_lat:.0f}")

    return {
        "group": "baseline-norag",
        "strategy": None,
        "total_queries": n,
        "retrieval_metrics": {"recall@5": 0.0, "mrr": 0.0, "map": 0.0},
        "generation_metrics": {
            "term_hit_rate": avg_term_hit,
            "source_citation_rate": 0.0,
            "answer_valid_rate": avg_valid,
        },
        "avg_latency_ms": avg_lat,
        "results": results,
    }


# ── 报告生成 ───────────────────────────────────────────────────


def build_comparison_report(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """组合各实验组结果，生成对比报告 JSON。"""
    if not groups:
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_groups": 0,
            "groups": [],
            "comparison": {},
        }
    best_term = max(groups, key=lambda g: g["generation_metrics"]["term_hit_rate"])
    best_src = max(
        groups, key=lambda g: g["generation_metrics"]["source_citation_rate"]
    )

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_groups": len(groups),
        "groups": [
            {
                "group": g["group"],
                "strategy": g["strategy"],
                "total_queries": g["total_queries"],
                "retrieval_metrics": g["retrieval_metrics"],
                "generation_metrics": g["generation_metrics"],
                "avg_latency_ms": g["avg_latency_ms"],
            }
            for g in groups
        ],
        "comparison": {
            "best_term_hit_rate": {
                "group": best_term["group"],
                "value": best_term["generation_metrics"]["term_hit_rate"],
            },
            "best_source_citation_rate": {
                "group": best_src["group"],
                "value": best_src["generation_metrics"]["source_citation_rate"],
            },
        },
    }


def print_markdown_table(groups: list[dict[str, Any]]) -> None:
    """打印 Markdown 格式对比表，方便直接查看。"""
    print("\n## 生成质量对比表\n")
    print(
        "| 实验组 | 检索策略 | Recall@5 | 术语命中率 | 来源引用率 | 回答有效率 | 延迟(ms) |"
    )
    print(
        "|--------|----------|----------|------------|------------|------------|----------|"
    )
    for g in groups:
        strategy = g["strategy"] or "无"
        rm = g["retrieval_metrics"]
        gm = g["generation_metrics"]
        recall = f"{rm['recall@5']:.4f}" if rm["recall@5"] else "N/A"
        print(
            f"| {g['group']} | {strategy} | {recall} "
            f"| {gm['term_hit_rate']:.4f} "
            f"| {gm['source_citation_rate']:.4f} "
            f"| {gm['answer_valid_rate']:.4f} "
            f"| {g['avg_latency_ms']:.0f} |"
        )


# ── 主函数 ─────────────────────────────────────────────────────


def main() -> None:
    outputController = config.getOutputController()
    parser = argparse.ArgumentParser(description="生成质量对比：RAG vs 无检索")
    parser.add_argument(
        "--rag-results",
        default=config.RAG_RESULTS_FILE,
        help="RAG 生成结果文件（JSONL）",
    )
    parser.add_argument(
        "--queries",
        default=os.path.join(config.EVALUATION_DIR, "queries.jsonl"),
        help="查询集文件（含 relevant_terms）",
    )
    parser.add_argument(
        "--all-methods",
        default=os.path.join(
            outputController.get_json_dir(), "full_eval", "all_methods.json"
        ),
        help="全量检索评测结果（用于填充检索指标）",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            outputController.get_json_dir(), "comparison_results.json"
        ),
        help="输出对比结果 JSON 路径",
    )
    parser.add_argument(
        "--norag-limit",
        type=int,
        default=15,
        help="无检索基线推理的查询数量（0 = skip）",
    )
    parser.add_argument(
        "--skip-norag",
        action="store_true",
        help="跳过无检索基线推理，仅评估 RAG",
    )
    args = parser.parse_args()
    args.output = outputController.normalize_json_path(
        args.output, "comparison_results.json"
    )

    print("=" * 60)
    print(" Math-RAG 生成质量对比实验")
    print("=" * 60)

    # 文件存在性检查
    for label, path in [("RAG结果", args.rag_results), ("查询集", args.queries)]:
        if not os.path.isfile(path):
            print(f"[错误] {label}文件不存在: {path}")
            sys.exit(1)

    # 加载数据
    rag_results = _load_rag_results(args.rag_results)
    gold_map = _load_queries(args.queries)
    retrieval_metrics = _load_retrieval_metrics(args.all_methods, method="BM25+")
    print(f"\n[数据] RAG 结果: {len(rag_results)} 条")
    print(f"[数据] 查询集: {len(gold_map)} 条")
    print(f"[数据] BM25+ Recall@5: {retrieval_metrics['recall@5']:.4f}")

    groups: list[dict[str, Any]] = []

    # 1. 评估 RAG 结果
    rag_group = evaluate_rag_results(rag_results, gold_map, retrieval_metrics)
    groups.append(rag_group)

    # 2. 无检索基线
    run_norag = not args.skip_norag and args.norag_limit > 0
    if run_norag:
        all_queries = sorted(gold_map.values(), key=lambda q: q["query"])
        norag_queries = all_queries[: args.norag_limit]
        norag_group = run_norag_baseline(norag_queries, gold_map)
        groups.append(norag_group)
    else:
        print("\n[跳过] 无检索基线推理")

    # 3. 生成报告
    report = build_comparison_report(groups)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 对比结果已保存: {args.output}")

    # 4. 打印表格
    print_markdown_table(groups)


if __name__ == "__main__":
    main()
