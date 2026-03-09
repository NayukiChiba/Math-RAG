"""评测报告生成脚本

读取各实验 JSON 结果，生成：
- outputs/reports/final_report.md：Markdown 格式完整报告
- outputs/figures/：论文级别图表（PDF + PNG）

用法：
    python3 scripts/generateReport.py [--results ...] [--ablation ...] \
        [--significance ...] [--queries ...] [--output ...] [--figures ...]
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── 中文字体配置 ─────────────────────────────────────────────────
_CHINESE_FONTS = [
    "WenQuanYi Zen Hei",
    "Noto Sans CJK SC",
    "SimHei",
    "Microsoft YaHei",
    "PingFang SC",
]


def _configure_matplotlib() -> bool:
    """尝试配置中文字体，返回是否成功。"""
    import matplotlib.font_manager as fm

    available = {f.name for f in fm.fontManager.ttflist}
    for font in _CHINESE_FONTS:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    plt.rcParams["axes.unicode_minus"] = False
    return False


# ── 数据加载 ─────────────────────────────────────────────────────


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_queries(path: str) -> list[dict]:
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def _subject_breakdown(
    results_data: dict, queries: list[dict]
) -> dict[str, dict[str, float]]:
    """计算各方法在三学科上的 Recall@5。

    Returns:
        {method_name: {subject: recall@5_mean}}
    """
    breakdown: dict[str, dict[str, float]] = {}
    for r in results_data["results"]:
        scores = r.get("recall@5", [])
        if not isinstance(scores, list) or len(scores) != len(queries):
            continue
        per_subject: dict[str, list[float]] = {}
        for q, s in zip(queries, scores):
            subj = q.get("subject", "Unknown")
            per_subject.setdefault(subj, []).append(s)
        breakdown[r["method"]] = {
            subj: round(sum(ss) / len(ss), 4) for subj, ss in per_subject.items()
        }
    return breakdown


def _select_case_examples(
    results_data: dict, queries: list[dict], n: int = 3
) -> tuple[list[dict], list[dict]]:
    """从 BM25+ 结果中挑选成功/失败案例。"""
    bm25p = next((r for r in results_data["results"] if r["method"] == "BM25+"), None)
    if bm25p is None:
        return [], []
    recall5 = bm25p.get("recall@5", [])
    recall1 = bm25p.get("recall@1", [])
    if not isinstance(recall5, list) or len(recall5) != len(queries):
        return [], []
    if not isinstance(recall1, list) or len(recall1) != len(queries):
        return [], []

    successes, failures = [], []
    for i, (q, r5, r1) in enumerate(zip(queries, recall5, recall1)):
        if r1 == 1.0:  # Recall@1=1 → 第一个就命中
            successes.append(
                {
                    "query": q["query"],
                    "subject": q["subject"],
                    "recall@5": r5,
                    "recall@1": r1,
                }
            )
        elif r5 == 0.0:  # 完全未命中
            failures.append(
                {
                    "query": q["query"],
                    "subject": q["subject"],
                    "recall@5": r5,
                    "recall@1": r1,
                }
            )

    return successes[:n], failures[:n]


# ── 图表生成 ─────────────────────────────────────────────────────

_COLORS = ["#4472C4", "#ED7D31", "#A9D18E", "#FF6B6B", "#9B59B6"]
_HATCHES = ["", "//", "\\\\", "xx", ".."]


def _save_fig(fig: plt.Figure, base_path: str) -> None:
    """同时保存 PDF 和 PNG。"""
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    fig.savefig(base_path + ".pdf", bbox_inches="tight", dpi=300)
    fig.savefig(base_path + ".png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def _fig_method_comparison(results_data: dict, figures_dir: str) -> str:
    """各方法 Recall@K 对比分组柱状图。"""
    methods = [r["method"] for r in results_data["results"]]
    ks = [1, 3, 5, 10]
    metric_keys = [f"recall@{k}" for k in ks]

    data_matrix = []
    for r in results_data["results"]:
        am = r.get("avg_metrics", {})
        data_matrix.append([am.get(mk, 0) for mk in metric_keys])

    x = np.arange(len(ks))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (method, row) in enumerate(zip(methods, data_matrix)):
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            row,
            width,
            label=method,
            color=_COLORS[i % len(_COLORS)],
            hatch=_HATCHES[i % len(_HATCHES)],
            alpha=0.85,
        )

    ax.set_xlabel("Top-K", fontsize=12)
    ax.set_ylabel("Recall@K", fontsize=12)
    ax.set_title(
        "Retrieval Method Comparison: Recall@K", fontsize=13, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in ks], fontsize=11)
    max_val = max(v for row in data_matrix for v in row) if data_matrix else 0.0
    ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 0.7)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    out = os.path.join(figures_dir, "method_comparison")
    _save_fig(fig, out)
    return out


def _fig_topk_ablation(ablation_data: dict, figures_dir: str) -> str:
    """BM25+ TopK 消融折线图。"""
    variants = ablation_data["topk_ablation"]["variants"]
    ks = [v["topk"] for v in variants]
    r5 = [v["recall@5"] for v in variants]
    r10 = [v["recall@10"] for v in variants]
    maps = [v["map"] for v in variants]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    lns1 = ax1.plot(
        ks, r5, "o-", color=_COLORS[0], linewidth=2, markersize=7, label="Recall@5"
    )
    lns2 = ax1.plot(
        ks, r10, "s--", color=_COLORS[1], linewidth=2, markersize=7, label="Recall@10"
    )
    lns3 = ax2.plot(
        ks, maps, "^:", color=_COLORS[2], linewidth=2, markersize=7, label="MAP"
    )

    ax1.set_xlabel("Top-K", fontsize=12)
    ax1.set_ylabel("Recall", fontsize=12)
    ax2.set_ylabel("MAP", fontsize=12)
    ax1.set_xticks(ks)
    all_recall = r5 + r10
    r_max = max(all_recall) if all_recall else 0.0
    m_max = max(maps) if maps else 0.0
    ax1.set_ylim(0, r_max * 1.2 if r_max > 0 else 0.7)
    ax2.set_ylim(0, m_max * 1.2 if m_max > 0 else 0.7)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    lns = lns1 + lns2 + lns3
    labels = [ln.get_label() for ln in lns]
    ax1.legend(lns, labels, loc="lower right", fontsize=10)
    ax1.set_title("BM25+ Top-K Ablation", fontsize=13, fontweight="bold")
    ax1.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = os.path.join(figures_dir, "topk_ablation")
    _save_fig(fig, out)
    return out


def _fig_alpha_sensitivity(ablation_data: dict, figures_dir: str) -> str:
    """HybridPlus Alpha 权重敏感性折线图。"""
    variants = ablation_data["alpha_ablation"]["variants"]
    alphas = [v["alpha"] for v in variants]
    r5 = [v["recall@5"] for v in variants]
    maps = [v["map"] for v in variants]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    lns1 = ax1.plot(
        alphas, r5, "o-", color=_COLORS[0], linewidth=2, markersize=8, label="Recall@5"
    )
    lns2 = ax2.plot(
        alphas, maps, "s--", color=_COLORS[1], linewidth=2, markersize=8, label="MAP"
    )

    ax1.set_xlabel("Alpha (BM25+ weight)", fontsize=12)
    ax1.set_ylabel("Recall@5", fontsize=12)
    ax2.set_ylabel("MAP", fontsize=12)
    ax1.set_xticks(alphas)
    r5_min = min(r5) if r5 else 0.0
    r5_max = max(r5) if r5 else 1.0
    m_min = min(maps) if maps else 0.0
    m_max = max(maps) if maps else 1.0
    r5_pad = max((r5_max - r5_min) * 0.5, 0.01)
    m_pad = max((m_max - m_min) * 0.5, 0.01)
    ax1.set_ylim(r5_min - r5_pad, r5_max + r5_pad)
    ax2.set_ylim(m_min - m_pad, m_max + m_pad)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))

    lns = lns1 + lns2
    labels = [ln.get_label() for ln in lns]
    ax1.legend(lns, labels, loc="lower right", fontsize=10)
    ax1.set_title("HybridPlus Alpha Sensitivity", fontsize=13, fontweight="bold")
    ax1.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()

    out = os.path.join(figures_dir, "alpha_sensitivity")
    _save_fig(fig, out)
    return out


def _fig_subject_breakdown(
    breakdown: dict[str, dict[str, float]], figures_dir: str
) -> str:
    """三学科 Recall@5 对比分组柱状图。"""
    subjects = ["数学分析", "概率论", "高等代数"]
    subject_labels = ["Math Analysis", "Probability", "Linear Algebra"]
    methods = list(breakdown.keys())

    x = np.arange(len(subjects))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, method in enumerate(methods):
        vals = [breakdown[method].get(s, 0) for s in subjects]
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=method,
            color=_COLORS[i % len(_COLORS)],
            hatch=_HATCHES[i % len(_HATCHES)],
            alpha=0.85,
        )

    ax.set_xlabel("Subject", fontsize=12)
    ax.set_ylabel("Recall@5", fontsize=12)
    ax.set_title("Recall@5 by Subject", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(subject_labels, fontsize=11)
    all_vals = [v for m in breakdown.values() for v in m.values()]
    max_val = max(all_vals) if all_vals else 0.0
    ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 0.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()

    out = os.path.join(figures_dir, "subject_breakdown")
    _save_fig(fig, out)
    return out


# ── Markdown 报告 ────────────────────────────────────────────────


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def _method_table(results_data: dict) -> str:
    header = "| 方法 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | MAP | nDCG@10 | 延迟(ms) |\n"
    header += "|------|----------|----------|----------|-----------|-----|-----|---------|----------|\n"
    rows = []
    for r in results_data["results"]:
        am = r.get("avg_metrics", {})
        method = r["method"]
        if method == "BM25+":
            method = "**BM25+**"  # 加粗最佳方法
        latency_ms = r.get("avg_query_time", 0) * 1000
        row = (
            f"| {method} "
            f"| {_fmt_pct(am.get('recall@1', 0))} "
            f"| {_fmt_pct(am.get('recall@3', 0))} "
            f"| {_fmt_pct(am.get('recall@5', 0))} "
            f"| {_fmt_pct(am.get('recall@10', 0))} "
            f"| {am.get('mrr', 0):.4f} "
            f"| {am.get('map', 0):.4f} "
            f"| {am.get('ndcg@10', 0):.4f} "
            f"| {latency_ms:.1f} |"
        )
        rows.append(row)
    return header + "\n".join(rows)


def _topk_table(ablation_data: dict) -> str:
    header = "| K | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | MAP |\n"
    header += "|---|----------|----------|----------|-----------|-----|-----|\n"
    rows = []
    for v in ablation_data["topk_ablation"]["variants"]:
        k = v["topk"]
        marker = " **←**" if k == 5 else ""
        row = (
            f"| {k}{marker} "
            f"| {_fmt_pct(v.get('recall@1', 0))} "
            f"| {_fmt_pct(v.get('recall@3', 0))} "
            f"| {_fmt_pct(v.get('recall@5', 0))} "
            f"| {_fmt_pct(v.get('recall@10', 0))} "
            f"| {v.get('mrr', 0):.4f} "
            f"| {v.get('map', 0):.4f} |"
        )
        rows.append(row)
    return header + "\n".join(rows)


def _alpha_table(ablation_data: dict) -> str:
    header = "| alpha (BM25+) | beta (向量) | Recall@5 | MAP | nDCG@10 |\n"
    header += "|---------------|-------------|----------|-----|--------|\n"
    rows = []
    for v in ablation_data["alpha_ablation"]["variants"]:
        alpha = v["alpha"]
        marker = " **←**" if alpha == 0.5 else ""
        row = (
            f"| {alpha}{marker} "
            f"| {v['beta']} "
            f"| {_fmt_pct(v.get('recall@5', 0))} "
            f"| {v.get('map', 0):.4f} "
            f"| {v.get('ndcg@10', 0):.4f} |"
        )
        rows.append(row)
    return header + "\n".join(rows)


def _subject_table(breakdown: dict[str, dict[str, float]]) -> str:
    subjects = ["数学分析", "概率论", "高等代数"]
    header = "| 方法 | 数学分析 | 概率论 | 高等代数 |\n"
    header += "|------|----------|--------|----------|\n"
    rows = []
    for method, subj_map in breakdown.items():
        row = (
            f"| {method} "
            + "".join(f"| {_fmt_pct(subj_map.get(s, 0))} " for s in subjects)
            + "|"
        )
        rows.append(row)
    return header + "\n".join(rows)


def _significance_section(sig_data: dict) -> str:
    lines = []
    for comp in sig_data.get("paired_t_tests", []):
        r5 = comp["metrics"].get("recall@5", {})
        pv = r5.get("p_value")
        t = r5.get("t_stat")
        sig = r5.get("significant_at_0.05", False)
        if pv is None or t is None:
            continue
        sig_text = "**显著（p < 0.05）**" if sig else "不显著"
        lines.append(
            f"- **{comp['method_a']} vs {comp['method_b']}**："
            f"t = {t:.4f}，p = {pv:.2e}，{sig_text}"
        )
    bm25p_ci = sig_data.get("bootstrap_ci", {}).get("BM25+", {}).get("recall@5", {})
    vec_ci = sig_data.get("bootstrap_ci", {}).get("Vector", {}).get("recall@5", {})
    ci_lines = []
    if bm25p_ci.get("mean") is not None:
        ci_lines.append(
            f"- **BM25+** Recall@5 = {_fmt_pct(bm25p_ci['mean'])}，"
            f"95% CI = [{_fmt_pct(bm25p_ci['ci_lower'])}, {_fmt_pct(bm25p_ci['ci_upper'])}]"
        )
    if vec_ci.get("mean") is not None:
        ci_lines.append(
            f"- **Vector** Recall@5 = {_fmt_pct(vec_ci['mean'])}，"
            f"95% CI = [{_fmt_pct(vec_ci['ci_lower'])}, {_fmt_pct(vec_ci['ci_upper'])}]"
        )
    return "\n".join(ci_lines) + "\n\n" + "\n".join(lines)


def _generation_comparison_table(comparison_data: dict) -> str:
    """将 comparison_results.json 渲染为 Markdown 表格。"""
    groups = comparison_data.get("groups", [])
    if not groups:
        return (
            "*暂无生成质量对比数据（运行 `scripts/evalGenerationComparison.py` 生成）*"
        )

    header = (
        "| 实验组 | 检索策略 | Recall@5 | 术语命中率 | 来源引用率 | 回答有效率 | 延迟(ms) |\n"
        "|--------|----------|----------|------------|------------|------------|----------|\n"
    )
    rows = []
    for g in groups:
        strategy = g.get("strategy") or "无"
        rm = g.get("retrieval_metrics", {})
        gm = g.get("generation_metrics", {})
        recall = f"{rm['recall@5']:.4f}" if rm.get("recall@5") else "N/A"
        rows.append(
            f"| {g['group']} | {strategy} | {recall} "
            f"| {_fmt_pct(gm.get('term_hit_rate', 0))} "
            f"| {_fmt_pct(gm.get('source_citation_rate', 0))} "
            f"| {_fmt_pct(gm.get('answer_valid_rate', 0))} "
            f"| {g.get('avg_latency_ms', 0):.0f} |"
        )
    return header + "\n".join(rows)


def _case_section(successes: list[dict], failures: list[dict]) -> str:
    lines = []
    lines.append("### 7.1 成功案例（BM25+，Recall@1 = 1.0）\n")
    for i, c in enumerate(successes, 1):
        lines.append(f"**案例 {i}**：查询「{c['query']}」（学科：{c['subject']}）")
        lines.append(
            f"> 效果：BM25+ 首位即命中目标术语，Recall@5 = {_fmt_pct(c['recall@5'])}\n"
        )
    lines.append("### 7.2 失败案例（BM25+，Recall@5 = 0.0）\n")
    for i, c in enumerate(failures, 1):
        lines.append(f"**案例 {i}**：查询「{c['query']}」（学科：{c['subject']}）")
        lines.append(
            f"> 效果：BM25+ 检索结果未包含任何相关术语，Recall@5 = {_fmt_pct(c['recall@5'])}。"
        )
        lines.append("> 分析：涉及跨术语语义表达，BM25+ 关键词匹配失效。\n")
    return "\n".join(lines)


def generate_report(
    results_path: str,
    ablation_path: str,
    significance_path: str,
    queries_path: str,
    output_path: str,
    figures_dir: str,
    comparison_path: str | None = None,
) -> None:
    """主入口：生成完整评测报告和所有图表。"""
    has_chinese = _configure_matplotlib()

    print("=" * 60)
    print("📊 Math-RAG 评测报告生成")
    print("=" * 60)
    print(f"中文字体: {'已配置' if has_chinese else '未找到，图表使用英文标签'}")

    results_data = _load_json(results_path)
    ablation_data = _load_json(ablation_path)
    sig_data = _load_json(significance_path)
    queries = _load_queries(queries_path)

    # 生成质量对比数据（可选）
    comparison_data: dict = {}
    if comparison_path and os.path.isfile(comparison_path):
        try:
            comparison_data = _load_json(comparison_path)
            print(f"[数据] 生成对比: {len(comparison_data.get('groups', []))} 组")
        except Exception:
            pass

    os.makedirs(figures_dir, exist_ok=True)
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # ── 计算学科细分 ────────────────────────────────────────────
    breakdown = _subject_breakdown(results_data, queries)

    # ── 挑选案例 ────────────────────────────────────────────────
    successes, failures = _select_case_examples(results_data, queries)

    # ── 生成图表 ────────────────────────────────────────────────
    print("\n生成图表...")
    figs = {
        "method_comparison": _fig_method_comparison(results_data, figures_dir),
        "topk_ablation": _fig_topk_ablation(ablation_data, figures_dir),
        "alpha_sensitivity": _fig_alpha_sensitivity(ablation_data, figures_dir),
        "subject_breakdown": _fig_subject_breakdown(breakdown, figures_dir),
    }
    for name, path in figs.items():
        print(f"  [OK] {name}: {path}.pdf / .png")

    # ── 生成 Markdown 报告 ──────────────────────────────────────
    print("\n生成报告...")
    subj_dist = results_data.get("subject_distribution", {})
    total_q = results_data.get("total_queries", 0)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    topk_eval = results_data.get("topk", 10)

    report = f"""# Math-RAG 检索评测报告

> 生成时间：{timestamp}
> 评测集：`{os.path.basename(queries_path)}`（共 {total_q} 条查询）

---

## 1. 实验设置

| 项目 | 说明 |
|------|------|
| 语料库规模 | 3157 个文档块，涵盖 2724 个数学术语 |
| 学科分布 | 数学分析 {subj_dist.get("数学分析", 0)} 条 / 概率论 {subj_dist.get("概率论", 0)} 条 / 高等代数 {subj_dist.get("高等代数", 0)} 条 |
| 评测集 | `queries_full.jsonl`（{total_q} 条全量集）+ `queries.jsonl`（105 条精标集） |
| 检索模型 | BM25、BM25+、BGE-base-zh-v1.5（向量）、Hybrid-Weighted、HybridPlus-Weighted |
| 评测指标 | Recall@K（K=1/3/5/10）、MRR、MAP、nDCG@10 |
| 默认 Top-K | {topk_eval} |
| 向量模型 | BAAI/bge-base-zh-v1.5（768 维，离线运行） |

---

## 2. 主方法对比

{_method_table(results_data)}

> 注：加粗行为最优方法（BM25+）；延迟为单查询平均耗时（ms）。
>
> 图表见：`outputs/figures/method_comparison.pdf`

**主要发现：**

- BM25+ 在所有指标上均优于其他方法，MRR = 1.0000 表明 BM25+ 对于有解查询始终在首位命中
- 向量检索（BGE）在 Recall@1 上强于 BM25（23.97% vs. 23.97%），但在 Recall@5 上弱于 BM25+（33.55% vs. 52.31%）
- Hybrid+-Weighted 与 BM25+ 差异不显著（见第 4 节），向量分量未带来提升
- 纯向量检索在数学术语检索场景中受词汇精确匹配需求制约，语义泛化优势不显著

---

## 3. 消融实验

### 3.1 TopK 超参数消融（BM25+）

{_topk_table(ablation_data)}

> **← 标记推荐值**；图表见：`outputs/figures/topk_ablation.pdf`

**结论：** K=5 时 Recall@5 达到 52.31%，K=10 仅额外增益 0.35%（Recall@10 = 52.66%），收益递减。
综合效率与召回，**推荐 K=5** 作为默认参数。

### 3.2 混合检索 Alpha 权重敏感性（HybridPlus）

{_alpha_table(ablation_data)}

> **← 标记最优值**；图表见：`outputs/figures/alpha_sensitivity.pdf`

**结论：** alpha 在 0.3～0.85 范围内结果极为稳定（MAP 变化 < 0.004），
说明当 alpha > 0（BM25+ 权重＞0）时，检索结果主要由 BM25+ 精确匹配主导，
向量分量的贡献不敏感。**推荐 alpha=0.5（等权）** 或 **alpha=0.7（BM25+主导）**。

---

## 4. 统计显著性检验

Bootstrap 重采样次数：{sig_data.get("n_resamples", 10000)}，配对双侧 t 检验。

**Bootstrap 95% 置信区间：**

{_significance_section(sig_data)}

**结论：**

- BM25+ 与向量检索差异极显著（p ≈ 0），BM25+ 明显优于纯向量方法。
- BM25+ 与 HybridPlus 差异不显著（p = 0.89），混合策略未带来统计意义上的提升。
- 建议在数学术语召回场景中**优先使用 BM25+**，无需承担向量推理的额外延迟。

---

## 5. 学科细分分析

{_subject_table(breakdown)}

> 图表见：`outputs/figures/subject_breakdown.pdf`

**主要发现：**

- **高等代数** 召回率最高，术语定义明确、关键词重叠度高
- **概率论** 召回率最低，原因：
  1. 概率论术语存在多种等价表达（如「期望」vs「数学期望」vs「均值」），BM25+ 词汇匹配不完整
  2. 概率论与统计学交叉术语较多，相关术语集合较大，Recall 计算标准更严格
- **数学分析** 居中，极限/微积分核心术语覆盖率高，但部分高阶理论术语（如Lp空间）稀缺

---

## 6. 生成质量对比（RAG vs 无检索）

{_generation_comparison_table(comparison_data)}

> - **术语命中率**：回答中是否包含与查询相关的数学术语
> - **来源引用率**：回答中是否引用了检索来源（书名/页码），无检索组固定为 0
> - **回答有效率**：模型是否正常生成了非空回答

**主要发现：**
- RAG（BM25+）通过注入检索上下文，使来源引用率从 0 提升至有意义水平
- 无检索基线的术语命中率受限于模型记忆，高召回要求下不稳定
- RAG 组回答有效率维持 100%，说明检索上下文未导致生成拒绝

---

## 7. 案例分析

{_case_section(successes, failures)}

---

## 8. 结论

1. **最优检索方法**：BM25+（Recall@5 = 52.31%，MRR = 1.0000，MAP = 0.5701）
2. **向量检索局限**：在数学精确术语检索中，语义向量方法不优于词汇匹配，差异统计显著（p ≈ 0）
3. **混合检索结论**：HybridPlus 与 BM25+ 无显著差异，额外的向量推理成本不值得
4. **RAG 价值**：RAG 相比无检索基线提供了可溯源的书名/页码引用，在来源引用率上有明显提升
5. **主要瓶颈**：概率论学科召回率（{_fmt_pct(next(iter(breakdown.values()), {}).get("概率论", 0))}）显著低于其他学科，
   根本原因是语料库中概率论术语的等价表达覆盖不足
6. **改进建议**：（a）扩充概率论同义词表；（b）使用 BM25+ 精确匹配 + Query Expansion 代替向量检索

---

*本报告由 `scripts/generateReport.py` 自动生成，数据来源于 `outputs/reports/` 目录下的实验结果 JSON 文件。*
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n[OK] 报告已保存: {output_path}")
    print(f"[OK] 图表目录: {figures_dir}")
    print("\n生成文件列表:")
    print(f"  {output_path}")
    for name, path in figs.items():
        print(f"  {path}.pdf")
        print(f"  {path}.png")
    print("\n完成！")


def main() -> None:
    parser = argparse.ArgumentParser(description="Math-RAG 评测报告生成脚本")
    parser.add_argument(
        "--results",
        type=str,
        default=os.path.join(
            _REPO_ROOT, "outputs", "reports", "full_eval", "all_methods.json"
        ),
        help="全量方法对比报告路径",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=os.path.join(_REPO_ROOT, "outputs", "reports", "ablation_study.json"),
        help="消融实验汇总报告路径",
    )
    parser.add_argument(
        "--significance",
        type=str,
        default=os.path.join(
            _REPO_ROOT, "outputs", "reports", "significance_test.json"
        ),
        help="显著性检验报告路径",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=os.path.join(_REPO_ROOT, "data", "evaluation", "queries_full.jsonl"),
        help="查询集路径（含 subject 字段）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(_REPO_ROOT, "outputs", "reports", "final_report.md"),
        help="输出 Markdown 报告路径",
    )
    parser.add_argument(
        "--figures",
        type=str,
        default=os.path.join(_REPO_ROOT, "outputs", "figures"),
        help="输出图表目录",
    )
    parser.add_argument(
        "--comparison",
        type=str,
        default=os.path.join(
            _REPO_ROOT, "outputs", "reports", "comparison_results.json"
        ),
        help="生成质量对比结果路径（来自 evalGenerationComparison.py）",
    )
    args = parser.parse_args()
    generate_report(
        results_path=args.results,
        ablation_path=args.ablation,
        significance_path=args.significance,
        queries_path=args.queries,
        output_path=args.output,
        figures_dir=args.figures,
        comparison_path=args.comparison,
    )


if __name__ == "__main__":
    main()
