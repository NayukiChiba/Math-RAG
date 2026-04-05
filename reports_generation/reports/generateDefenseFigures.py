"""
开题答辩图表生成脚本

用途：基于项目真实数据，生成一系列用于开题答辩的可视化图表。
输出目录：outputs/figures/defense/

使用方法：
    python scripts/generateDefenseFigures.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from core import config

# ── 初始化路径（与 [paths] + [reports_generation] 对齐）──────────────
_paths = config.getPathsConfig()
_rg = config.getReportsGenerationConfig()

STATS_FILE = os.path.join(_paths["stats_dir"], _rg["chunk_statistics_basename"])
CORPUS_FILE = os.path.join(_paths["processed_dir"], _rg["corpus_relpath"])
QUERIES_FILE = os.path.join(_paths["evaluation_dir"], _rg["queries_basename"])
QUERIES_FULL_FILE = os.path.join(_paths["evaluation_dir"], _rg["queries_full_basename"])
GOLDEN_SET_FILE = os.path.join(_paths["evaluation_dir"], _rg["golden_set_basename"])
TERM_MAPPING_FILE = os.path.join(_paths["evaluation_dir"], _rg["term_mapping_basename"])
OUTPUT_DIR = os.path.join(_paths["figures_dir"], _rg["defense_output_subdir"])
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 中文字体配置 ──────────────────────────────────────────────

_font_path = str(_rg.get("defense_cjk_font_file") or "").strip()
if _font_path and os.path.isfile(_font_path):
    fm.fontManager.addfont(_font_path)
    _font_prop = fm.FontProperties(fname=_font_path)
    _font_name = _font_prop.get_name()
    plt.rcParams["font.sans-serif"] = [_font_name] + plt.rcParams["font.sans-serif"]
else:
    plt.rcParams["font.sans-serif"] = list(
        _rg["defense_matplotlib_fallback_fonts"]
    ) + list(plt.rcParams["font.sans-serif"])
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = _rg["defense_matplotlib_font_size"]

# ── 配色方案 ────────────────────────────────────────────────
COLORS = dict(_rg["defense_colors"])
PALETTE = list(_rg["defense_palette"])
GRADIENT_BLUES = list(_rg["defense_gradient_blues"])
GRADIENT_MULTI = list(_rg["defense_gradient_multi"])


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(
        path,
        dpi=_rg["defense_save_dpi"],
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    print(f"  ✅ {name}")


def buildBarColors(baseColors, count, tailColor=None):
    """构造与柱状图数量严格一致的颜色列表。"""
    if count <= 0:
        return []

    colors = list(baseColors)
    if not colors:
        colors = [COLORS["primary"]]

    while len(colors) < count:
        colors.extend(baseColors if baseColors else colors)

    colors = colors[:count]
    if tailColor is not None:
        colors[-1] = tailColor
    return colors


# ═══════════════════════════════════════════════════════════
#  图 1：各教材术语数量分布（饼图 + 柱状图组合）
# ═══════════════════════════════════════════════════════════
def fig01_terms_by_book(stats):
    by_book = stats["byBook"]
    short_names = {
        "数学分析(第5版)上(华东师范大学数学系)": "数学分析(上)",
        "数学分析(第5版)下(华东师范大学数学系)": "数学分析(下)",
        "高等代数(第五版)(王萼芳石生明)": "高等代数",
        "概率论与数理统计教程第三版(茆诗松)": "概率论与数理统计",
    }
    names = [short_names.get(k, k) for k in by_book]
    counts = [v["count"] for v in by_book.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 饼图
    wedges, texts, autotexts = ax1.pie(
        counts,
        labels=names,
        autopct="%1.1f%%",
        colors=PALETTE[: len(counts)],
        startangle=90,
        textprops={"fontsize": 11},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_fontsize(12)
    ax1.set_title("各教材术语占比", fontsize=15, fontweight="bold", pad=15)

    # 柱状图
    bars = ax2.bar(
        names,
        counts,
        color=PALETTE[: len(counts)],
        edgecolor="white",
        linewidth=1.5,
        width=0.6,
    )
    for bar, count in zip(bars, counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            str(count),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax2.set_ylabel("术语数量", fontsize=13)
    ax2.set_title("各教材术语数量", fontsize=15, fontweight="bold", pad=15)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="x", labelsize=10)
    ax2.set_ylim(0, max(counts) * 1.15)

    fig.suptitle(
        f"Math-RAG 知识库统计（共 {sum(counts)} 个术语）",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "01_terms_by_book.png")


# ═══════════════════════════════════════════════════════════
#  图 2：各学科术语数量对比
# ═══════════════════════════════════════════════════════════
def fig02_terms_by_subject(stats):
    by_subject = stats["bySubject"]
    # 合并错别字
    merged = {}
    for k, v in by_subject.items():
        clean = k.replace("概率论与数理理统计", "概率论与数理统计")
        merged[clean] = merged.get(clean, 0) + v
    subjects = list(merged.keys())
    counts = list(merged.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        subjects, counts, color=PALETTE[: len(subjects)], edgecolor="white", height=0.5
    )
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + 15,
            bar.get_y() + bar.get_height() / 2,
            f"{count} ({count / sum(counts) * 100:.1f}%)",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_xlabel("术语数量", fontsize=13)
    ax.set_title("各学科术语数量对比", fontsize=16, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(counts) * 1.3)
    fig.tight_layout()
    save_fig(fig, "02_terms_by_subject.png")


# ═══════════════════════════════════════════════════════════
#  图 3：字段覆盖率热力图
# ═══════════════════════════════════════════════════════════
def fig03_field_coverage(stats):
    coverage = stats["fieldCoverage"]
    # 选取关键字段
    fields_order = [
        "term",
        "definitions",
        "formula",
        "usage",
        "applications",
        "notation",
        "aliases",
        "disambiguation",
        "related_terms",
        "sources",
        "search_keys",
        "confidence",
    ]
    field_names_cn = {
        "term": "术语名",
        "definitions": "定义",
        "formula": "公式",
        "usage": "用法",
        "applications": "应用",
        "notation": "符号",
        "aliases": "别名",
        "disambiguation": "消歧",
        "related_terms": "关联术语",
        "sources": "来源",
        "search_keys": "搜索关键词",
        "confidence": "置信度",
    }

    names = [field_names_cn.get(f, f) for f in fields_order]
    rates = [coverage[f]["coverageRate"] for f in fields_order]
    missing = [coverage[f]["missing"] for f in fields_order]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(
        names,
        rates,
        color=[
            COLORS["accent"]
            if r >= 99
            else COLORS["primary"]
            if r >= 90
            else COLORS["warm"]
            if r >= 70
            else COLORS["danger"]
            for r in rates
        ],
        edgecolor="white",
        linewidth=1.5,
        width=0.6,
    )

    for bar, rate, miss in zip(bars, rates, missing):
        label = f"{rate:.1f}%"
        if miss > 0:
            label += f"\n(缺{miss})"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.axhline(y=100, color="#CBD5E1", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(y=90, color="#FDE68A", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylabel("覆盖率 (%)", fontsize=13)
    ax.set_title("术语数据字段覆盖率", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylim(0, 108)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=11, rotation=30)

    # 图例
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=COLORS["accent"], label="≥ 99%"),
        Patch(facecolor=COLORS["primary"], label="90-99%"),
        Patch(facecolor=COLORS["warm"], label="70-90%"),
        Patch(facecolor=COLORS["danger"], label="< 70%"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "03_field_coverage.png")


# ═══════════════════════════════════════════════════════════
#  图 4：定义类型分布 + 定义长度箱线图
# ═══════════════════════════════════════════════════════════
def fig04_definitions_analysis(stats):
    defs = stats["definitionsAnalysis"]
    types = defs["types"]
    text_len = defs["textLength"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 定义类型分布
    type_names_cn = {
        "strict": "严格定义",
        "alternative": "替代定义",
        "informal": "非正式定义",
    }
    labels = [type_names_cn.get(k, k) for k in types]
    values = list(types.values())
    wedges, texts, autotexts = ax1.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=[COLORS["primary"], COLORS["secondary"], COLORS["warm"]],
        startangle=140,
        textprops={"fontsize": 12},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontweight("bold")
    ax1.set_title(
        f"定义类型分布（共 {defs['totalDefinitions']} 条）",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # 定义长度分位数图（模拟箱线图数据）
    percentiles = [
        text_len["min"],
        text_len["p25"],
        text_len["p50"],
        text_len["p75"],
        text_len["p90"],
        text_len["p95"],
        text_len["max"],
    ]
    perc_labels = ["Min", "P25", "P50\n(中位数)", "P75", "P90", "P95", "Max"]
    bars2 = ax2.bar(
        perc_labels,
        percentiles,
        color=GRADIENT_BLUES[:7]
        if len(GRADIENT_BLUES) >= 7
        else GRADIENT_BLUES + GRADIENT_BLUES,
        edgecolor="white",
        width=0.6,
    )
    for bar, val in zip(bars2, percentiles):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(val),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax2.set_ylabel("字符数", fontsize=13)
    ax2.set_title(
        f"定义文本长度分布（均值 {text_len['mean']:.0f} 字符）",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "04_definitions_analysis.png")


# ═══════════════════════════════════════════════════════════
#  图 5：术语名称长度分布直方图
# ═══════════════════════════════════════════════════════════
def fig05_term_length_distribution(stats):
    tld = stats["termLengthDistribution"]
    fig, ax = plt.subplots(figsize=(10, 6))

    percentiles = {
        "Min": tld["min"],
        "P25": tld["p25"],
        "P50": tld["p50"],
        "P75": tld["p75"],
        "P90": tld["p90"],
        "P95": tld["p95"],
        "P99": tld["p99"],
        "Max": tld["max"],
    }

    bars = ax.bar(
        percentiles.keys(),
        percentiles.values(),
        color=buildBarColors(
            baseColors=GRADIENT_BLUES,
            count=len(percentiles),
            tailColor=COLORS["danger"],
        ),
        edgecolor="white",
        width=0.55,
    )
    for bar, val in zip(bars, percentiles.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            str(val),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.axhline(
        y=tld["mean"],
        color=COLORS["danger"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"均值 = {tld['mean']:.1f}",
    )
    ax.set_ylabel("字符数", fontsize=13)
    ax.set_title("术语名称长度分布", fontsize=16, fontweight="bold", pad=15)
    ax.legend(fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "05_term_length_dist.png")


# ═══════════════════════════════════════════════════════════
#  图 6：关键字段长度分布（分位数对比雷达图）
# ═══════════════════════════════════════════════════════════
def fig06_field_length_radar(stats):
    fld = stats["fieldLengthDistribution"]
    fields = ["usage", "applications", "disambiguation", "notation"]
    field_cn = {
        "usage": "用法",
        "applications": "应用",
        "disambiguation": "消歧",
        "notation": "符号",
    }

    means = [fld[f]["mean"] for f in fields]
    p50s = [fld[f]["p50"] for f in fields]
    p90s = [fld[f]["p90"] for f in fields]
    labels = [field_cn[f] for f in fields]

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(fields))
    width = 0.25

    bars1 = ax.bar(
        x - width,
        means,
        width,
        label="均值",
        color=COLORS["primary"],
        edgecolor="white",
    )
    bars2 = ax.bar(
        x, p50s, width, label="中位数 (P50)", color=COLORS["accent"], edgecolor="white"
    )
    bars3 = ax.bar(
        x + width, p90s, width, label="P90", color=COLORS["warm"], edgecolor="white"
    )

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1,
                f"{h:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("字符数", fontsize=13)
    ax.set_title("关键字段文本长度对比", fontsize=16, fontweight="bold", pad=15)
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "06_field_length_comparison.png")


# ═══════════════════════════════════════════════════════════
#  图 7：语料库文本长度分布（基于 corpus.jsonl）
# ═══════════════════════════════════════════════════════════
def fig07_corpus_text_length(corpus):
    text_lens = [len(item.get("text", "")) for item in corpus]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 直方图
    ax1.hist(
        text_lens,
        bins=50,
        color=COLORS["primary"],
        edgecolor="white",
        alpha=0.85,
        linewidth=0.8,
    )
    ax1.axvline(
        np.mean(text_lens),
        color=COLORS["danger"],
        linestyle="--",
        linewidth=2,
        label=f"均值 = {np.mean(text_lens):.0f}",
    )
    ax1.axvline(
        np.median(text_lens),
        color=COLORS["warm"],
        linestyle="--",
        linewidth=2,
        label=f"中位数 = {np.median(text_lens):.0f}",
    )
    ax1.set_xlabel("文本长度（字符）", fontsize=13)
    ax1.set_ylabel("频次", fontsize=13)
    ax1.set_title("语料库文档文本长度分布", fontsize=15, fontweight="bold", pad=15)
    ax1.legend(fontsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # 按学科的箱线图
    subjects_data = {}
    for item in corpus:
        subj = item.get("subject", "未知")
        text_len = len(item.get("text", ""))
        subjects_data.setdefault(subj, []).append(text_len)

    subj_names = sorted(subjects_data.keys())
    subj_values = [subjects_data[s] for s in subj_names]
    bp = ax2.boxplot(
        subj_values,
        labels=subj_names,
        patch_artist=True,
        medianprops={"color": COLORS["danger"], "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel("文本长度（字符）", fontsize=13)
    ax2.set_title("各学科语料文本长度对比", fontsize=15, fontweight="bold", pad=15)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="x", labelsize=10)
    fig.tight_layout()
    save_fig(fig, "07_corpus_text_length.png")


# ═══════════════════════════════════════════════════════════
#  图 8：评测查询难度分布
# ═══════════════════════════════════════════════════════════
def fig08_query_difficulty(golden_set):
    difficulty_counts = {}
    subject_counts = {}
    for item in golden_set:
        diff = item.get("difficulty", "unknown")
        subj = item.get("subject", "未知")
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        subject_counts[subj] = subject_counts.get(subj, 0) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 难度分布
    diff_cn = {"easy": "简单", "medium": "中等", "hard": "困难"}
    diff_order = ["easy", "medium", "hard"]
    diff_labels = [diff_cn.get(d, d) for d in diff_order if d in difficulty_counts]
    diff_values = [difficulty_counts[d] for d in diff_order if d in difficulty_counts]
    diff_colors = [COLORS["accent"], COLORS["warm"], COLORS["danger"]]

    wedges1, _, autotexts1 = ax1.pie(
        diff_values,
        labels=diff_labels,
        autopct="%1.1f%%",
        colors=diff_colors[: len(diff_values)],
        startangle=90,
        textprops={"fontsize": 13},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts1:
        t.set_fontweight("bold")
        t.set_fontsize(13)
    ax1.set_title(
        f"评测查询难度分布（共 {len(golden_set)} 条）",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # 学科分布
    subj_labels = list(subject_counts.keys())
    subj_values = list(subject_counts.values())
    bars = ax2.bar(
        subj_labels,
        subj_values,
        color=PALETTE[: len(subj_labels)],
        edgecolor="white",
        width=0.5,
    )
    for bar, val in zip(bars, subj_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(val),
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
    ax2.set_ylabel("查询数量", fontsize=13)
    ax2.set_title("评测查询学科分布", fontsize=14, fontweight="bold", pad=15)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "08_query_difficulty.png")


# ═══════════════════════════════════════════════════════════
#  图 9：检索策略对比（模拟数据 - 标注需替换）
# ═══════════════════════════════════════════════════════════
def fig09_retrieval_strategy_comparison():
    """模拟检索策略对比实验结果。"""
    strategies = ["BM25", "BM25+", "Vector", "Hybrid\n(RRF)", "HybridPlus\n(加权)"]
    metrics = {
        "Recall@5": [0.62, 0.71, 0.65, 0.78, 0.85],
        "Recall@10": [0.74, 0.82, 0.76, 0.87, 0.92],
        "MRR": [0.55, 0.63, 0.58, 0.72, 0.79],
    }

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(strategies))
    width = 0.25
    colors = [COLORS["primary"], COLORS["accent"], COLORS["secondary"]]

    for i, (metric, values) in enumerate(metrics.items()):
        bars = ax.bar(
            x + i * width,
            values,
            width,
            label=metric,
            color=colors[i],
            edgecolor="white",
            linewidth=1,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(strategies, fontsize=12)
    ax.set_ylabel("得分", fontsize=13)
    ax.set_title("不同检索策略性能对比", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=12, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=0.8, color="#CBD5E1", linestyle=":", linewidth=1)

    # 注释
    ax.annotate(
        "⚠️ 模拟数据，请替换为真实实验结果",
        xy=(0.5, 0.01),
        xycoords="axes fraction",
        fontsize=9,
        color="#94A3B8",
        ha="center",
    )
    fig.tight_layout()
    save_fig(fig, "09_retrieval_strategy_comparison.png")


# ═══════════════════════════════════════════════════════════
#  图 10：混合检索权重敏感性分析（模拟数据）
# ═══════════════════════════════════════════════════════════
def fig10_hybrid_weight_sensitivity():
    """模拟混合检索 α/β 权重对性能的影响。"""
    alphas = np.arange(0.0, 1.05, 0.1)

    recall5 = [0.65, 0.70, 0.75, 0.79, 0.83, 0.85, 0.87, 0.88, 0.86, 0.82, 0.76]
    recall10 = [0.76, 0.80, 0.84, 0.87, 0.90, 0.92, 0.93, 0.93, 0.91, 0.88, 0.83]
    mrr = [0.58, 0.63, 0.67, 0.71, 0.75, 0.79, 0.78, 0.76, 0.73, 0.69, 0.62]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(
        alphas,
        recall5,
        "o-",
        color=COLORS["primary"],
        linewidth=2.5,
        markersize=7,
        label="Recall@5",
    )
    ax.plot(
        alphas,
        recall10,
        "s-",
        color=COLORS["accent"],
        linewidth=2.5,
        markersize=7,
        label="Recall@10",
    )
    ax.plot(
        alphas,
        mrr,
        "^-",
        color=COLORS["secondary"],
        linewidth=2.5,
        markersize=7,
        label="MRR",
    )

    # 标注最优点
    best_idx = np.argmax(recall5)
    ax.annotate(
        f"最优 α={alphas[best_idx]:.1f}",
        xy=(alphas[best_idx], recall5[best_idx]),
        xytext=(alphas[best_idx] + 0.12, recall5[best_idx] + 0.03),
        arrowprops=dict(arrowstyle="->", color=COLORS["danger"], lw=1.5),
        fontsize=11,
        fontweight="bold",
        color=COLORS["danger"],
    )

    ax.fill_between(alphas, recall5, alpha=0.08, color=COLORS["primary"])
    ax.set_xlabel("α (BM25 权重)    ←→    β = 1 - α (向量权重)", fontsize=13)
    ax.set_ylabel("得分", fontsize=13)
    ax.set_title("混合检索权重敏感性分析", fontsize=16, fontweight="bold", pad=15)
    ax.legend(fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.5, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2)
    ax.annotate(
        "⚠️ 模拟数据，请替换为真实实验结果",
        xy=(0.5, 0.01),
        xycoords="axes fraction",
        fontsize=9,
        color="#94A3B8",
        ha="center",
    )
    fig.tight_layout()
    save_fig(fig, "10_hybrid_weight_sensitivity.png")


# ═══════════════════════════════════════════════════════════
#  图 11：Top-K 参数对召回率的影响（模拟数据）
# ═══════════════════════════════════════════════════════════
def fig11_topk_recall_curve():
    """模拟不同 Top-K 下的召回率曲线。"""
    ks = [1, 3, 5, 10, 15, 20, 30, 50]
    recall_bm25 = [0.28, 0.45, 0.62, 0.74, 0.79, 0.82, 0.85, 0.87]
    recall_vec = [0.32, 0.48, 0.65, 0.76, 0.80, 0.83, 0.86, 0.88]
    recall_hybrid = [0.40, 0.58, 0.78, 0.87, 0.91, 0.93, 0.95, 0.96]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(
        ks,
        recall_bm25,
        "o-",
        color=COLORS["primary"],
        linewidth=2.5,
        markersize=8,
        label="BM25+",
    )
    ax.plot(
        ks,
        recall_vec,
        "s-",
        color=COLORS["secondary"],
        linewidth=2.5,
        markersize=8,
        label="Vector",
    )
    ax.plot(
        ks,
        recall_hybrid,
        "D-",
        color=COLORS["accent"],
        linewidth=2.5,
        markersize=8,
        label="HybridPlus",
    )

    ax.fill_between(ks, recall_hybrid, alpha=0.08, color=COLORS["accent"])
    ax.axhline(y=0.9, color="#CBD5E1", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Top-K", fontsize=13)
    ax.set_ylabel("Recall@K", fontsize=13)
    ax.set_title("Top-K 参数对召回率的影响", fontsize=16, fontweight="bold", pad=15)
    ax.legend(fontsize=12)
    ax.set_ylim(0.2, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2)
    ax.annotate(
        "⚠️ 模拟数据，请替换为真实实验结果",
        xy=(0.5, 0.01),
        xycoords="axes fraction",
        fontsize=9,
        color="#94A3B8",
        ha="center",
    )
    fig.tight_layout()
    save_fig(fig, "11_topk_recall_curve.png")


# ═══════════════════════════════════════════════════════════
#  图 12：数据处理流水线各阶段数量统计
# ═══════════════════════════════════════════════════════════
def fig12_pipeline_stats(stats, corpus):
    """展示数据从 PDF 到检索索引各阶段的数量。"""
    stages = ["教材 PDF", "OCR 页面", "抽取术语", "结构化 JSON", "语料文档", "评测查询"]
    # 基于真实数据推算
    n_books = len(stats["byBook"])
    n_terms = stats["summary"]["totalTerms"]
    n_corpus = len(corpus)
    n_queries_full = 3102  # from wc -l
    # 估算 OCR 页面数：每本书约 300-500 页
    n_pages_est = n_books * 400

    counts = [n_books, n_pages_est, n_terms, n_terms, n_corpus, n_queries_full]

    fig, ax = plt.subplots(figsize=(13, 7))
    colors_stages = GRADIENT_MULTI[: len(stages)]
    bars = ax.bar(
        stages, counts, color=colors_stages, edgecolor="white", linewidth=2, width=0.55
    )

    for bar, count in zip(bars, counts):
        if count < 100:
            label = str(count)
        elif count < 10000:
            label = f"{count:,}"
        else:
            label = f"{count:,}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            label,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    # 在柱间画箭头
    for i in range(len(stages) - 1):
        ax.annotate(
            "→",
            xy=(i + 0.4, max(counts) * 0.5),
            fontsize=18,
            ha="center",
            va="center",
            color="#94A3B8",
        )

    ax.set_ylabel("数量", fontsize=13)
    ax.set_title("数据处理流水线各阶段数据量", fontsize=16, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(counts) * 1.15)
    fig.tight_layout()
    save_fig(fig, "12_pipeline_stats.png")


# ═══════════════════════════════════════════════════════════
#  图 13：跨教材术语重复分析（基于真实数据）
# ═══════════════════════════════════════════════════════════
def fig13_duplicate_analysis(stats):
    dupes = stats.get("duplicates", {})
    dupe_count = dupes.get("count", 0)
    total = stats["summary"]["totalTerms"]
    unique = total - dupe_count

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 重复 vs 唯一
    ax1.pie(
        [unique, dupe_count],
        labels=[f"唯一术语\n{unique}", f"跨书重复\n{dupe_count}"],
        autopct="%1.1f%%",
        colors=[COLORS["primary"], COLORS["warm"]],
        startangle=90,
        textprops={"fontsize": 13},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    ax1.set_title("术语唯一性分析", fontsize=15, fontweight="bold", pad=15)

    # 重复术语跨书频次分布
    details = dupes.get("details", {})
    cross_counts = {}
    for term, entries in details.items():
        n = len(entries)
        cross_counts[n] = cross_counts.get(n, 0) + 1

    cross_labels = [f"{k} 本书" for k in sorted(cross_counts.keys())]
    cross_values = [cross_counts[k] for k in sorted(cross_counts.keys())]
    bars = ax2.bar(
        cross_labels,
        cross_values,
        color=PALETTE[: len(cross_labels)],
        edgecolor="white",
        width=0.5,
    )
    for bar, val in zip(bars, cross_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val),
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
    ax2.set_ylabel("术语数量", fontsize=13)
    ax2.set_title("重复术语跨书频次分布", fontsize=14, fontweight="bold", pad=15)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "13_duplicate_analysis.png")


# ═══════════════════════════════════════════════════════════
#  图 14：公式与关联术语数量分布
# ═══════════════════════════════════════════════════════════
def fig14_formula_related_terms(stats):
    fld = stats["fieldLengthDistribution"]
    formula = fld["formula"]
    related = fld["related_terms"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 公式数量分布
    formula_vals = [
        formula["min"],
        formula["p25"],
        formula["p50"],
        formula["p75"],
        formula["p90"],
        formula["p95"],
        formula["max"],
    ]
    perc_labels = ["Min", "P25", "P50", "P75", "P90", "P95", "Max"]
    bars1 = ax1.bar(
        perc_labels,
        formula_vals,
        color=buildBarColors(
            baseColors=GRADIENT_BLUES,
            count=len(formula_vals),
            tailColor=COLORS["danger"],
        ),
        edgecolor="white",
        width=0.55,
    )
    for bar, val in zip(bars1, formula_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(val),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax1.set_ylabel("数量", fontsize=13)
    ax1.set_title(
        f"每个术语的公式数量分布（均值 {formula['mean']:.1f}）",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # 关联术语数量分布
    related_vals = [
        related["min"],
        related["p25"],
        related["p50"],
        related["p75"],
        related["p90"],
        related["p95"],
        related["max"],
    ]
    bars2 = ax2.bar(
        perc_labels,
        related_vals,
        color=buildBarColors(
            baseColors=GRADIENT_BLUES,
            count=len(related_vals),
            tailColor=COLORS["danger"],
        ),
        edgecolor="white",
        width=0.55,
    )
    for bar, val in zip(bars2, related_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(val),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax2.set_ylabel("数量", fontsize=13)
    ax2.set_title(
        f"每个术语的关联术语数量（均值 {related['mean']:.1f}）",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig.tight_layout()
    save_fig(fig, "14_formula_related_terms.png")


# ═══════════════════════════════════════════════════════════
#  图 15：语料库学科 × 来源分布热力图
# ═══════════════════════════════════════════════════════════
def fig15_corpus_heatmap(corpus):
    source_subject = {}
    for item in corpus:
        src = item.get("source", "未知")
        subj = item.get("subject", "未知")
        # 简化来源名
        short_src = src
        for long, short in [
            ("数学分析(第5版)上(华东师范大学数学系)", "数分(上)"),
            ("数学分析(第5版)下(华东师范大学数学系)", "数分(下)"),
            ("高等代数(第五版)(王萼芳石生明)", "高代"),
            ("概率论与数理统计教程第三版(茆诗松)", "概率统计"),
        ]:
            if long in src:
                short_src = short
                break
        source_subject.setdefault(short_src, {}).setdefault(subj, 0)
        source_subject[short_src][subj] += 1

    sources = sorted(source_subject.keys())
    subjects = sorted(set(s for d in source_subject.values() for s in d))

    matrix = []
    for src in sources:
        row = [source_subject[src].get(subj, 0) for subj in subjects]
        matrix.append(row)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")

    ax.set_xticks(np.arange(len(subjects)))
    ax.set_yticks(np.arange(len(sources)))
    ax.set_xticklabels(subjects, fontsize=11)
    ax.set_yticklabels(sources, fontsize=11)

    for i in range(len(sources)):
        for j in range(len(subjects)):
            val = matrix[i, j]
            if val > 0:
                color = "white" if val > matrix.max() * 0.5 else "black"
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color=color,
                )

    ax.set_title("语料库 来源×学科 分布矩阵", fontsize=16, fontweight="bold", pad=15)
    fig.colorbar(im, ax=ax, label="术语数量", shrink=0.8)
    fig.tight_layout()
    save_fig(fig, "15_corpus_heatmap.png")


# ═══════════════════════════════════════════════════════════
#  图 16：系统各层数据规模汇总（多指标仪表盘风格）
# ═══════════════════════════════════════════════════════════
def fig16_system_dashboard(stats, corpus, golden_set):
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Math-RAG 系统数据概览", fontsize=20, fontweight="bold", y=1.01)

    metrics_list = [
        ("教材数量", len(stats["byBook"]), "本", COLORS["primary"]),
        ("术语总数", stats["summary"]["totalTerms"], "个", COLORS["secondary"]),
        ("语料文档", len(corpus), "条", COLORS["accent"]),
        (
            "定义总数",
            stats["definitionsAnalysis"]["totalDefinitions"],
            "条",
            COLORS["warm"],
        ),
        (
            "学科数量",
            len(set(s for s in stats["bySubject"] if "理理" not in s)),
            "个",
            COLORS["rose"],
        ),
        ("评测查询", len(golden_set), "条", COLORS["cyan"]),
        (
            "跨书重复术语",
            stats.get("duplicates", {}).get("count", 0),
            "个",
            COLORS["danger"],
        ),
        (
            "平均定义数/术语",
            stats["definitionsAnalysis"]["avgPerTerm"],
            "条",
            COLORS["slate"],
        ),
    ]

    for ax, (name, value, unit, color) in zip(axes.flat, metrics_list):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # 背景圆
        circle = plt.Circle((0.5, 0.55), 0.35, color=color, alpha=0.12)
        ax.add_patch(circle)

        # 数值
        if isinstance(value, float):
            val_text = f"{value:.1f}"
        else:
            val_text = f"{value:,}" if value >= 1000 else str(value)
        ax.text(
            0.5,
            0.6,
            val_text,
            ha="center",
            va="center",
            fontsize=28,
            fontweight="bold",
            color=color,
        )
        ax.text(0.5, 0.38, unit, ha="center", va="center", fontsize=13, color="#64748B")
        ax.text(
            0.5,
            0.12,
            name,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#334155",
        )

    fig.tight_layout()
    save_fig(fig, "16_system_dashboard.png")


# ═══════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Math-RAG 开题答辩图表生成")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 60)

    # 加载数据
    print("\n📂 加载数据...")
    stats = load_json(STATS_FILE)
    print(f"  chunkStatistics.json: {stats['summary']['totalTerms']} 个术语")

    corpus = load_jsonl(CORPUS_FILE)
    print(f"  corpus.jsonl: {len(corpus)} 条语料")

    golden_set = load_jsonl(GOLDEN_SET_FILE)
    print(f"  golden_set.jsonl: {len(golden_set)} 条评测查询")

    # 生成图表
    print("\n📊 生成图表...")

    fig01_terms_by_book(stats)
    fig02_terms_by_subject(stats)
    fig03_field_coverage(stats)
    fig04_definitions_analysis(stats)
    fig05_term_length_distribution(stats)
    fig06_field_length_radar(stats)
    fig07_corpus_text_length(corpus)
    fig08_query_difficulty(golden_set)
    fig09_retrieval_strategy_comparison()
    fig10_hybrid_weight_sensitivity()
    fig11_topk_recall_curve()
    fig12_pipeline_stats(stats, corpus)
    fig13_duplicate_analysis(stats)
    fig14_formula_related_terms(stats)
    fig15_corpus_heatmap(corpus)
    fig16_system_dashboard(stats, corpus, golden_set)

    print("\n✅ 全部完成！共生成 16 张图表")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("\n提示：图 9/10/11 为模拟数据，请运行实验后替换为真实结果。")


if __name__ == "__main__":
    main()
