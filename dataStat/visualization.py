"""数据统计可视化逻辑。"""

import os
import warnings
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib 未安装，将跳过可视化功能")


if HAS_MATPLOTLIB:
    from pathlib import Path

    import matplotlib.font_manager as fm

    # WSL 场景下，优先注册 Windows 字体目录中的中文字体文件。
    windows_font_candidates = [
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/msyhbd.ttc",
        "/mnt/c/Windows/Fonts/msyhl.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
        "/mnt/c/Windows/Fonts/simsun.ttc",
        "/mnt/c/Windows/Fonts/Deng.ttf",
    ]
    for font_path in windows_font_candidates:
        if Path(font_path).exists():
            try:
                fm.fontManager.addfont(font_path)
            except Exception:
                pass

    # 统一通过 rcParams 设置中文字体优先级，并仅使用系统可用字体。
    preferred_cn_fonts = [
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "Droid Sans Fallback",
    ]
    available_names = {font.name for font in fm.fontManager.ttflist}
    selected_fonts = [name for name in preferred_cn_fonts if name in available_names]
    if not selected_fonts:
        warnings.warn("未检测到可用中文字体，中文可能显示异常。")
    selected_fonts.append("DejaVu Sans")

    plt.rcParams["font.sans-serif"] = selected_fonts
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.figsize"] = (12, 8)


def createVisualization(stats: dict[str, Any], outputDir: str) -> None:
    """生成可视化图表。"""
    if not HAS_MATPLOTLIB:
        print("  跳过可视化：matplotlib 未安装")
        return

    print("\n 生成可视化图表...")
    vizDir = os.path.join(outputDir, "visualizations")
    os.makedirs(vizDir, exist_ok=True)

    createBookDistributionChart(stats, vizDir)
    createSubjectDistributionChart(stats, vizDir)
    createFieldCoverageChart(stats, vizDir)
    createTermLengthDistribution(stats, vizDir)
    createDefinitionTypeChart(stats, vizDir)
    createComprehensiveDashboard(stats, vizDir)

    print(f" 可视化图表已保存到: {vizDir}")


def createBookDistributionChart(stats: dict[str, Any], outputDir: str) -> None:
    """书籍术语分布柱状图。"""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))

        books = list(stats["byBook"].keys())
        counts = [stats["byBook"][book]["count"] for book in books]

        shortNames = [
            book.replace("(第5版)", "").replace("(第五版)", "").replace("第三版", "")
            for book in books
        ]

        colors = plt.cm.Set3(range(len(books)))
        bars = ax.bar(
            range(len(books)), counts, color=colors, edgecolor="black", linewidth=1.2
        )

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

        ax.set_xlabel("教材名称", fontsize=14, fontweight="bold")
        ax.set_ylabel("术语数量", fontsize=14, fontweight="bold")
        ax.set_title("各教材术语数量分布", fontsize=16, fontweight="bold", pad=20)
        ax.set_xticks(range(len(books)))
        ax.set_xticklabels(shortNames, rotation=15, ha="right", fontsize=10)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()
        plt.savefig(os.path.join(outputDir, "1_书籍术语分布.png"), bbox_inches="tight")
        plt.close()

        print("   书籍术语分布图")
    except Exception as e:
        print(f"   书籍术语分布图生成失败: {e}")


def createSubjectDistributionChart(stats: dict[str, Any], outputDir: str) -> None:
    """学科分布饼图。"""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))

        subjects = list(stats["bySubject"].keys())
        counts = [stats["bySubject"][s] for s in subjects]

        filteredData = [(s, c) for s, c in zip(subjects, counts) if c > 10]
        subjects, counts = zip(*filteredData) if filteredData else ([], [])

        colors = plt.cm.Pastel1(range(len(subjects)))
        explode = [0.05] * len(subjects)

        wedges, texts, autotexts = ax.pie(
            counts,
            labels=subjects,
            autopct="%1.1f%%",
            colors=colors,
            explode=explode,
            startangle=90,
            textprops={"fontsize": 12, "fontweight": "bold"},
        )

        for i, (subject, count) in enumerate(zip(subjects, counts)):
            texts[i].set_text(f"{subject}\n({count}个)")

        ax.set_title("学科术语分布", fontsize=16, fontweight="bold", pad=20)

        plt.tight_layout()
        plt.savefig(os.path.join(outputDir, "2_学科分布.png"), bbox_inches="tight")
        plt.close()

        print("   学科分布图")
    except Exception as e:
        print(f"   学科分布图生成失败: {e}")


def createFieldCoverageChart(stats: dict[str, Any], outputDir: str) -> None:
    """字段覆盖率横向柱状图。"""
    try:
        fig, ax = plt.subplots(figsize=(12, 8))

        total = stats["summary"]["validFiles"]
        fieldNames = []
        coverageRates = []

        for field, fieldStats in stats["fields"].items():
            rate = fieldStats["present"] / total * 100 if total > 0 else 0
            fieldNames.append(field)
            coverageRates.append(rate)

        sortedData = sorted(
            zip(fieldNames, coverageRates), key=lambda x: x[1], reverse=True
        )
        fieldNames, coverageRates = zip(*sortedData)

        colors = [
            "#2ecc71" if r >= 95 else "#f39c12" if r >= 90 else "#e74c3c"
            for r in coverageRates
        ]

        bars = ax.barh(
            range(len(fieldNames)),
            coverageRates,
            color=colors,
            edgecolor="black",
            linewidth=1,
        )

        for bar, rate in zip(bars, coverageRates):
            ax.text(
                rate + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}%",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xlabel("覆盖率 (%)", fontsize=14, fontweight="bold")
        ax.set_ylabel("字段名称", fontsize=14, fontweight="bold")
        ax.set_title("字段覆盖率统计", fontsize=16, fontweight="bold", pad=20)
        ax.set_yticks(range(len(fieldNames)))
        ax.set_yticklabels(fieldNames, fontsize=10)
        ax.set_xlim(0, 105)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.axvline(
            x=95,
            color="green",
            linestyle="--",
            alpha=0.5,
            linewidth=2,
            label="优秀线 (95%)",
        )
        ax.legend(loc="lower right", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(outputDir, "3_字段覆盖率.png"), bbox_inches="tight")
        plt.close()

        print("   字段覆盖率图")
    except Exception as e:
        print(f"   字段覆盖率图生成失败: {e}")


def createTermLengthDistribution(stats: dict[str, Any], outputDir: str) -> None:
    """术语长度分布直方图。"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        termLengths = stats["termLengths"]
        ax1.hist(
            termLengths,
            bins=range(0, max(termLengths) + 2),
            edgecolor="black",
            color="skyblue",
            alpha=0.7,
        )
        ax1.axvline(
            np.mean(termLengths),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"平均值: {np.mean(termLengths):.1f}",
        )
        ax1.axvline(
            np.median(termLengths),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"中位数: {np.median(termLengths):.1f}",
        )
        ax1.set_xlabel("术语长度（字符数）", fontsize=12, fontweight="bold")
        ax1.set_ylabel("术语数量", fontsize=12, fontweight="bold")
        ax1.set_title("术语长度分布", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3, linestyle="--")

        defLengths = stats["definitions"]["textLengths"]
        ax2.hist(defLengths, bins=50, edgecolor="black", color="lightcoral", alpha=0.7)
        ax2.axvline(
            np.mean(defLengths),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"平均值: {np.mean(defLengths):.0f}",
        )
        ax2.axvline(
            np.median(defLengths),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"中位数: {np.median(defLengths):.0f}",
        )
        ax2.set_xlabel("定义文本长度（字符数）", fontsize=12, fontweight="bold")
        ax2.set_ylabel("定义数量", fontsize=12, fontweight="bold")
        ax2.set_title("定义文本长度分布", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3, linestyle="--")

        plt.tight_layout()
        plt.savefig(os.path.join(outputDir, "4_长度分布.png"), bbox_inches="tight")
        plt.close()

        print("   长度分布图")
    except Exception as e:
        print(f"   长度分布图生成失败: {e}")


def createDefinitionTypeChart(stats: dict[str, Any], outputDir: str) -> None:
    """定义类型分布图。"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        types = list(stats["definitions"]["types"].keys())
        counts = [stats["definitions"]["types"][t] for t in types]

        colors = plt.cm.Set2(range(len(types)))
        bars = ax.bar(types, counts, color=colors, edgecolor="black", linewidth=1.2)

        total = sum(counts)
        for bar in bars:
            height = bar.get_height()
            percentage = height / total * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}\n({percentage:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_xlabel("定义类型", fontsize=14, fontweight="bold")
        ax.set_ylabel("定义数量", fontsize=14, fontweight="bold")
        ax.set_title("定义类型分布", fontsize=16, fontweight="bold", pad=20)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()
        plt.savefig(os.path.join(outputDir, "5_定义类型分布.png"), bbox_inches="tight")
        plt.close()

        print("   定义类型分布图")
    except Exception as e:
        print(f"   定义类型分布图生成失败: {e}")


def createComprehensiveDashboard(stats: dict[str, Any], outputDir: str) -> None:
    """综合统计面板。"""
    try:
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis("off")
        summaryText = f"""
        数学术语数据统计报告

        总术语数: {stats["summary"]["totalTerms"]:,} 个    |    有效文件: {stats["summary"]["validFiles"]:,}    |    无效文件: {stats["summary"]["invalidFiles"]}

        平均术语长度: {np.mean(stats["termLengths"]):.1f} 字符    |    平均定义数: {stats["definitions"]["totalCount"] / stats["summary"]["validFiles"]:.1f} 个/术语
        """
        ax1.text(
            0.5,
            0.5,
            summaryText,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax2 = fig.add_subplot(gs[1, 0])
        books = list(stats["byBook"].keys())
        counts = [stats["byBook"][book]["count"] for book in books]
        shortNames = [book.split("(")[0][:8] for book in books]
        ax2.bar(range(len(books)), counts, color=plt.cm.Set3(range(len(books))))
        ax2.set_xticks(range(len(books)))
        ax2.set_xticklabels(shortNames, rotation=45, ha="right", fontsize=9)
        ax2.set_title("书籍分布", fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

        ax3 = fig.add_subplot(gs[1, 1])
        subjects = [s for s, c in stats["bySubject"].items() if c > 10]
        subCounts = [stats["bySubject"][s] for s in subjects]
        ax3.pie(
            subCounts,
            labels=subjects,
            autopct="%1.1f%%",
            colors=plt.cm.Pastel1(range(len(subjects))),
        )
        ax3.set_title("学科分布", fontweight="bold")

        ax4 = fig.add_subplot(gs[1, 2])
        total = stats["summary"]["validFiles"]
        fieldRates = [
            (f, s["present"] / total * 100) for f, s in stats["fields"].items()
        ]
        fieldRates.sort(key=lambda x: x[1], reverse=True)
        topFields = fieldRates[:10]
        fields, rates = zip(*topFields)
        colors = [
            "#2ecc71" if r >= 95 else "#f39c12" if r >= 90 else "#e74c3c" for r in rates
        ]
        ax4.barh(range(len(fields)), rates, color=colors)
        ax4.set_yticks(range(len(fields)))
        ax4.set_yticklabels(fields, fontsize=9)
        ax4.set_xlim(0, 105)
        ax4.set_title("字段覆盖率 TOP10", fontweight="bold")
        ax4.grid(axis="x", alpha=0.3)

        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(
            stats["termLengths"],
            bins=range(0, max(stats["termLengths"]) + 2),
            edgecolor="black",
            color="skyblue",
            alpha=0.7,
        )
        ax5.set_xlabel("术语长度")
        ax5.set_ylabel("数量")
        ax5.set_title("术语长度分布", fontweight="bold")
        ax5.grid(alpha=0.3)

        ax6 = fig.add_subplot(gs[2, 1])
        ax6.hist(
            stats["definitions"]["textLengths"],
            bins=50,
            edgecolor="black",
            color="lightcoral",
            alpha=0.7,
        )
        ax6.set_xlabel("定义文本长度")
        ax6.set_ylabel("数量")
        ax6.set_title("定义文本长度分布", fontweight="bold")
        ax6.grid(alpha=0.3)

        ax7 = fig.add_subplot(gs[2, 2])
        types = list(stats["definitions"]["types"].keys())
        typeCounts = [stats["definitions"]["types"][t] for t in types]
        ax7.bar(types, typeCounts, color=plt.cm.Set2(range(len(types))))
        ax7.set_title("定义类型分布", fontweight="bold")
        ax7.grid(axis="y", alpha=0.3)

        plt.suptitle("数学术语数据综合统计面板", fontsize=18, fontweight="bold", y=0.98)
        plt.savefig(os.path.join(outputDir, "0_综合统计面板.png"), bbox_inches="tight")
        plt.close()

        print("   综合统计面板")
    except Exception as e:
        print(f"   综合统计面板生成失败: {e}")
