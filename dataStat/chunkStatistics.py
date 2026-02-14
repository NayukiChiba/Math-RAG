"""
术语数据统计与可视化

功能：
1. 遍历 data/processed/chunk/ 下所有术语 JSON 文件
2. 统计字段缺失率、长度分布、学科覆盖等信息
3. 生成可视化图表
4. 输出统计报告到 data/stats/

使用方法：
    python -m dataStat.chunkStatistics
    或
    python dataStat/chunkStatistics.py
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import warnings

# 路径调整：添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# 导入可视化库
try:
    import matplotlib

    matplotlib.use("Agg")  # 非交互式后端
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib import rcParams
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib 未安装，将跳过可视化功能")


# 配置中文字体（支持 Windows 系统）
if HAS_MATPLOTLIB:
    # 尝试使用微软雅黑或其他中文字体
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.figsize"] = (12, 8)


def loadJsonFile(filepath: str) -> Dict[str, Any]:
    """加载 JSON 文件"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载文件失败: {filepath}, 错误: {e}")
        return None


def calculateFieldStats(data: Dict[str, Any], fieldName: str, fieldStats: Dict) -> None:
    """计算单个字段的统计信息"""
    if fieldName in data and data[fieldName]:
        fieldStats["present"] += 1
        value = data[fieldName]

        # 记录长度信息
        if isinstance(value, str):
            fieldStats["lengths"].append(len(value))
        elif isinstance(value, list):
            fieldStats["lengths"].append(len(value))
            # 如果是列表，统计列表中元素的长度
            if value and isinstance(value[0], str):
                for item in value:
                    fieldStats["itemLengths"].append(len(item))
            elif value and isinstance(value[0], dict):
                fieldStats["itemCounts"].append(len(value))
    else:
        fieldStats["missing"] += 1


def analyzeDefinitions(definitions: List[Dict]) -> Dict:
    """分析 definitions 字段的详细信息"""
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


def buildStatistics(chunkDir: str) -> Dict[str, Any]:
    """构建完整的统计信息"""

    # 统计结果结构
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
        "duplicates": defaultdict(list),  # 记录重复术语
    }

    # 定义需要统计的字段
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

    # 初始化字段统计
    for field in fieldsToCheck:
        stats["fields"][field] = {
            "present": 0,
            "missing": 0,
            "lengths": [],
            "itemLengths": [],
            "itemCounts": [],
        }

    # 遍历所有书籍目录
    if not os.path.exists(chunkDir):
        print(f"❌ 目录不存在: {chunkDir}")
        return stats

    for bookName in os.listdir(chunkDir):
        bookPath = os.path.join(chunkDir, bookName)
        if not os.path.isdir(bookPath):
            continue

        print(f"📖 处理书籍: {bookName}")

        # 遍历该书籍下的所有 JSON 文件
        jsonFiles = [f for f in os.listdir(bookPath) if f.endswith(".json")]

        for jsonFile in jsonFiles:
            filepath = os.path.join(bookPath, jsonFile)
            stats["summary"]["totalFiles"] += 1

            # 加载 JSON
            data = loadJsonFile(filepath)
            if data is None:
                stats["summary"]["invalidFiles"] += 1
                continue

            stats["summary"]["validFiles"] += 1
            stats["summary"]["totalTerms"] += 1

            # 记录书籍统计
            stats["byBook"][bookName]["count"] += 1

            # 统计学科
            subject = data.get("subject", "unknown")
            stats["bySubject"][subject] += 1
            stats["byBook"][bookName]["subjects"][subject] += 1

            # 统计术语长度
            term = data.get("term", "")
            if term:
                stats["termLengths"].append(len(term))
                # 检查重复术语
                stats["duplicates"][term].append(
                    {"book": bookName, "file": jsonFile, "subject": subject}
                )

            # 统计各字段
            for field in fieldsToCheck:
                calculateFieldStats(data, field, stats["fields"][field])

            # 特殊处理：definitions 的详细统计
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

    # 找出真正的重复术语（出现在多个地方）
    actualDuplicates = {
        term: locations
        for term, locations in stats["duplicates"].items()
        if len(locations) > 1
    }
    stats["duplicates"] = actualDuplicates

    return stats


def calculatePercentiles(
    values: List[float], percentiles: List[int] = [25, 50, 75, 90, 95, 99]
) -> Dict:
    """计算百分位数"""
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


def formatStatistics(stats: Dict[str, Any]) -> Dict[str, Any]:
    """格式化统计结果，使其更易读"""

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

    # 转换 byBook（处理 defaultdict 和 Counter）
    for book, bookStats in stats["byBook"].items():
        formatted["byBook"][book] = {
            "count": bookStats["count"],
            "subjects": dict(bookStats["subjects"]),
        }

    # 字段覆盖率统计
    total = stats["summary"]["validFiles"]
    for field, fieldStats in stats["fields"].items():
        formatted["fieldCoverage"][field] = {
            "present": fieldStats["present"],
            "missing": fieldStats["missing"],
            "coverageRate": round(fieldStats["present"] / total * 100, 2)
            if total > 0
            else 0,
        }

    # 字段长度分布统计
    for field, fieldStats in stats["fields"].items():
        if fieldStats["lengths"]:
            formatted["fieldLengthDistribution"][field] = calculatePercentiles(
                fieldStats["lengths"]
            )

        # 对于列表类型字段，统计元素数量分布
        if fieldStats["itemCounts"]:
            formatted["fieldLengthDistribution"][f"{field}Count"] = (
                calculatePercentiles(fieldStats["itemCounts"])
            )

    # definitions 分析
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

    # 术语长度分布
    formatted["termLengthDistribution"] = (
        calculatePercentiles(stats["termLengths"]) if stats["termLengths"] else {}
    )

    return formatted


def createVisualization(stats: Dict[str, Any], outputDir: str) -> None:
    """生成可视化图表"""
    if not HAS_MATPLOTLIB:
        print("⚠️  跳过可视化：matplotlib 未安装")
        return

    print("\n🎨 生成可视化图表...")

    # 创建输出目录
    vizDir = os.path.join(outputDir, "visualizations")
    os.makedirs(vizDir, exist_ok=True)

    # 1. 书籍术语分布柱状图
    createBookDistributionChart(stats, vizDir)

    # 2. 学科分布饼图
    createSubjectDistributionChart(stats, vizDir)

    # 3. 字段覆盖率热力图
    createFieldCoverageChart(stats, vizDir)

    # 4. 术语长度分布直方图
    createTermLengthDistribution(stats, vizDir)

    # 5. 定义类型分布
    createDefinitionTypeChart(stats, vizDir)

    # 6. 综合统计面板
    createComprehensiveDashboard(stats, vizDir)

    print(f"✅ 可视化图表已保存到: {vizDir}")


def createBookDistributionChart(stats: Dict[str, Any], outputDir: str) -> None:
    """书籍术语分布柱状图"""
    try:
        fig, ax = plt.subplots(figsize=(12, 6))

        books = list(stats["byBook"].keys())
        counts = [stats["byBook"][book]["count"] for book in books]

        # 简化书名显示
        shortNames = [
            book.replace("(第5版)", "").replace("(第五版)", "").replace("第三版", "")
            for book in books
        ]

        colors = plt.cm.Set3(range(len(books)))
        bars = ax.bar(
            range(len(books)), counts, color=colors, edgecolor="black", linewidth=1.2
        )

        # 添加数值标签
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

        print("  ✓ 书籍术语分布图")
    except Exception as e:
        print(f"  ✗ 书籍术语分布图生成失败: {e}")


def createSubjectDistributionChart(stats: Dict[str, Any], outputDir: str) -> None:
    """学科分布饼图"""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))

        subjects = list(stats["bySubject"].keys())
        counts = [stats["bySubject"][s] for s in subjects]

        # 过滤掉拼写错误的学科
        filteredData = [(s, c) for s, c in zip(subjects, counts) if c > 10]
        subjects, counts = zip(*filteredData) if filteredData else ([], [])

        colors = plt.cm.Pastel1(range(len(subjects)))
        explode = [0.05] * len(subjects)  # 分离饼图

        wedges, texts, autotexts = ax.pie(
            counts,
            labels=subjects,
            autopct="%1.1f%%",
            colors=colors,
            explode=explode,
            startangle=90,
            textprops={"fontsize": 12, "fontweight": "bold"},
        )

        # 添加数量标注
        for i, (subject, count) in enumerate(zip(subjects, counts)):
            texts[i].set_text(f"{subject}\n({count}个)")

        ax.set_title("学科术语分布", fontsize=16, fontweight="bold", pad=20)

        plt.tight_layout()
        plt.savefig(os.path.join(outputDir, "2_学科分布.png"), bbox_inches="tight")
        plt.close()

        print("  ✓ 学科分布图")
    except Exception as e:
        print(f"  ✗ 学科分布图生成失败: {e}")


def createFieldCoverageChart(stats: Dict[str, Any], outputDir: str) -> None:
    """字段覆盖率横向柱状图"""
    try:
        fig, ax = plt.subplots(figsize=(12, 8))

        # 计算覆盖率
        total = stats["summary"]["validFiles"]
        fieldNames = []
        coverageRates = []

        for field, fieldStats in stats["fields"].items():
            rate = fieldStats["present"] / total * 100 if total > 0 else 0
            fieldNames.append(field)
            coverageRates.append(rate)

        # 排序
        sortedData = sorted(
            zip(fieldNames, coverageRates), key=lambda x: x[1], reverse=True
        )
        fieldNames, coverageRates = zip(*sortedData)

        # 颜色编码：>95% 绿色，90-95% 黄色，<90% 红色
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

        # 添加百分比标签
        for i, (bar, rate) in enumerate(zip(bars, coverageRates)):
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

        print("  ✓ 字段覆盖率图")
    except Exception as e:
        print(f"  ✗ 字段覆盖率图生成失败: {e}")


def createTermLengthDistribution(stats: Dict[str, Any], outputDir: str) -> None:
    """术语长度分布直方图"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 术语长度分布
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

        # 定义文本长度分布
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

        print("  ✓ 长度分布图")
    except Exception as e:
        print(f"  ✗ 长度分布图生成失败: {e}")


def createDefinitionTypeChart(stats: Dict[str, Any], outputDir: str) -> None:
    """定义类型分布图"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        types = list(stats["definitions"]["types"].keys())
        counts = [stats["definitions"]["types"][t] for t in types]

        colors = plt.cm.Set2(range(len(types)))
        bars = ax.bar(types, counts, color=colors, edgecolor="black", linewidth=1.2)

        # 添加数值和百分比标签
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

        print("  ✓ 定义类型分布图")
    except Exception as e:
        print(f"  ✗ 定义类型分布图生成失败: {e}")


def createComprehensiveDashboard(stats: Dict[str, Any], outputDir: str) -> None:
    """综合统计面板"""
    try:
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. 总体统计文本
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis("off")
        summaryText = f"""
        📊 数学术语数据统计报告
        
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

        # 2. 书籍分布
        ax2 = fig.add_subplot(gs[1, 0])
        books = list(stats["byBook"].keys())
        counts = [stats["byBook"][book]["count"] for book in books]
        shortNames = [book.split("(")[0][:8] for book in books]
        ax2.bar(range(len(books)), counts, color=plt.cm.Set3(range(len(books))))
        ax2.set_xticks(range(len(books)))
        ax2.set_xticklabels(shortNames, rotation=45, ha="right", fontsize=9)
        ax2.set_title("书籍分布", fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

        # 3. 学科分布
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

        # 4. 字段覆盖率 TOP 10
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

        # 5. 术语长度分布
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

        # 6. 定义文本长度分布
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

        # 7. 定义类型分布
        ax7 = fig.add_subplot(gs[2, 2])
        types = list(stats["definitions"]["types"].keys())
        typeCounts = [stats["definitions"]["types"][t] for t in types]
        ax7.bar(types, typeCounts, color=plt.cm.Set2(range(len(types))))
        ax7.set_title("定义类型分布", fontweight="bold")
        ax7.grid(axis="y", alpha=0.3)

        plt.suptitle("数学术语数据综合统计面板", fontsize=18, fontweight="bold", y=0.98)
        plt.savefig(os.path.join(outputDir, "0_综合统计面板.png"), bbox_inches="tight")
        plt.close()

        print("  ✓ 综合统计面板")
    except Exception as e:
        print(f"  ✗ 综合统计面板生成失败: {e}")


def main():
    """主函数"""
    print("=" * 70)
    print(" " * 20 + "📊 数学术语数据统计与可视化")
    print("=" * 70)

    # 输入输出路径
    chunkDir = config.CHUNK_DIR
    statsDir = os.path.join(config.PROJECT_ROOT, "data", "stats")
    outputFile = os.path.join(statsDir, "chunkStatistics.json")

    # 确保输出目录存在
    os.makedirs(statsDir, exist_ok=True)

    print(f"\n📂 输入目录: {chunkDir}")
    print(f"📂 输出目录: {statsDir}\n")

    # 构建统计信息
    print("🔄 开始统计分析...\n")
    rawStats = buildStatistics(chunkDir)

    # 格式化统计结果
    print("\n🔄 格式化统计结果...")
    formattedStats = formatStatistics(rawStats)

    # 保存到文件
    print(f"💾 保存统计报告: {outputFile}")
    with open(outputFile, "w", encoding="utf-8") as f:
        json.dump(formattedStats, f, ensure_ascii=False, indent=2)

    # 生成可视化
    createVisualization(rawStats, statsDir)

    # 打印摘要
    print("\n" + "=" * 70)
    print(" " * 25 + "✅ 统计完成！")
    print("=" * 70)
    print(f"\n📊 总体统计:")
    print(f"  • 总文件数: {formattedStats['summary']['totalFiles']:,}")
    print(f"  • 有效文件: {formattedStats['summary']['validFiles']:,}")
    print(f"  • 术语总数: {formattedStats['summary']['totalTerms']:,}")

    print(f"\n📖 各书籍术语数量:")
    for book, bookStats in sorted(
        formattedStats["byBook"].items(), key=lambda x: x[1]["count"], reverse=True
    ):
        print(f"  • {book}: {bookStats['count']} 个")

    print(f"\n📚 学科分布:")
    for subject, count in sorted(
        formattedStats["bySubject"].items(), key=lambda x: x[1], reverse=True
    ):
        percentage = count / formattedStats["summary"]["totalTerms"] * 100
        print(f"  • {subject}: {count} 个 ({percentage:.1f}%)")

    print(f"\n🔄 重复术语: {formattedStats['duplicates']['count']} 个")

    print(f"\n💡 输出文件:")
    print(f"  • 统计报告: {outputFile}")
    if HAS_MATPLOTLIB:
        print(f"  • 可视化图表: {os.path.join(statsDir, 'visualizations')}")

    print("\n" + "=" * 70)
    print(" " * 20 + "🎉 所有任务完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
