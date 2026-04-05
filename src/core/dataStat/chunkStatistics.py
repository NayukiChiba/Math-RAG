"""
术语数据统计与可视化（入口模块）

功能：
1. 遍历 data/processed/chunk/ 下所有术语 JSON 文件
2. 统计字段缺失率、长度分布、学科覆盖等信息
3. 生成可视化图表
4. 输出统计报告到 data/stats/，图表输出到 outputs/

使用方法：
    python -m dataStat.chunkStatistics
    或
    python dataStat/chunkStatistics.py
"""

import json
import os

from core import config
from core.dataStat import (
    buildStatistics,
    formatStatistics,
)


def run_statistics() -> None:
    """统计入口。"""
    print("=" * 70)
    print(" " * 20 + " 数学术语数据统计与可视化")
    print("=" * 70)

    chunkDir = config.CHUNK_DIR
    statsDir = config.STATS_DIR
    vizBaseDir = config.OUTPUTS_DIR
    outputFile = os.path.join(statsDir, "chunkStatistics.json")

    os.makedirs(statsDir, exist_ok=True)

    print(f"\n 输入目录: {chunkDir}")
    print(f" 输出目录: {statsDir}\n")
    print(f" 可视化目录: {vizBaseDir}\n")

    print(" 开始统计分析...\n")
    rawStats = buildStatistics(chunkDir)

    print("\n 格式化统计结果...")
    formattedStats = formatStatistics(rawStats)

    print(f" 保存统计报告: {outputFile}")
    with open(outputFile, "w", encoding="utf-8") as f:
        json.dump(formattedStats, f, ensure_ascii=False, indent=2)

    try:
        from reports_generation.viz.visualization import createVisualization

        createVisualization(rawStats, vizBaseDir)
    except ImportError:
        print("  跳过可视化：reports_generation 未安装")

    print("\n" + "=" * 70)
    print(" " * 25 + " 统计完成！")
    print("=" * 70)
    print("\n 总体统计:")
    print(f"  • 总文件数: {formattedStats['summary']['totalFiles']:,}")
    print(f"  • 有效文件: {formattedStats['summary']['validFiles']:,}")
    print(f"  • 术语总数: {formattedStats['summary']['totalTerms']:,}")

    print("\n 各书籍术语数量:")
    for book, bookStats in sorted(
        formattedStats["byBook"].items(), key=lambda x: x[1]["count"], reverse=True
    ):
        print(f"  • {book}: {bookStats['count']} 个")

    print("\n 学科分布:")
    for subject, count in sorted(
        formattedStats["bySubject"].items(), key=lambda x: x[1], reverse=True
    ):
        percentage = count / formattedStats["summary"]["totalTerms"] * 100
        print(f"  • {subject}: {count} 个 ({percentage:.1f}%)")

    print(f"\n 重复术语: {formattedStats['duplicates']['count']} 个")

    print("\n 输出文件:")
    print(f"  • 统计报告: {outputFile}")
    try:
        from reports_generation.viz.visualization import HAS_MATPLOTLIB

        if HAS_MATPLOTLIB:
            from core.config import getReportsGenerationConfig

            _viz_sub = getReportsGenerationConfig()["viz_output_subdir"]
            print(f"  • 可视化图表: {os.path.join(vizBaseDir, _viz_sub)}")
    except ImportError:
        pass

    print("\n" + "=" * 70)
    print(" " * 20 + " 所有任务完成！")
    print("=" * 70)


if __name__ == "__main__":
    run_statistics()
