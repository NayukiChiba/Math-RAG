"""数据统计包统一导出。"""

from research.dataStat.loaders import (
    loadJsonFile,
)
from research.dataStat.stats_builder import (
    analyzeDefinitions,
    buildStatistics,
    calculateFieldStats,
)
from research.dataStat.stats_formatter import (
    calculatePercentiles,
    formatStatistics,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "analyzeDefinitions",
    "buildStatistics",
    "calculateFieldStats",
    "calculatePercentiles",
    "formatStatistics",
    "loadJsonFile",
    "run_statistics",
]


def run_statistics() -> None:
    """延迟导入入口，避免包初始化阶段循环依赖。"""
    from research.dataStat.chunkStatistics import run_statistics as _run

    _run()
