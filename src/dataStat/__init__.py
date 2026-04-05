"""数据统计包统一导出。"""

from dataStat.loaders import (
    loadJsonFile,
)
from dataStat.stats_builder import (
    analyzeDefinitions,
    buildStatistics,
    calculateFieldStats,
)
from dataStat.stats_formatter import (
    calculatePercentiles,
    formatStatistics,
)
from dataStat.visualization import (
    HAS_MATPLOTLIB,
    createVisualization,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "analyzeDefinitions",
    "buildStatistics",
    "calculateFieldStats",
    "calculatePercentiles",
    "HAS_MATPLOTLIB",
    "createVisualization",
    "formatStatistics",
    "loadJsonFile",
    "run_statistics",
]


def run_statistics() -> None:
    """延迟导入入口，避免包初始化阶段循环依赖。"""
    from dataStat.chunkStatistics import run_statistics as _run

    _run()
