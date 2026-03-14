"""数据统计包统一导出。"""

from dataStat.chunkStatistics import (
    analyzeDefinitions,
    buildStatistics,
    calculateFieldStats,
    calculatePercentiles,
    createVisualization,
    formatStatistics,
    loadJsonFile,
)
from dataStat.chunkStatistics import (
    main as run_statistics,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "analyzeDefinitions",
    "buildStatistics",
    "calculateFieldStats",
    "calculatePercentiles",
    "createVisualization",
    "formatStatistics",
    "loadJsonFile",
    "run_statistics",
]
