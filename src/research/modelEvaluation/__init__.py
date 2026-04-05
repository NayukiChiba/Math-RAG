"""评测包统一导出。"""

from research.modelEvaluation.evalGeneration import (
    calculateBleuScore,
    calculateRougeScores,
    evaluateGeneration,
    findBestWorstExamples,
    loadGoldQueries,
    loadRagResults,
    printExamples,
    printSummary,
)
from research.modelEvaluation.evalGeneration import (
    main as runEvalGeneration,
)
from research.modelEvaluation.evalRetrieval import (
    calculateAP,
    calculateDCG,
    calculateIDCG,
    calculateMRR,
    calculateNDCG,
    calculateRecallAtK,
    loadQueries,
)
from research.modelEvaluation.evalRetrieval import (
    evaluateMethod as evaluateRetrievalMethod,
)
from research.modelEvaluation.evalRetrieval import (
    main as runEvalRetrieval,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "calculateAP",
    "calculateBleuScore",
    "calculateDCG",
    "calculateIDCG",
    "calculateMRR",
    "calculateNDCG",
    "calculateRecallAtK",
    "calculateRougeScores",
    "evaluateGeneration",
    "evaluateRetrievalMethod",
    "findBestWorstExamples",
    "loadGoldQueries",
    "loadQueries",
    "loadRagResults",
    "printExamples",
    "printSummary",
    "runEvalGeneration",
    "runEvalRetrieval",
]
