"""评测包统一导出。"""

from modelEvaluation.evalGeneration import (
    calculateBleuScore,
    calculateRougeScores,
    evaluateGeneration,
    findBestWorstExamples,
    loadGoldQueries,
    loadRagResults,
    printExamples,
    printSummary,
)
from modelEvaluation.evalGeneration import (
    main as runEvalGeneration,
)
from modelEvaluation.evalRetrieval import (
    calculateAP,
    calculateDCG,
    calculateIDCG,
    calculateMRR,
    calculateNDCG,
    calculateRecallAtK,
    generateComparisonChart,
    loadQueries,
)
from modelEvaluation.evalRetrieval import (
    evaluateMethod as evaluateRetrievalMethod,
)
from modelEvaluation.evalRetrieval import (
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
    "generateComparisonChart",
    "loadGoldQueries",
    "loadQueries",
    "loadRagResults",
    "printExamples",
    "printSummary",
    "runEvalGeneration",
    "runEvalRetrieval",
]
