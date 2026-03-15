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
from modelEvaluation.quickEval import (
    calculateMAP,
    runEval,
)
from modelEvaluation.quickEval import (
    main as run_quick_eval,
)
from modelEvaluation.quickEval import (
    saveReport as saveQuickEvalReport,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "calculateAP",
    "calculateBleuScore",
    "calculateDCG",
    "calculateIDCG",
    "calculateMAP",
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
    "run_quick_eval",
    "runEval",
    "saveQuickEvalReport",
]
