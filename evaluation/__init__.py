"""评测包统一导出。"""

from evaluation.evalGeneration import (
    calculateBleuScore,
    calculateRougeScores,
    evaluateGeneration,
    findBestWorstExamples,
    loadGoldQueries,
    loadRagResults,
    printExamples,
    printSummary,
)
from evaluation.evalGeneration import (
    main as runEvalGeneration,
)
from evaluation.evalRetrieval import (
    calculateAP,
    calculateDCG,
    calculateIDCG,
    calculateMRR,
    calculateNDCG,
    calculateRecallAtK,
    generateComparisonChart,
    loadQueries,
)
from evaluation.evalRetrieval import (
    evaluateMethod as evaluateRetrievalMethod,
)
from evaluation.evalRetrieval import (
    main as runEvalRetrieval,
)
from evaluation.quickEval import (
    calculateMAP,
    runEval,
)
from evaluation.quickEval import (
    main as run_quick_eval,
)
from evaluation.quickEval import (
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
