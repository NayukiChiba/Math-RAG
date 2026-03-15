"""快速评测模块。"""

from modelEvaluation.quickEvalCore.constants import (
    ALL_METHODS,
    BASIC_METHODS,
    OPTIMIZED_METHODS,
)
from modelEvaluation.quickEvalCore.evaluator import evaluateMethod
from modelEvaluation.quickEvalCore.runner import runEval, saveReport

__all__ = [
    "ALL_METHODS",
    "BASIC_METHODS",
    "OPTIMIZED_METHODS",
    "evaluateMethod",
    "runEval",
    "saveReport",
]
