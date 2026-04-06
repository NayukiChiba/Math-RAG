"""快速评测模块。"""

from reports_generation.quick_eval.quickEvalCore.constants import (
    ALL_METHODS,
    BASIC_METHODS,
    OPTIMIZED_METHODS,
)
from reports_generation.quick_eval.quickEvalCore.evaluator import evaluateMethod
from reports_generation.quick_eval.quickEvalCore.runner import runEval, saveReport

__all__ = [
    "ALL_METHODS",
    "BASIC_METHODS",
    "OPTIMIZED_METHODS",
    "evaluateMethod",
    "runEval",
    "saveReport",
]
