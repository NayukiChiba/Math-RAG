"""快速评测模块。"""

from evaluation.quick_eval.constants import (
    ALL_METHODS,
    BASIC_METHODS,
    OPTIMIZED_METHODS,
)
from evaluation.quick_eval.evaluator import evaluateMethod
from evaluation.quick_eval.runner import runEval, saveReport

__all__ = [
    "ALL_METHODS",
    "BASIC_METHODS",
    "OPTIMIZED_METHODS",
    "evaluateMethod",
    "runEval",
    "saveReport",
]
