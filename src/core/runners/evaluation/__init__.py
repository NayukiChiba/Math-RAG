"""评测与报告脚本集合。"""

from core.runners.evaluation.buildEvalTermMapping import main as build_eval_term_mapping
from core.runners.evaluation.evalGenerationComparison import (
    main as eval_generation_comparison,
)
from core.runners.evaluation.significanceTest import main as significance_test

__all__ = [
    "build_eval_term_mapping",
    "eval_generation_comparison",
    "significance_test",
]
