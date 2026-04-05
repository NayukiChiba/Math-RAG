"""评测与报告脚本集合。"""

from runners.evaluation.buildEvalTermMapping import main as build_eval_term_mapping
from runners.evaluation.evalGenerationComparison import (
    main as eval_generation_comparison,
)
from runners.evaluation.generateReport import main as generate_report
from runners.evaluation.significanceTest import main as significance_test

__all__ = [
    "build_eval_term_mapping",
    "eval_generation_comparison",
    "generate_report",
    "significance_test",
]
