"""研究线脚本编排层。"""

from research.runners.addMissingTerms import main as add_missing_terms
from research.runners.buildTermMapping import main as build_term_mapping
from research.runners.evalGenerationComparison import main as eval_generation_comparison
from research.runners.experimentWebUI import main as launch_experiment_webui
from research.runners.runExperiments import main as run_experiments
from research.runners.significanceTest import main as significance_test

__all__ = [
    "add_missing_terms",
    "build_term_mapping",
    "eval_generation_comparison",
    "launch_experiment_webui",
    "run_experiments",
    "significance_test",
]
