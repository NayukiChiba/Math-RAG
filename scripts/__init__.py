"""脚本包统一导出。"""

from scripts.addMissingTerms import main as add_missing_terms
from scripts.buildTermMapping import main as build_term_mapping
from scripts.evalGenerationComparison import main as eval_generation_comparison
from scripts.experimentWebUI import main as launch_experiment_webui
from scripts.generateReport import main as generate_report
from scripts.runExperiments import main as run_experiments
from scripts.runRag import main as run_rag
from scripts.significanceTest import main as significance_test

__all__ = [
    "add_missing_terms",
    "build_term_mapping",
    "eval_generation_comparison",
    "generate_report",
    "launch_experiment_webui",
    "run_experiments",
    "run_rag",
    "significance_test",
]
