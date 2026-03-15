"""生成质量评测脚本。"""

from __future__ import annotations

import os

import config
from evaluation.generation_eval.cli import buildParser
from evaluation.generation_eval.evaluator import (
    evaluateGeneration as evaluateGeneration,
)
from evaluation.generation_eval.evaluator import (
    findBestWorstExamples as findBestWorstExamples,
)
from evaluation.generation_eval.ioOps import (
    loadGoldQueries as loadGoldQueries,
)
from evaluation.generation_eval.ioOps import (
    loadRagResults as loadRagResults,
)
from evaluation.generation_eval.metrics import (
    calculateBleuScore as calculateBleuScore,
)
from evaluation.generation_eval.metrics import (
    calculateRougeScores as calculateRougeScores,
)
from evaluation.generation_eval.metrics import (
    calculateSourceCitationRate as calculateSourceCitationRate,
)
from evaluation.generation_eval.metrics import (
    calculateTermHitRate as calculateTermHitRate,
)
from evaluation.generation_eval.metrics import (
    isAnswerValid as isAnswerValid,
)
from evaluation.generation_eval.reporting import printExamples as printExamples
from evaluation.generation_eval.reporting import printSummary as printSummary
from evaluation.generation_eval.runner import runEvalGeneration


def main() -> int:
    output_controller = config.getOutputController()
    default_output = os.path.join(
        output_controller.get_json_dir(), "generation_metrics.json"
    )

    parser = buildParser(default_output)
    args = parser.parse_args()
    args.output = output_controller.normalize_json_path(
        args.output, "generation_metrics.json"
    )
    return runEvalGeneration(args)


if __name__ == "__main__":
    raise SystemExit(main())
