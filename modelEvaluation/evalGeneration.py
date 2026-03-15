"""生成质量评测脚本。"""

from __future__ import annotations

import os

import config
from modelEvaluation.generationEval.cli import buildParser
from modelEvaluation.generationEval.evaluator import (
    evaluateGeneration as evaluateGeneration,
)
from modelEvaluation.generationEval.evaluator import (
    findBestWorstExamples as findBestWorstExamples,
)
from modelEvaluation.generationEval.ioOps import (
    loadGoldQueries as loadGoldQueries,
)
from modelEvaluation.generationEval.ioOps import (
    loadRagResults as loadRagResults,
)
from modelEvaluation.generationEval.metrics import (
    calculateBleuScore as calculateBleuScore,
)
from modelEvaluation.generationEval.metrics import (
    calculateRougeScores as calculateRougeScores,
)
from modelEvaluation.generationEval.metrics import (
    calculateSourceCitationRate as calculateSourceCitationRate,
)
from modelEvaluation.generationEval.metrics import (
    calculateTermHitRate as calculateTermHitRate,
)
from modelEvaluation.generationEval.metrics import (
    isAnswerValid as isAnswerValid,
)
from modelEvaluation.generationEval.reporting import printExamples as printExamples
from modelEvaluation.generationEval.reporting import printSummary as printSummary
from modelEvaluation.generationEval.runner import runEvalGeneration


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
