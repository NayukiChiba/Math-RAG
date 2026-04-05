"""检索评测脚本。"""

from __future__ import annotations

import os

import config
from modelEvaluation.common.metrics import (
    calculateAP as calculateAP,
)
from modelEvaluation.common.metrics import (
    calculateDCG as calculateDCG,
)
from modelEvaluation.common.metrics import (
    calculateIDCG as calculateIDCG,
)
from modelEvaluation.common.metrics import (
    calculateMRR as calculateMRR,
)
from modelEvaluation.common.metrics import (
    calculateNDCG as calculateNDCG,
)
from modelEvaluation.common.metrics import (
    calculateRecallAtK as calculateRecallAtK,
)
from modelEvaluation.retrievalEval.charting import (
    generateComparisonChart as generateComparisonChart,
)
from modelEvaluation.retrievalEval.cli import buildParser
from modelEvaluation.retrievalEval.evaluator import evaluateMethod as evaluateMethod
from modelEvaluation.retrievalEval.ioOps import loadQueries as loadQueries
from modelEvaluation.retrievalEval.runner import runEvalRetrieval


def main() -> int:
    output_controller = config.getOutputController()
    default_output = os.path.join(
        output_controller.get_json_dir(), "retrieval_metrics.json"
    )

    parser = buildParser(default_output)
    args = parser.parse_args()
    args.output = output_controller.normalize_json_path(
        args.output, "retrieval_metrics.json"
    )
    return runEvalRetrieval(args)


if __name__ == "__main__":
    raise SystemExit(main())
