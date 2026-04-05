"""检索评测脚本。"""

from __future__ import annotations

import os

from core import config
from research.modelEvaluation.common.metrics import (
    calculateAP as calculateAP,
)
from research.modelEvaluation.common.metrics import (
    calculateDCG as calculateDCG,
)
from research.modelEvaluation.common.metrics import (
    calculateIDCG as calculateIDCG,
)
from research.modelEvaluation.common.metrics import (
    calculateMRR as calculateMRR,
)
from research.modelEvaluation.common.metrics import (
    calculateNDCG as calculateNDCG,
)
from research.modelEvaluation.common.metrics import (
    calculateRecallAtK as calculateRecallAtK,
)

try:
    from reports_generation.viz.charting import (
        generateComparisonChart as generateComparisonChart,
    )
except ImportError:
    generateComparisonChart = None  # type: ignore[assignment,misc]
from research.modelEvaluation.retrievalEval.cli import buildParser
from research.modelEvaluation.retrievalEval.evaluator import (
    evaluateMethod as evaluateMethod,
)
from research.modelEvaluation.retrievalEval.ioOps import loadQueries as loadQueries
from research.modelEvaluation.retrievalEval.runner import runEvalRetrieval


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
