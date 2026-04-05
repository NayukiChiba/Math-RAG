"""检索评测脚本。"""

from __future__ import annotations

import os

from core import config
from core.modelEvaluation.common.metrics import (
    calculateAP as calculateAP,
)
from core.modelEvaluation.common.metrics import (
    calculateDCG as calculateDCG,
)
from core.modelEvaluation.common.metrics import (
    calculateIDCG as calculateIDCG,
)
from core.modelEvaluation.common.metrics import (
    calculateMRR as calculateMRR,
)
from core.modelEvaluation.common.metrics import (
    calculateNDCG as calculateNDCG,
)
from core.modelEvaluation.common.metrics import (
    calculateRecallAtK as calculateRecallAtK,
)

try:
    from reports_generation.viz.charting import (
        generateComparisonChart as generateComparisonChart,
    )
except ImportError:
    generateComparisonChart = None  # type: ignore[assignment,misc]
from core.modelEvaluation.retrievalEval.cli import buildParser
from core.modelEvaluation.retrievalEval.evaluator import (
    evaluateMethod as evaluateMethod,
)
from core.modelEvaluation.retrievalEval.ioOps import loadQueries as loadQueries
from core.modelEvaluation.retrievalEval.runner import runEvalRetrieval


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
