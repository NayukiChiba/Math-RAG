"""检索评测脚本。"""

from __future__ import annotations

import os

import config
from evaluation.common.metrics import (
    calculateAP as calculateAP,
)
from evaluation.common.metrics import (
    calculateDCG as calculateDCG,
)
from evaluation.common.metrics import (
    calculateIDCG as calculateIDCG,
)
from evaluation.common.metrics import (
    calculateMRR as calculateMRR,
)
from evaluation.common.metrics import (
    calculateNDCG as calculateNDCG,
)
from evaluation.common.metrics import (
    calculateRecallAtK as calculateRecallAtK,
)
from evaluation.retrieval_eval.charting import (
    generateComparisonChart as generateComparisonChart,
)
from evaluation.retrieval_eval.cli import buildParser
from evaluation.retrieval_eval.evaluator import evaluateMethod as evaluateMethod
from evaluation.retrieval_eval.ioOps import loadQueries as loadQueries
from evaluation.retrieval_eval.runner import runEvalRetrieval


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
