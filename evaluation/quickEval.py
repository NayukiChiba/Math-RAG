"""快速检索测试系统。"""

from __future__ import annotations

import os
from typing import Any

import config
from evaluation.common.metrics import calculateMAP as calculateMAP
from evaluation.common.metrics import calculateMRR as calculateMRR
from evaluation.common.metrics import calculateNDCG as calculateNDCG
from evaluation.common.metrics import calculateRecallAtK as calculateRecallAtK
from evaluation.quick_eval.cli import buildParser
from evaluation.quick_eval.constants import (
    ALL_METHODS as _ALL_METHODS,
)
from evaluation.quick_eval.constants import (
    BASIC_METHODS as _BASIC_METHODS,
)
from evaluation.quick_eval.constants import (
    OPTIMIZED_METHODS as _OPTIMIZED_METHODS,
)
from evaluation.quick_eval.dataOps import loadCorpus as loadCorpus
from evaluation.quick_eval.dataOps import loadQueries as loadQueries
from evaluation.quick_eval.evaluator import evaluateMethod as evaluateMethod
from evaluation.quick_eval.runner import runEval as _runEvalImpl
from evaluation.quick_eval.runner import saveReport as _saveReportImpl

# 兼容旧常量名
_BASIC_METHODS = _BASIC_METHODS
_OPTIMIZED_METHODS = _OPTIMIZED_METHODS
_ALL_METHODS = _ALL_METHODS


def runEval(
    methods: list[str] | None = None,
    mode: str = "basic",
    numQueries: int = 20,
    allQueries: bool = False,
    topK: int = 10,
) -> dict[str, Any]:
    return _runEvalImpl(
        methods=methods,
        mode=mode,
        num_queries=numQueries,
        all_queries=allQueries,
        top_k=topK,
    )


def saveReport(metrics: dict[str, Any], outputFile: str) -> None:
    _saveReportImpl(metrics, outputFile)


def main() -> int:
    output_controller = config.getOutputController()
    parser = buildParser()
    args = parser.parse_args()

    metrics = _runEvalImpl(
        methods=args.methods,
        mode=args.mode,
        num_queries=args.num_queries,
        all_queries=args.all_queries,
        top_k=args.topk,
    )

    if metrics and args.output:
        _saveReportImpl(
            metrics,
            output_controller.normalize_json_path(
                args.output, f"quick_eval_{args.mode}.json"
            ),
        )
    elif metrics:
        default_output = os.path.join(
            output_controller.get_json_dir(), f"quick_eval_{args.mode}.json"
        )
        _saveReportImpl(metrics, default_output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
