"""统计显著性检验入口。"""

from __future__ import annotations

import argparse
import os

from research.runners.evaluation import significanceTest as sig


def main(argv: list[str] | None = None) -> None:
    outputController = sig.config.getOutputController()
    parser = argparse.ArgumentParser(description="检索方法统计显著性检验")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(
            outputController.get_json_dir(), "full_eval", "all_methods.json"
        ),
        help="全量方法对比报告路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(outputController.get_json_dir(), "significance_test.json"),
        help="输出显著性检验报告路径",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=10000,
        dest="n_resamples",
        help="Bootstrap 重采样次数（默认 10000）",
    )
    args = parser.parse_args(argv)
    args.output = outputController.normalize_json_path(
        args.output, "significance_test.json"
    )
    sig.run_significance_test(args.input, args.output, args.n_resamples)


if __name__ == "__main__":
    main()
