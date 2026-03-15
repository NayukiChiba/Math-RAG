"""统计显著性检验脚本

对 BM25+ 与 Vector 检索方法进行 Bootstrap 置信区间估计和配对 t 检验，
基于 log 时间目录中 all_methods.json 的逐查询 Recall@5 数组。

用法：
    python3 scripts/evaluation/significanceTest.py [--input <all_methods.json>] [--output <significance_test.json>]
"""

import argparse
import json
import os

import numpy as np
from scipy import stats

import config
from utils import getFileLoader

_LOADER = getFileLoader()


def bootstrap_ci(
    scores: list[float], n_resamples: int = 10000, ci: float = 0.95
) -> dict:
    """Bootstrap 法估计均值的置信区间（numpy 向量化）。"""
    arr = np.array(scores, dtype=np.float32)
    rng = np.random.default_rng(42)
    # 分批次避免内存溢出（每批 500 次）
    batch = min(500, n_resamples)
    means = []
    n_full = n_resamples // batch
    remainder = n_resamples % batch
    for _ in range(n_full):
        idx = rng.integers(0, len(arr), size=(batch, len(arr)))
        means.append(arr[idx].mean(axis=1))
    if remainder:
        idx = rng.integers(0, len(arr), size=(remainder, len(arr)))
        means.append(arr[idx].mean(axis=1))
    means_arr = np.concatenate(means)
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": round(float(arr.mean()), 6),
        "ci_lower": round(float(np.percentile(means_arr, alpha * 100)), 6),
        "ci_upper": round(float(np.percentile(means_arr, (1 - alpha) * 100)), 6),
        "std": round(float(arr.std(ddof=1)), 6),
    }


def paired_t_test(a: list[float], b: list[float]) -> dict:
    """配对 t 检验（双侧）。

    Args:
        a: 方法 A 的逐查询得分
        b: 方法 B 的逐查询得分

    Returns:
        dict: t_stat / p_value / significant (p<0.05)
    """
    if len(a) != len(b):
        raise ValueError(
            f"配对 t 检验要求两组得分长度相同，但 len(a)={len(a)}, len(b)={len(b)}"
        )
    t_stat, p_value = stats.ttest_rel(a, b)
    return {
        "t_stat": round(float(t_stat), 6),
        "p_value": round(float(p_value), 8),
        "significant_at_0.05": bool(p_value < 0.05),
        "significant_at_0.01": bool(p_value < 0.01),
    }


def run_significance_test(
    input_path: str, output_path: str, n_resamples: int = 10000
) -> None:
    """读取全量方法对比文件，输出显著性检验报告。"""
    data = _LOADER.json(input_path)

    results_map: dict[str, dict] = {r["method"]: r for r in data["results"]}

    required_methods = ["BM25+", "Vector", "Hybrid+-Weighted"]
    for m in required_methods:
        if m not in results_map:
            raise KeyError(
                f"方法 '{m}' 不在输入文件中，可用方法: {list(results_map.keys())}"
            )

    metrics = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "map"]

    # ── 逐指标 Bootstrap CI ─────────────────────────────────────
    bootstrap_results: dict[str, dict[str, dict]] = {}
    for method_name, method_data in results_map.items():
        bootstrap_results[method_name] = {}
        for metric in metrics:
            scores = method_data.get(metric)
            if isinstance(scores, list) and len(scores) > 0:
                bootstrap_results[method_name][metric] = bootstrap_ci(
                    scores, n_resamples
                )
            else:
                # 标量指标，跳过 bootstrap
                bootstrap_results[method_name][metric] = {
                    "mean": method_data.get(metric, None)
                }

    # ── 配对 t 检验：BM25+ vs Vector ────────────────────────────
    comparisons = [
        ("BM25+", "Vector"),
        ("BM25+", "Hybrid+-Weighted"),
        ("Hybrid+-Weighted", "Vector"),
    ]
    t_test_results: list[dict] = []
    for method_a, method_b in comparisons:
        comparison: dict = {"method_a": method_a, "method_b": method_b, "metrics": {}}
        for metric in metrics:
            a_scores = results_map[method_a].get(metric)
            b_scores = results_map[method_b].get(metric)
            if isinstance(a_scores, list) and isinstance(b_scores, list):
                comparison["metrics"][metric] = paired_t_test(a_scores, b_scores)
        t_test_results.append(comparison)

    first_result = next(iter(results_map.values()))
    first_r5 = first_result.get("recall@5", [])
    default_total = len(first_r5) if isinstance(first_r5, list) else 0
    report = {
        "source_file": os.path.basename(input_path),
        "total_queries": data.get("total_queries", default_total),
        "n_resamples": n_resamples,
        "methods_tested": list(results_map.keys()),
        "bootstrap_ci": bootstrap_results,
        "paired_t_tests": t_test_results,
    }

    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── 控制台摘要 ───────────────────────────────────────────────
    print("=" * 60)
    print(" 显著性检验结果摘要")
    print("=" * 60)
    print(f"\n{'方法':<22} {'Recall@5 均值':>14} {'95% CI':>22}")
    print("-" * 60)
    for m in results_map:
        ci = bootstrap_results[m].get("recall@5", {})
        mean = ci.get("mean")
        lo = ci.get("ci_lower")
        hi = ci.get("ci_upper")
        if (
            isinstance(mean, int | float)
            and isinstance(lo, int | float)
            and isinstance(hi, int | float)
        ):
            print(f"{m:<22} {mean:>14.4f} [{lo:.4f}, {hi:.4f}]")
        else:
            print(f"{m:<22} {'N/A':>14} [?, ?]")

    print(f"\n{'对比':<35} {'Recall@5 p值':>14} {'显著(p<0.05)':>14}")
    print("-" * 65)
    for comp in t_test_results:
        label = f"{comp['method_a']} vs {comp['method_b']}"
        r5 = comp["metrics"].get("recall@5", {})
        pv = r5.get("p_value")
        sig = "[yes]" if r5.get("significant_at_0.05") else "[no]"
        if isinstance(pv, int | float):
            print(f"{label:<35} {pv:>14.2e} {sig:>14}")
        else:
            print(f"{label:<35} {'N/A':>14} {sig:>14}")

    print(f"\n 显著性检验报告已保存: {output_path}")


def main() -> None:
    outputController = config.getOutputController()
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
    args = parser.parse_args()
    args.output = outputController.normalize_json_path(
        args.output, "significance_test.json"
    )
    run_significance_test(args.input, args.output, args.n_resamples)


if __name__ == "__main__":
    main()
