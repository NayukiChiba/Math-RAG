"""Charting utilities for retrieval results."""

from __future__ import annotations

import os
from typing import Any


def generateComparisonChart(all_metrics: list[dict[str, Any]], output_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        methods = [item["method"] for item in all_metrics]
        recall_k = [1, 3, 5, 10]
        ndcg_k = [3, 5, 10]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("检索评测指标对比", fontsize=16, fontweight="bold")

        ax = axes[0, 0]
        x = np.arange(len(methods))
        width = 0.2
        for i, k in enumerate(recall_k):
            values = [item["avg_metrics"][f"recall@{k}"] for item in all_metrics]
            ax.bar(x + i * width, values, width, label=f"Recall@{k}")
        ax.set_ylabel("Recall")
        ax.set_title("Recall@K")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        ax = axes[0, 1]
        x = np.arange(len(methods))
        width = 0.25
        for i, k in enumerate(ndcg_k):
            values = [item["avg_metrics"][f"ndcg@{k}"] for item in all_metrics]
            ax.bar(x + i * width, values, width, label=f"nDCG@{k}")
        ax.set_ylabel("nDCG")
        ax.set_title("nDCG@K")
        ax.set_xticks(x + width)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        ax = axes[1, 0]
        x = np.arange(len(methods))
        width = 0.35
        mrr = [item["avg_metrics"]["mrr"] for item in all_metrics]
        map_scores = [item["avg_metrics"]["map"] for item in all_metrics]
        ax.bar(x - width / 2, mrr, width, label="MRR")
        ax.bar(x + width / 2, map_scores, width, label="MAP")
        ax.set_ylabel("Score")
        ax.set_title("MRR 和 MAP 对比")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        ax = axes[1, 1]
        x = np.arange(len(methods))
        times = [item["avg_query_time"] * 1000 for item in all_metrics]
        bars = ax.bar(x, times, color="skyblue")
        ax.set_ylabel("时间 (ms)")
        ax.set_title("平均查询时间")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.3)

        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{time_val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        chart_path = os.path.join(output_dir, "retrieval_comparison.png")
        plt.savefig(chart_path, dpi=300, bbox_inches="tight")
        print(f" 对比图表已保存: {chart_path}")
    except ImportError:
        print("  跳过图表生成：matplotlib 未安装")
    except Exception as exc:
        print(f"  图表生成失败: {exc}")
