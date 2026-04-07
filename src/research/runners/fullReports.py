"""全量评测与报告总控入口。

目标：
1. 串联检索/生成/显著性/报告流程；
2. 时间戳跑次落盘到 outputs/log/<run_id>/（json、run_trace、exports）；
3. 定稿（final_report、figures、conclusions、json 快照）写入 outputs/reports/；
4. 生成 run_trace、可视化导出数据（csv/jsonl/parquet）保留在 log 跑次目录。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Any

from core import config
from core.cli.runner import run_module_main
from core.utils import getFileLoader

_LOADER = getFileLoader()


@dataclass
class StageResult:
    name: str
    ok: bool
    started_at: float
    ended_at: float
    message: str = ""

    @property
    def elapsed_s(self) -> float:
        return self.ended_at - self.started_at


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _append_jsonl(path: str, obj: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _run_module_stage(
    stage_name: str,
    module_name: str,
    argv: list[str],
    commands_path: str,
    status_path: str,
) -> StageResult:
    started = time.time()
    command_display = f"python -m {module_name} {' '.join(argv)}".strip()
    with open(commands_path, "a", encoding="utf-8") as f:
        f.write(command_display + "\n")

    try:
        run_module_main(module_name, argv)
        ok = True
        message = "ok"
    except SystemExit as exc:  # runner 中正常返回也可能抛 SystemExit(0)
        code = exc.code if isinstance(exc.code, int) else 1
        ok = code == 0
        message = f"exit={code}"
    except Exception as exc:  # noqa: BLE001
        ok = False
        message = f"error={exc}"

    ended = time.time()
    status = {
        "stage": stage_name,
        "module": module_name,
        "argv": argv,
        "ok": ok,
        "message": message,
        "started_at": started,
        "ended_at": ended,
        "elapsed_s": ended - started,
    }
    _append_jsonl(status_path, status)
    return StageResult(stage_name, ok, started, ended, message)


def _extract_experiment_rag_results(json_dir: str) -> str:
    """把 runExperiments 产出的 detailed JSONL 规整为 eval-generation 可用格式。"""
    target_candidates = []
    for file in sorted(os.listdir(json_dir)):
        if not file.endswith(".jsonl"):
            continue
        lower = file.lower()
        if "detailed_results_" in lower and (
            "_exp-hybrid.jsonl" in lower or "_baseline-bm25.jsonl" in lower
        ):
            target_candidates.append(os.path.join(json_dir, file))

    if not target_candidates:
        raise FileNotFoundError("未找到可用的 detailed_results_*_exp-hybrid.jsonl")

    src = target_candidates[0]
    rows = _LOADER.jsonl(src)
    out_path = os.path.join(json_dir, "rag_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            normalized = {
                "query": row.get("query", ""),
                "answer": row.get("answer", ""),
                "retrieved_terms": row.get("retrieved_terms", []),
                "sources": row.get("sources", []),
                "latency": {"total_ms": row.get("latency_ms", 0)},
            }
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
    return out_path


def _load_first_method_avg(path: str) -> dict[str, Any]:
    data = _LOADER.json(path)
    results = data.get("results", [])
    if not results:
        return {}
    return results[0].get("avg_metrics", {})


def _method_avg_after_stage(
    json_dir: str, output_arg: str, stage_ok: bool
) -> dict[str, Any]:
    """在子阶段结束后安全读取评测 JSON（失败或缺文件时返回空字典）。"""
    resolved = os.path.join(json_dir, os.path.basename(output_arg))
    if not stage_ok or not os.path.isfile(resolved):
        return {}
    try:
        return _load_first_method_avg(resolved)
    except Exception:  # noqa: BLE001
        return {}


def _build_ablation(
    queries_file: str,
    json_dir: str,
    commands_path: str,
    status_path: str,
) -> tuple[str, list[StageResult]]:
    """通过多次 eval-retrieval 自动构造 ablation_study.json。"""
    stage_results: list[StageResult] = []

    topk_variants: list[dict[str, Any]] = []
    for k in [3, 5, 10]:
        out = os.path.join(json_dir, f"_tmp_ablation_topk_{k}.json")
        stage_results.append(
            _run_module_stage(
                stage_name=f"ablation-topk-{k}",
                module_name="research.modelEvaluation.evalRetrieval",
                argv=[
                    "--queries",
                    queries_file,
                    "--methods",
                    "bm25plus",
                    "--topk",
                    str(k),
                    "--output",
                    out,
                ],
                commands_path=commands_path,
                status_path=status_path,
            )
        )
        avg = _method_avg_after_stage(json_dir, out, stage_results[-1].ok)
        topk_variants.append(
            {
                "topk": k,
                "recall@1": avg.get("recall@1", 0.0),
                "recall@3": avg.get("recall@3", 0.0),
                "recall@5": avg.get("recall@5", 0.0),
                "recall@10": avg.get("recall@10", 0.0),
                "mrr": avg.get("mrr", 0.0),
                "map": avg.get("map", 0.0),
            }
        )

    alpha_variants: list[dict[str, Any]] = []
    for alpha in [0.3, 0.5, 0.7, 0.85]:
        beta = round(1.0 - alpha, 2)
        out = os.path.join(json_dir, f"_tmp_ablation_alpha_{alpha}.json")
        stage_results.append(
            _run_module_stage(
                stage_name=f"ablation-alpha-{alpha}",
                module_name="research.modelEvaluation.evalRetrieval",
                argv=[
                    "--queries",
                    queries_file,
                    "--methods",
                    "hybrid-plus-weighted",
                    "--topk",
                    "10",
                    "--alpha",
                    str(alpha),
                    "--beta",
                    str(beta),
                    "--output",
                    out,
                ],
                commands_path=commands_path,
                status_path=status_path,
            )
        )
        avg = _method_avg_after_stage(json_dir, out, stage_results[-1].ok)
        alpha_variants.append(
            {
                "alpha": alpha,
                "beta": beta,
                "recall@5": avg.get("recall@5", 0.0),
                "map": avg.get("map", 0.0),
                "ndcg@10": avg.get("ndcg@10", 0.0),
            }
        )

    ablation = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "topk_ablation": {"variants": topk_variants},
        "alpha_ablation": {"variants": alpha_variants},
    }
    out_path = os.path.join(json_dir, "ablation_study.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ablation, f, ensure_ascii=False, indent=2)
    return out_path, stage_results


def _collect_numeric_rows(run_id: str, json_dir: str) -> tuple[list[dict], list[dict]]:
    tidy: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []

    all_methods_candidates = [
        os.path.join(json_dir, "all_methods.json"),
        os.path.join(json_dir, "full_eval", "all_methods.json"),
    ]
    all_methods = next((p for p in all_methods_candidates if os.path.isfile(p)), "")
    if all_methods:
        data = _LOADER.json(all_methods)
        for r in data.get("results", []):
            method = r.get("method", "unknown")
            for k, v in (r.get("avg_metrics") or {}).items():
                if isinstance(v, (int, float)):
                    tidy.append(
                        {
                            "run_id": run_id,
                            "dataset": "all_methods",
                            "method": method,
                            "metric": k,
                            "value": v,
                        }
                    )
            for k in ["recall@1", "recall@3", "recall@5", "recall@10"]:
                arr = r.get(k, [])
                if isinstance(arr, list):
                    for i, val in enumerate(arr):
                        if isinstance(val, (int, float)):
                            detailed.append(
                                {
                                    "run_id": run_id,
                                    "dataset": "all_methods_per_query",
                                    "method": method,
                                    "metric": k,
                                    "query_idx": i,
                                    "value": val,
                                }
                            )

    comparison = os.path.join(json_dir, "comparison_results.json")
    if os.path.isfile(comparison):
        data = _LOADER.json(comparison)
        for g in data.get("groups", []):
            method = g.get("group", "unknown")
            for scope in ["retrieval_metrics", "generation_metrics"]:
                for k, v in (g.get(scope) or {}).items():
                    if isinstance(v, (int, float)):
                        tidy.append(
                            {
                                "run_id": run_id,
                                "dataset": "comparison",
                                "method": method,
                                "metric": f"{scope}.{k}",
                                "value": v,
                            }
                        )

    for file in os.listdir(json_dir):
        if file.startswith("detailed_results_") and file.endswith(".jsonl"):
            for i, row in enumerate(_LOADER.jsonl(os.path.join(json_dir, file))):
                detailed.append(
                    {
                        "run_id": run_id,
                        "dataset": "experiment_detail",
                        "file": file,
                        "query_idx": i,
                        "query": row.get("query", ""),
                        "latency_ms": row.get("latency_ms", 0),
                        "recall@5": row.get("recall@5", 0),
                        "mrr": row.get("mrr", 0),
                        "term_hit_rate": (row.get("term_hit") or {}).get("rate", 0),
                        "source_citation_rate": (row.get("source_citation") or {}).get(
                            "rate", 0
                        ),
                    }
                )

    return tidy, detailed


def _write_csv(rows: list[dict[str, Any]], path: str) -> None:
    if not rows:
        _write_text(path, "")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_parquet(rows: list[dict[str, Any]], path: str) -> str:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return f"skip parquet (pandas unavailable: {exc})"

    try:
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"skip parquet ({exc})"


def _export_visualization_data(run_dir: str, json_dir: str) -> dict[str, Any]:
    run_id = os.path.basename(run_dir)
    exports_root = _ensure_dir(os.path.join(run_dir, "exports"))
    csv_dir = _ensure_dir(os.path.join(exports_root, "csv"))
    jsonl_dir = _ensure_dir(os.path.join(exports_root, "jsonl"))
    parquet_dir = _ensure_dir(os.path.join(exports_root, "parquet"))

    tidy, detailed = _collect_numeric_rows(run_id, json_dir)

    _write_csv(tidy, os.path.join(csv_dir, "tidy_metrics.csv"))
    _write_csv(detailed, os.path.join(csv_dir, "detailed_events.csv"))
    _write_jsonl(tidy, os.path.join(jsonl_dir, "tidy_metrics.jsonl"))
    _write_jsonl(detailed, os.path.join(jsonl_dir, "detailed_events.jsonl"))
    p1 = _write_parquet(tidy, os.path.join(parquet_dir, "tidy_metrics.parquet"))
    p2 = _write_parquet(detailed, os.path.join(parquet_dir, "detailed_events.parquet"))

    schema = {
        "tidy_metrics_fields": sorted({k for row in tidy for k in row.keys()}),
        "detailed_events_fields": sorted({k for row in detailed for k in row.keys()}),
        "parquet_status": {"tidy_metrics": p1, "detailed_events": p2},
    }
    _write_text(
        os.path.join(exports_root, "README.md"),
        "# Exports\n\n- csv/: 直接给可视化工具使用\n- jsonl/: 便于流式处理\n- parquet/: 列式分析\n",
    )
    with open(os.path.join(exports_root, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    return {"tidy_rows": len(tidy), "detailed_rows": len(detailed), "schema": schema}


def _resolve_queries_for_publish(json_dir: str, queries_override: str | None) -> str:
    """定稿生成报告时使用的查询集路径（优先 CLI，其次 all_methods 内记录）。"""
    if queries_override and os.path.isfile(queries_override):
        return queries_override
    am = os.path.join(json_dir, "all_methods.json")
    if os.path.isfile(am):
        data = _LOADER.json(am)
        qf = data.get("queries_file")
        if isinstance(qf, str) and os.path.isfile(qf):
            return qf
    rg = config.getReportsGenerationConfig()
    return os.path.join(config.EVALUATION_DIR, rg["queries_full_basename"])


def _snapshot_json_for_publish(json_dir: str, publish_root: str) -> None:
    out = _ensure_dir(os.path.join(publish_root, "json"))
    for name in (
        "all_methods.json",
        "ablation_study.json",
        "significance_test.json",
        "comparison_results.json",
    ):
        src = os.path.join(json_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out, name))


def _copy_defense_figures_to_publish(publish_root: str) -> None:
    src = os.path.join(
        config.FIGURES_DIR, config.getReportsGenerationConfig()["defense_output_subdir"]
    )
    if not os.path.isdir(src):
        return
    dst = os.path.join(publish_root, "figures", "defense")
    _ensure_dir(os.path.dirname(dst))
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _copy_supplementary_conclusions(log_run_dir: str, publish_root: str) -> None:
    """将 log 跑次 conclusions/ 下除标准三文件外的 Markdown 一并复制到定稿区。"""
    src_dir = os.path.join(log_run_dir, "conclusions")
    if not os.path.isdir(src_dir):
        return
    dst = _ensure_dir(os.path.join(publish_root, "conclusions"))
    reserved = {"key_findings.md", "limitations.md", "thesis_ready_summary.md"}
    for name in os.listdir(src_dir):
        if name in reserved or not name.endswith(".md"):
            continue
        src = os.path.join(src_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst, name))


def _write_conclusions(conclusions_parent_dir: str, json_dir: str) -> None:
    conclusions_dir = _ensure_dir(os.path.join(conclusions_parent_dir, "conclusions"))

    findings: list[str] = []
    all_methods_candidates = [
        os.path.join(json_dir, "all_methods.json"),
        os.path.join(json_dir, "full_eval", "all_methods.json"),
    ]
    all_methods = next((p for p in all_methods_candidates if os.path.isfile(p)), "")
    if all_methods:
        data = _LOADER.json(all_methods)
        rows = data.get("results", [])

        def _top_by(metric: str, n: int = 3) -> list[tuple[str, float]]:
            scored: list[tuple[str, float]] = []
            for r in rows:
                name = str(r.get("method", "unknown"))
                val = float((r.get("avg_metrics") or {}).get(metric, 0.0))
                scored.append((name, val))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:n]

        for metric, label in [
            ("recall@5", "Recall@5"),
            ("mrr", "MRR"),
            ("map", "MAP"),
            ("ndcg@10", "nDCG@10"),
        ]:
            top = _top_by(metric, 3)
            if top:
                parts = [f"`{m}`={v:.4f}" for m, v in top]
                findings.append(f"- 检索 Top3（{label}）：" + " / ".join(parts))

    significance = os.path.join(json_dir, "significance_test.json")
    if os.path.isfile(significance):
        data = _LOADER.json(significance)
        comps = data.get("paired_t_tests", [])
        sig_count = 0
        for comp in comps:
            r5 = (comp.get("metrics") or {}).get("recall@5", {})
            if bool(r5.get("significant_at_0.05")):
                sig_count += 1
        findings.append(f"- 显著性检验中 p<0.05 的比较数：{sig_count}")

    comparison = os.path.join(json_dir, "comparison_results.json")
    if os.path.isfile(comparison):
        data = _LOADER.json(comparison)
        groups = data.get("groups", [])
        if groups:
            best = max(
                groups,
                key=lambda g: float(
                    (g.get("generation_metrics") or {}).get("term_hit_rate", 0.0)
                ),
            )
            hit = (best.get("generation_metrics") or {}).get("term_hit_rate", 0.0)
            findings.append(
                f"- 生成术语命中率最高组：`{best.get('group')}` = {hit:.4f}"
            )

    if not findings:
        findings = ["- 本次运行未提取到可用指标，请检查上游评测是否成功。"]

    _write_text(
        os.path.join(conclusions_dir, "key_findings.md"),
        "# Key Findings\n\n" + "\n".join(findings) + "\n",
    )
    _write_text(
        os.path.join(conclusions_dir, "limitations.md"),
        "# Limitations\n\n- 若部分阶段失败，结论可能基于不完整数据。\n- 复杂公式质量仍受 OCR 与模型限制。\n",
    )
    _write_text(
        os.path.join(conclusions_dir, "thesis_ready_summary.md"),
        "# Thesis Ready Summary\n\n"
        "本轮实验在统一输出目录下完成了检索、生成、显著性与报告链路，"
        "可直接复用于论文实验章节的数据与图表引用。\n",
    )


def _copy_defense_figures_to_run(run_dir: str) -> None:
    src = os.path.join(
        config.FIGURES_DIR, config.getReportsGenerationConfig()["defense_output_subdir"]
    )
    if not os.path.isdir(src):
        return
    dst = os.path.join(run_dir, "figures", "defense")
    _ensure_dir(os.path.dirname(dst))
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def publish_to_reports(
    log_run_dir: str,
    queries_file: str | None = None,
    *,
    publish_root: str | None = None,
    commands_path: str | None = None,
    status_path: str | None = None,
) -> int:
    """从某次 log 跑次的 json/ 生成定稿到 outputs/reports/（或 publish_root）。"""
    publish_root = publish_root or config.REPORTS_PUBLISH_DIR
    json_dir = os.path.join(log_run_dir, "json")
    if not os.path.isdir(json_dir):
        print(f"[错误] 缺少 json 目录: {json_dir}")
        return 1
    all_methods_path = os.path.join(json_dir, "all_methods.json")
    if not os.path.isfile(all_methods_path):
        print(f"[错误] 缺少 {all_methods_path}")
        return 1
    queries = queries_file or _resolve_queries_for_publish(json_dir, None)
    pub_figures = _ensure_dir(os.path.join(publish_root, "figures"))
    comparison_path = os.path.join(json_dir, "comparison_results.json")
    report_argv = [
        "--results",
        all_methods_path,
        "--ablation",
        os.path.join(json_dir, "ablation_study.json"),
        "--significance",
        os.path.join(json_dir, "significance_test.json"),
        "--queries",
        queries,
        "--output",
        os.path.join(publish_root, "final_report.md"),
        "--figures",
        pub_figures,
    ]
    if os.path.isfile(comparison_path):
        report_argv.extend(["--comparison", comparison_path])

    stage_name = "publish-generate-report"
    if commands_path and status_path:
        sr = _run_module_stage(
            stage_name,
            "reports_generation.reports.generateReport",
            report_argv,
            commands_path,
            status_path,
        )
        if not sr.ok:
            return 1
    else:
        try:
            run_module_main(
                "reports_generation.reports.generateReport",
                report_argv,
            )
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            if code != 0:
                return code

    _copy_defense_figures_to_publish(publish_root)
    _write_conclusions(publish_root, json_dir)
    _copy_supplementary_conclusions(log_run_dir, publish_root)
    _snapshot_json_for_publish(json_dir, publish_root)
    print(f"[OK] 定稿已写入: {publish_root}")
    return 0


def _write_manifest(
    run_dir: str, status_path: str, export_meta: dict[str, Any]
) -> None:
    records = _LOADER.jsonl(status_path) if os.path.isfile(status_path) else []
    manifest = {
        "run_id": os.path.basename(run_dir),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": run_dir,
        "stages": records,
        "export_meta": export_meta,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass

    default_queries = os.path.join(
        config.EVALUATION_DIR,
        config.getReportsGenerationConfig()["queries_full_basename"],
    )
    parser = argparse.ArgumentParser(description="全量评测与报告总控")
    parser.add_argument(
        "--queries",
        default=default_queries,
        help="全量查询集路径",
    )
    parser.add_argument(
        "--publish-from-log",
        metavar="RUN_ID",
        default=None,
        help="不跑评测，仅从 outputs/log/<RUN_ID> 发布定稿到 outputs/reports/",
    )
    parser.add_argument(
        "--publish-root",
        default=None,
        help="定稿根目录（默认 config 中 reports_base_dir）",
    )
    parser.add_argument(
        "--skip-publish",
        action="store_true",
        help="定稿仍写在本次 log 目录（json/final_report.md 与 run_dir/figures），不写 outputs/reports/",
    )
    parser.add_argument(
        "--skip-norag",
        action="store_true",
        help="跳过 evalGenerationComparison 中的无检索基线",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="阶段失败后继续执行后续阶段",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="仅运行检索/消融/显著性/报告（跳过生成相关阶段）",
    )
    args = parser.parse_args(argv)

    if args.publish_from_log:
        log_run = args.publish_from_log
        if not os.path.isabs(log_run):
            log_run = os.path.join(config.LOG_BASE_DIR, log_run)
        q_override = args.queries if args.queries != default_queries else None
        queries_path = _resolve_queries_for_publish(
            os.path.join(log_run, "json"), q_override
        )
        return publish_to_reports(
            log_run,
            queries_path,
            publish_root=args.publish_root,
        )

    output_controller = config.getOutputController()
    run_dir = output_controller.get_run_dir()
    json_dir = output_controller.get_json_dir()
    run_trace_dir = _ensure_dir(os.path.join(run_dir, "run_trace"))
    figures_dir = _ensure_dir(os.path.join(run_dir, "figures"))
    _ensure_dir(os.path.join(run_dir, "tables"))

    commands_path = os.path.join(run_trace_dir, "commands.txt")
    status_path = os.path.join(run_trace_dir, "stage_status.jsonl")
    params_path = os.path.join(run_trace_dir, "params_snapshot.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(" Math-RAG 全量评测总控")
    print("=" * 60)
    print(f"run_dir: {run_dir}")

    stages: list[StageResult] = []

    def run_or_stop(stage: StageResult) -> bool:
        stages.append(stage)
        if stage.ok or args.continue_on_error:
            return True
        print(f"[停止] 阶段失败: {stage.name} ({stage.message})")
        return False

    # 1) 构建术语映射
    ok = run_or_stop(
        _run_module_stage(
            "build-term-mapping",
            "research.runners.buildTermMapping",
            [],
            commands_path,
            status_path,
        )
    )
    if not ok:
        return 1

    # 2) 全量检索评测（主表）
    # 注意：evalRetrieval 会把输出归一到 json_dir/<basename>
    all_methods_path = os.path.join(json_dir, "all_methods.json")
    ok = run_or_stop(
        _run_module_stage(
            "eval-retrieval-all-methods",
            "research.modelEvaluation.evalRetrieval",
            [
                "--queries",
                args.queries,
                "--methods",
                "bm25",
                "bm25plus",
                "vector",
                "hybrid-weighted",
                "hybrid-plus-weighted",
                "hybrid-plus-rrf",
                "--topk",
                "10",
                "--output",
                "all_methods.json",
                "--visualize",
            ],
            commands_path,
            status_path,
        )
    )
    if not ok:
        return 1

    # 3) 构造消融数据
    started = time.time()
    try:
        _build_ablation(args.queries, json_dir, commands_path, status_path)
        ablation_ok = True
        msg = "ok"
    except Exception as exc:  # noqa: BLE001
        ablation_ok = False
        msg = str(exc)
    stages.append(StageResult("build-ablation", ablation_ok, started, time.time(), msg))
    _append_jsonl(
        status_path,
        {
            "stage": "build-ablation",
            "ok": ablation_ok,
            "message": msg,
            "started_at": started,
            "ended_at": time.time(),
        },
    )
    if not ablation_ok and not args.continue_on_error:
        return 1

    # 4) 显著性检验
    ok = run_or_stop(
        _run_module_stage(
            "significance-test",
            "research.runners.significanceTest",
            [
                "--input",
                all_methods_path,
                "--output",
                os.path.join(json_dir, "significance_test.json"),
            ],
            commands_path,
            status_path,
        )
    )
    if not ok:
        return 1

    # 5) 端到端对比实验
    if not args.retrieval_only:
        ok = run_or_stop(
            _run_module_stage(
                "run-experiments",
                "research.runners.runExperiments",
                [
                    "--groups",
                    "norag",
                    "bm25",
                    "vector",
                    "hybrid",
                    "hybrid-rrf",
                    "--topk",
                    "5",
                    "--query-file",
                    args.queries,
                    "--output-dir",
                    json_dir,
                ],
                commands_path,
                status_path,
            )
        )
        if not ok:
            return 1

    # 6) 规整 rag_results 并跑生成评测
    if not args.retrieval_only:
        started = time.time()
        rag_results_path = ""
        try:
            rag_results_path = _extract_experiment_rag_results(json_dir)
            rag_ok = True
            msg = rag_results_path
        except Exception as exc:  # noqa: BLE001
            rag_ok = False
            msg = str(exc)
        stages.append(
            StageResult("extract-rag-results", rag_ok, started, time.time(), msg)
        )
        _append_jsonl(
            status_path,
            {
                "stage": "extract-rag-results",
                "ok": rag_ok,
                "message": msg,
                "started_at": started,
                "ended_at": time.time(),
            },
        )
        if not rag_ok and not args.continue_on_error:
            return 1

        if rag_ok:
            ok = run_or_stop(
                _run_module_stage(
                    "eval-generation",
                    "research.modelEvaluation.evalGeneration",
                    [
                        "--results",
                        rag_results_path,
                        "--gold",
                        args.queries,
                        "--output",
                        "generation_metrics.json",
                    ],
                    commands_path,
                    status_path,
                )
            )
            if not ok:
                return 1

            comparison_argv = [
                "--rag-results",
                rag_results_path,
                "--queries",
                args.queries,
                "--all-methods",
                all_methods_path,
                "--output",
                "comparison_results.json",
            ]
            if args.skip_norag:
                comparison_argv.append("--skip-norag")
            ok = run_or_stop(
                _run_module_stage(
                    "eval-generation-comparison",
                    "research.runners.evalGenerationComparison",
                    comparison_argv,
                    commands_path,
                    status_path,
                )
            )
            if not ok:
                return 1

    # 7) 最终报告 + 图表（默认写入定稿目录 outputs/reports/）
    comparison_path = os.path.join(json_dir, "comparison_results.json")
    publish_root = args.publish_root or config.REPORTS_PUBLISH_DIR
    if args.skip_publish:
        report_output_md = os.path.join(json_dir, "final_report.md")
        report_figures_dir = figures_dir
    else:
        _ensure_dir(publish_root)
        report_output_md = os.path.join(publish_root, "final_report.md")
        report_figures_dir = _ensure_dir(os.path.join(publish_root, "figures"))

    report_argv = [
        "--results",
        all_methods_path,
        "--ablation",
        os.path.join(json_dir, "ablation_study.json"),
        "--significance",
        os.path.join(json_dir, "significance_test.json"),
        "--queries",
        args.queries,
        "--output",
        report_output_md,
        "--figures",
        report_figures_dir,
    ]
    if os.path.isfile(comparison_path):
        report_argv.extend(["--comparison", comparison_path])

    ok = run_or_stop(
        _run_module_stage(
            "generate-report",
            "reports_generation.reports.generateReport",
            report_argv,
            commands_path,
            status_path,
        )
    )
    if not ok:
        return 1

    # 8) 答辩图（单独处理逻辑：先跑，再复制进本次 log；定稿时同时复制到 outputs/reports/）
    _run_module_stage(
        "generate-defense-figures",
        "reports_generation.reports.generateDefenseFigures",
        [],
        commands_path,
        status_path,
    )
    _copy_defense_figures_to_run(run_dir)
    if not args.skip_publish:
        _copy_defense_figures_to_publish(publish_root)

    # 9) 导出可视化数据（仅 log）+ 结论与快照
    export_meta = _export_visualization_data(run_dir, json_dir)
    if args.skip_publish:
        _write_conclusions(run_dir, json_dir)
    else:
        _write_conclusions(publish_root, json_dir)
        _copy_supplementary_conclusions(run_dir, publish_root)
        _snapshot_json_for_publish(json_dir, publish_root)
    _write_manifest(run_dir, status_path, export_meta)

    print("\n完成：评测日志已写入")
    print(run_dir)
    if not args.skip_publish:
        print(f"定稿目录: {publish_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
