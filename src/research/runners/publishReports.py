"""从已有 outputs/log/<run_id> 发布定稿到 outputs/reports/（不重跑评测）。"""

from __future__ import annotations

import argparse
import os
import sys

from core import config


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
    parser = argparse.ArgumentParser(
        description="从 log 跑次发布定稿（final_report、figures、conclusions、json 快照）"
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="outputs/log 下的目录名，例如 20260406_164049",
    )
    parser.add_argument(
        "--queries",
        default=default_queries,
        help="覆盖查询集路径（默认优先使用 all_methods.json 内 queries_file）",
    )
    parser.add_argument(
        "--publish-root",
        default=None,
        help="定稿根目录（默认 config [paths].reports_base_dir）",
    )
    args = parser.parse_args(argv)

    from research.runners.fullReports import (  # noqa: PLC0415
        _resolve_queries_for_publish,
        publish_to_reports,
    )

    log_run = os.path.join(config.LOG_BASE_DIR, args.run_id)
    json_dir = os.path.join(log_run, "json")
    q_override = args.queries if args.queries != default_queries else None
    queries_path = _resolve_queries_for_publish(json_dir, q_override)
    return publish_to_reports(
        log_run,
        queries_path,
        publish_root=args.publish_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
