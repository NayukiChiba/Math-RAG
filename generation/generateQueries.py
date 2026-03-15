"""自动生成评测查询数据（独立脚本入口）。"""

from __future__ import annotations

from typing import Any

from generation.query_generation.cli import buildParser
from generation.query_generation.generator import (
    generateQueries as _generate_queries,
)
from generation.query_generation.generator import (
    normalizeSubject as _normalize_subject,
)
from generation.query_generation.ioOps import (
    loadAllTerms as _load_all_terms,
)
from generation.query_generation.ioOps import (
    loadExistingQueries as _load_existing_queries,
)
from generation.query_generation.ioOps import (
    mergeQueries as _merge_queries,
)
from generation.query_generation.ioOps import (
    saveQueries as _save_queries,
)
from generation.query_generation.runner import runGenerateQueries


def loadAllTerms(chunkDir: str) -> list[dict[str, Any]]:
    return _load_all_terms(chunkDir)


def normalizeSubject(subject: str) -> str:
    return _normalize_subject(subject)


def generateQueries(
    terms: list[dict[str, Any]],
    numPerSubject: dict[str, int] | None = None,
    minRelatedTerms: int = 1,
    useAll: bool = False,
    sampleRatio: float | None = None,
) -> list[dict[str, Any]]:
    return _generate_queries(
        terms,
        num_per_subject=numPerSubject,
        min_related_terms=minRelatedTerms,
        use_all=useAll,
        sample_ratio=sampleRatio,
    )


def loadExistingQueries(filepath: str) -> list[dict[str, Any]]:
    return _load_existing_queries(filepath)


def mergeQueries(
    existing: list[dict[str, Any]], generated: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return _merge_queries(existing, generated)


def saveQueries(queries: list[dict[str, Any]], filepath: str) -> None:
    _save_queries(queries, filepath)


def main() -> int:
    parser = buildParser()
    args = parser.parse_args()
    return runGenerateQueries(args)


if __name__ == "__main__":
    raise SystemExit(main())
