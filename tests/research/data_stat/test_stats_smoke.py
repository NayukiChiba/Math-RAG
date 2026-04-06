"""research.dataStat 统计构建与格式化轻量回归（对应 PR #63 审查建议）。"""

from __future__ import annotations

from research.dataStat import (
    buildStatistics,
    calculatePercentiles,
    formatStatistics,
)


def test_build_statistics_from_chunk_fixture(chunk_snapshot_dir):
    stats = buildStatistics(str(chunk_snapshot_dir))
    assert isinstance(stats, dict)
    assert "summary" in stats
    assert stats["summary"]["validFiles"] >= 1
    assert stats["summary"]["totalTerms"] >= 1
    assert "duplicates" in stats


def test_format_statistics_stable(chunk_snapshot_dir):
    stats = buildStatistics(str(chunk_snapshot_dir))
    a = formatStatistics(stats)
    b = formatStatistics(stats)
    assert a == b
    assert a["summary"] == stats["summary"]


def test_calculate_percentiles_basic():
    out = calculatePercentiles([1.0, 2.0, 3.0, 4.0, 100.0])
    assert "p25" in out and "p50" in out
    assert out["min"] == 1.0 and out["max"] == 100.0
