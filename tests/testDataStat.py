"""dataStat 功能测试。"""

import json
import os
import tempfile
import unittest

import config
from dataStat import (
    HAS_MATPLOTLIB,
    analyzeDefinitions,
    buildStatistics,
    calculateFieldStats,
    calculatePercentiles,
    createVisualization,
    formatStatistics,
    loadJsonFile,
    run_statistics,
)


def _sample_term(term: str, subject: str) -> dict:
    return {
        "id": f"id-{term}",
        "term": term,
        "aliases": [f"{term}-别名"],
        "sense_id": "1",
        "subject": subject,
        "definitions": [
            {
                "type": "strict",
                "text": f"{term} 的严格定义",
                "conditions": "x>0",
                "notation": "x",
                "reference": "book-p1",
            }
        ],
        "notation": "x",
        "formula": "$x+y$",
        "usage": "示例",
        "applications": ["应用A"],
        "disambiguation": "",
        "related_terms": ["相关术语"],
        "sources": ["book"],
        "search_keys": [term],
        "lang": "zh",
        "confidence": 0.9,
    }


class TestDataStat(unittest.TestCase):
    """覆盖 dataStat 的读取、统计、格式化、可视化与入口调用。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = self.tmp.name

        self.chunk_dir = os.path.join(self.base, "chunk")
        self.stats_dir = os.path.join(self.base, "stats")
        os.makedirs(self.chunk_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)

        book_a = os.path.join(self.chunk_dir, "bookA")
        book_b = os.path.join(self.chunk_dir, "bookB")
        os.makedirs(book_a, exist_ok=True)
        os.makedirs(book_b, exist_ok=True)

        with open(os.path.join(book_a, "t1.json"), "w", encoding="utf-8") as f:
            json.dump(_sample_term("导数", "数学分析"), f, ensure_ascii=False)
        with open(os.path.join(book_b, "t2.json"), "w", encoding="utf-8") as f:
            json.dump(_sample_term("导数", "高等代数"), f, ensure_ascii=False)

        with open(os.path.join(book_b, "bad.json"), "w", encoding="utf-8") as f:
            f.write("{bad json")

        self.orig_chunk_dir = config.CHUNK_DIR
        self.orig_stats_dir = config.STATS_DIR
        config.CHUNK_DIR = self.chunk_dir
        config.STATS_DIR = self.stats_dir

    def tearDown(self):
        config.CHUNK_DIR = self.orig_chunk_dir
        config.STATS_DIR = self.orig_stats_dir
        self.tmp.cleanup()

    def test_load_json_file(self):
        valid_path = os.path.join(self.chunk_dir, "bookA", "t1.json")
        invalid_path = os.path.join(self.chunk_dir, "bookB", "bad.json")
        self.assertIsInstance(loadJsonFile(valid_path), dict)
        self.assertIsNone(loadJsonFile(invalid_path))

    def test_field_and_definition_helpers(self):
        field_stats = {
            "present": 0,
            "missing": 0,
            "lengths": [],
            "itemLengths": [],
            "itemCounts": [],
        }
        calculateFieldStats({"aliases": ["a", "bc"]}, "aliases", field_stats)
        self.assertEqual(field_stats["present"], 1)
        self.assertIn(2, field_stats["lengths"])

        def_stats = analyzeDefinitions(_sample_term("积分", "数学分析")["definitions"])
        self.assertEqual(def_stats["count"], 1)
        self.assertEqual(def_stats["hasConditions"], 1)
        self.assertEqual(def_stats["hasNotation"], 1)
        self.assertEqual(def_stats["hasReference"], 1)

    def test_build_and_format_statistics(self):
        raw_stats = buildStatistics(self.chunk_dir)
        self.assertEqual(raw_stats["summary"]["totalFiles"], 3)
        self.assertEqual(raw_stats["summary"]["validFiles"], 2)
        self.assertEqual(raw_stats["summary"]["invalidFiles"], 1)
        self.assertIn("导数", raw_stats["duplicates"])

        formatted = formatStatistics(raw_stats)
        self.assertIn("fieldCoverage", formatted)
        self.assertIn("definitionsAnalysis", formatted)
        self.assertEqual(formatted["duplicates"]["count"], 1)

    def test_percentiles(self):
        p = calculatePercentiles([1, 2, 3, 4, 5])
        self.assertEqual(p["min"], 1)
        self.assertEqual(p["max"], 5)
        self.assertIn("p50", p)

    def test_visualization_smoke(self):
        raw_stats = buildStatistics(self.chunk_dir)
        createVisualization(raw_stats, self.stats_dir)
        if HAS_MATPLOTLIB:
            self.assertTrue(
                os.path.isdir(os.path.join(self.stats_dir, "visualizations"))
            )

    def test_run_statistics_end_to_end(self):
        run_statistics()
        output_json = os.path.join(self.stats_dir, "chunkStatistics.json")
        self.assertTrue(os.path.isfile(output_json))
        with open(output_json, encoding="utf-8") as f:
            report = json.load(f)
        self.assertIn("summary", report)
        self.assertEqual(report["summary"]["totalFiles"], 3)


if __name__ == "__main__":
    unittest.main()
