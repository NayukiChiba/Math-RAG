"""FileLoader 统一加载器测试。"""

import json
import os
import pickle
import tempfile
import unittest

from utils import FileLoader, getFileLoader


class TestFileLoader(unittest.TestCase):
    """验证 FileLoader 命名、实例方法和复用行为。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = self.tmp.name

        self.json_path = os.path.join(self.base, "sample.json")
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump({"name": "math-rag", "value": 53}, f, ensure_ascii=False)

        self.jsonl_path = os.path.join(self.base, "sample.jsonl")
        with open(self.jsonl_path, "w", encoding="utf-8") as f:
            f.write('{"id": 1, "q": "导数"}\n')
            f.write('{"id": 2, "q": "矩阵"}\n')

        self.toml_path = os.path.join(self.base, "sample.toml")
        with open(self.toml_path, "w", encoding="utf-8") as f:
            f.write("[paths]\n")
            f.write('outputs_dir = "outputs"\n')

        self.pickle_path = os.path.join(self.base, "sample.pkl")
        with open(self.pickle_path, "wb") as f:
            pickle.dump({"ok": True, "count": 2}, f)

    def tearDown(self):
        self.tmp.cleanup()

    def test_class_name_and_instance_api(self):
        loader = FileLoader()
        self.assertEqual(loader.__class__.__name__, "FileLoader")
        self.assertTrue(callable(loader.json))
        self.assertTrue(callable(loader.jsonl))
        self.assertTrue(callable(loader.toml))
        self.assertTrue(callable(loader.pickle))

    def test_singleton_reuse(self):
        loader1 = getFileLoader()
        loader2 = getFileLoader()
        self.assertIs(loader1, loader2)

    def test_load_json(self):
        loader = FileLoader()
        data = loader.json(self.json_path)
        self.assertEqual(data["name"], "math-rag")
        self.assertEqual(data["value"], 53)

    def test_load_jsonl(self):
        loader = FileLoader()
        rows = loader.jsonl(self.jsonl_path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["q"], "导数")
        self.assertEqual(rows[1]["q"], "矩阵")

    def test_load_toml(self):
        loader = FileLoader()
        data = loader.toml(self.toml_path)
        self.assertIn("paths", data)
        self.assertEqual(data["paths"]["outputs_dir"], "outputs")

    def test_load_pickle(self):
        loader = FileLoader()
        data = loader.pickle(self.pickle_path)
        self.assertTrue(data["ok"])
        self.assertEqual(data["count"], 2)


if __name__ == "__main__":
    unittest.main()
