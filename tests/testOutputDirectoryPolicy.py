"""输出目录策略测试。

目标：保证同一进程的一次运行只使用一个时间目录。
"""

import os
import tempfile
import unittest

import config


class TestOutputDirectoryPolicy(unittest.TestCase):
    """验证 outputs/log/<timestamp>/json|text 的单目录约束。"""

    def setUp(self):
        self._orig_log_base = config.LOG_BASE_DIR
        self._orig_reports_base = config.REPORTS_BASE_DIR
        self._orig_run_log_dir = config._runLogDir
        self._orig_json_log_dir = config._jsonLogDir
        self._orig_text_log_dir = config._textLogDir

        self._tmp = tempfile.TemporaryDirectory()
        sandbox_log_base = os.path.join(self._tmp.name, "outputs", "log")

        # 重定向到沙箱，避免污染真实工作区。
        config.LOG_BASE_DIR = sandbox_log_base
        config.REPORTS_BASE_DIR = sandbox_log_base
        config._runLogDir = None
        config._jsonLogDir = None
        config._textLogDir = None

    def tearDown(self):
        config.LOG_BASE_DIR = self._orig_log_base
        config.REPORTS_BASE_DIR = self._orig_reports_base
        config._runLogDir = self._orig_run_log_dir
        config._jsonLogDir = self._orig_json_log_dir
        config._textLogDir = self._orig_text_log_dir
        self._tmp.cleanup()

    def test_single_timestamp_folder_per_run(self):
        controller = config.getOutputController()
        run_dir_1 = config.getRunLogDir()
        run_dir_2 = config.getRunLogDir()
        json_dir_1 = controller.get_json_dir()
        json_dir_2 = config.getJsonLogDir()
        text_dir_1 = config.getTextLogDir()
        text_dir_2 = config.getTextLogDir()

        self.assertEqual(run_dir_1, run_dir_2)
        self.assertEqual(json_dir_1, json_dir_2)
        self.assertEqual(text_dir_1, text_dir_2)

        self.assertEqual(json_dir_1, os.path.join(run_dir_1, "json"))
        self.assertEqual(text_dir_1, os.path.join(run_dir_1, "text"))

        # 同一次运行下，outputs/log 只能存在一个时间目录。
        ts_dirs = [
            d
            for d in os.listdir(config.LOG_BASE_DIR)
            if os.path.isdir(os.path.join(config.LOG_BASE_DIR, d))
        ]
        self.assertEqual(len(ts_dirs), 1)

    def test_normalize_outputs_always_stay_in_single_run_folder(self):
        controller = config.getOutputController()
        p1 = controller.normalize_json_path(
            "outputs/reports/old_name.json", "default.json"
        )
        p2 = controller.normalize_json_path("custom/new_metrics.json", "default.json")
        p3 = controller.normalize_text_path("outputs/reports/run.log", "default.log")

        run_dir = config.getRunLogDir()
        json_dir = config.getJsonLogDir()
        text_dir = config.getTextLogDir()

        self.assertTrue(p1.startswith(json_dir + os.sep))
        self.assertTrue(p2.startswith(json_dir + os.sep))
        self.assertTrue(p3.startswith(text_dir + os.sep))

        self.assertIn(run_dir, p1)
        self.assertIn(run_dir, p2)
        self.assertIn(run_dir, p3)

        # 归一化后路径中不应再出现 reports 目录。
        self.assertNotIn(f"{os.sep}reports{os.sep}", p1)
        self.assertNotIn(f"{os.sep}reports{os.sep}", p2)
        self.assertNotIn(f"{os.sep}reports{os.sep}", p3)


if __name__ == "__main__":
    unittest.main()
