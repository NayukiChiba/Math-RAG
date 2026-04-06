"""报告生成 quick_eval 包可导入。"""


def test_quick_eval_module():
    import reports_generation.quick_eval.quickEval as qe

    assert hasattr(qe, "main") or hasattr(qe, "buildParser")
