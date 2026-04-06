"""全链路 e2e 占位：当前无真实断言，见 tests/README.md 说明。"""

import pytest

pytestmark = pytest.mark.e2e


def test_full_rag_placeholder():
    pytest.skip(
        "e2e 占位：尚未实现端到端 RAG 断言；"
        "实现后删除本跳过并写入用例。"
        "环境变量 MATHRAG_RUN_E2E 仅作将来筛选重测试预留，与当前跳过无耦合。"
    )
