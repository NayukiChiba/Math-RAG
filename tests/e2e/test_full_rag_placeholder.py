"""全链路 e2e：需真实索引与模型时手动开启。"""

import os

import pytest

pytestmark = pytest.mark.e2e


def test_full_rag_placeholder():
    if not os.environ.get("MATHRAG_RUN_E2E"):
        pytest.skip("设置环境变量 MATHRAG_RUN_E2E=1 后可在此扩展全链路用例")
    pytest.skip("TODO: 全链路 RAG 断言尚未实现；实现真实用例后删除本跳过并写入断言")
