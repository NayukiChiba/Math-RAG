"""全链路 e2e：需真实索引与模型时手动开启。"""

import os

import pytest

pytestmark = pytest.mark.e2e


@pytest.mark.skipif(
    not os.environ.get("MATHRAG_RUN_E2E"),
    reason="设置环境变量 MATHRAG_RUN_E2E=1 后在此实现全链路断言",
)
def test_full_rag_placeholder():
    assert False, "在此接入真实 RAG 断言"
