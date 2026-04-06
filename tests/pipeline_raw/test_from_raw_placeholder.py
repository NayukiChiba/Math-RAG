"""从 raw PDF 起步的慢测占位（OCR / 抽取 / 入库）。"""

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.raw_pipeline]


@pytest.mark.skip(reason="本地按需实现：需真实 PDF 与较长运行时间")
def test_ingest_from_raw_pdf_placeholder():
    assert False
