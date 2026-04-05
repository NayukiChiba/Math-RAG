"""数据生成包统一导出。"""

from dataGen.clean_failed_ocr import main as clean_failed_ocr
from dataGen.data_gen import _generate_for_book as generate_term_data_for_book
from dataGen.data_gen import main as generate_term_data
from dataGen.extract_terms_from_ocr import (
    _extract_terms_for_book as extract_terms_for_book,
)
from dataGen.extract_terms_from_ocr import main as extract_terms
from dataGen.filter_terms import call_llm, filter_terms_batch, filter_terms_file
from dataGen.pix2text_ocr import main as run_ocr

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "call_llm",
    "clean_failed_ocr",
    "extract_terms",
    "extract_terms_for_book",
    "filter_terms_batch",
    "filter_terms_file",
    "generate_term_data",
    "generate_term_data_for_book",
    "run_ocr",
]
