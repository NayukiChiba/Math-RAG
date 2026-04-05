"""项目级统一导出。"""

from core import answerGeneration as answerGeneration
from core import config as config
from core import dataGen as dataGen
from core import retrieval as retrieval
from core import runners as runners
from core.mathRag import build_parser, main

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "build_parser",
    "config",
    "dataGen",
    "answerGeneration",
    "main",
    "retrieval",
    "runners",
]
