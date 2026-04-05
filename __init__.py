"""项目级统一导出。"""

from src import answerGeneration as answerGeneration
from src import config as config
from src import dataGen as dataGen
from src import dataStat as dataStat
from src import evaluationData as evaluationData
from src import modelEvaluation as modelEvaluation
from src import retrieval as retrieval
from src import runners as runners
from src.mathRag import build_parser, main

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "build_parser",
    "config",
    "dataGen",
    "dataStat",
    "modelEvaluation",
    "answerGeneration",
    "evaluationData",
    "main",
    "retrieval",
    "runners",
]
