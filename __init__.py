"""项目级统一导出。"""

import answerGeneration as answerGeneration
import config as config
import dataGen as dataGen
import dataStat as dataStat
import evaluationData as evaluationData
import modelEvaluation as modelEvaluation
import retrieval as retrieval
import scripts as scripts
from mathRag import build_parser, main

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
    "scripts",
]
