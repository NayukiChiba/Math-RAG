"""runRag 管线模块与 core.runners.runRag 引用一致。"""

from core.runners import runRag as entry
from core.runners.pipelines import runRag as rag


def test_same_rag_pipeline_symbol():
    assert entry.rag.RagPipeline is rag.RagPipeline
