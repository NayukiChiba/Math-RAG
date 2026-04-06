"""产品线与研究线关键包可导入。"""


def test_import_core_config():
    from core import config

    assert config.PROCESSED_DIR or config.DATA_DIR


def test_import_corpus_builder():
    from core.retrieval.corpusBuilder.builder import buildCorpus

    assert callable(buildCorpus)


def test_import_rag_pipeline():
    from core.answerGeneration.ragPipeline import RagPipeline

    assert RagPipeline is not None


def test_import_research_metrics():
    from research.modelEvaluation.common import metrics

    assert hasattr(metrics, "calculateRecallAtK")


def test_import_reports_quick_eval_module():
    import reports_generation.quick_eval.quickEval as qe

    assert qe is not None


def test_pipelines_runrag_reexports():
    from core.runners.pipelines import runRag as rag

    assert rag.RagPipeline is not None
    assert callable(rag.loadQueries)
    assert callable(rag.saveResults)
    assert rag.config is not None
