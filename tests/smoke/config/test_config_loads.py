"""配置模块可读且返回结构化字典。"""


def test_retrieval_config_keys():
    from core import config

    cfg = config.getRetrievalConfig()
    assert isinstance(cfg, dict)
    assert "default_vector_model" in cfg or "out_of_scope_score_threshold" in cfg
