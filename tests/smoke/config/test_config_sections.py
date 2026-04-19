"""core.config 中新增的分段读取函数测试。

覆盖：
- getApiConfig(scope) 的三种 scope
- getTermsModelConfig / getEmbeddingConfig / getRerankerConfig
- getGenerationConfig 读的是 [rag_gen]
"""

from __future__ import annotations


def test_getGenerationConfig_reads_rag_gen():
    from core import config

    cfg = config.getGenerationConfig()
    assert isinstance(cfg, dict)
    # engine 字段必存在（有默认值 "api"）
    assert cfg.get("engine") in ("api", "local")
    # 关键参数键
    for key in (
        "max_context_chars",
        "max_chars_per_term",
        "temperature",
        "top_p",
        "max_new_tokens",
    ):
        assert key in cfg


def test_getApiConfig_rag_scope():
    from core import config

    cfg = config.getApiConfig("rag")
    assert set(cfg.keys()) >= {"api_base", "model", "api_key_env", "stream"}


def test_getApiConfig_terms_scope():
    from core import config

    cfg = config.getApiConfig("terms")
    assert set(cfg.keys()) >= {"api_base", "model", "api_key_env", "stream"}


def test_getApiConfig_ocr_scope():
    from core import config

    cfg = config.getApiConfig("ocr")
    assert set(cfg.keys()) >= {"api_base", "model", "api_key_env", "stream"}


def test_getApiConfig_default_scope_is_rag():
    """不带 scope 默认走 RAG，等价于 scope="rag"."""
    from core import config

    assert config.getApiConfig() == config.getApiConfig("rag")


def test_getTermsModelConfig_structure():
    from core import config

    cfg = config.getTermsModelConfig()
    assert cfg["engine"] in ("api", "local")
    # local / api 两种分支都需要的共享字段
    for key in (
        "local_model_dir",
        "api_base",
        "model",
        "api_key_env",
        "temperature",
        "top_p",
        "max_tokens",
    ):
        assert key in cfg


def test_getEmbeddingConfig_returns_local_model():
    from core import config

    cfg = config.getEmbeddingConfig()
    assert "local_model" in cfg
    assert isinstance(cfg["local_model"], str) and cfg["local_model"]


def test_getRerankerConfig_returns_local_model():
    from core import config

    cfg = config.getRerankerConfig()
    assert "local_model" in cfg
    assert isinstance(cfg["local_model"], str) and cfg["local_model"]
