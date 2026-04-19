"""core.dataGen.termsGeneratorFactory engine 分发测试。

通过 monkeypatch 把 LocalInference / ApiInference 替换为桩，
验证工厂按 [terms_gen].engine 正确分发。
"""

from __future__ import annotations


def test_createTermsGenerator_dispatches_api(monkeypatch):
    from core import config
    from core.dataGen import termsGeneratorFactory

    monkeypatch.setattr(
        config,
        "getTermsModelConfig",
        lambda: {
            "engine": "api",
            "local_model_dir": "",
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 500,
        },
    )

    calls = {}

    class _StubApi:
        def __init__(self, temperature, topP, maxNewTokens, scope):
            calls["scope"] = scope
            calls["temperature"] = temperature
            calls["topP"] = topP
            calls["maxNewTokens"] = maxNewTokens

    import core.answerGeneration.apiInference as apiInfMod

    monkeypatch.setattr(apiInfMod, "ApiInference", _StubApi)
    # 重置单例，避免前序测试污染
    termsGeneratorFactory.resetTermsGenerator()

    inst = termsGeneratorFactory.createTermsGenerator()
    assert isinstance(inst, _StubApi)
    assert calls == {
        "scope": "terms",
        "temperature": 0.3,
        "topP": 0.9,
        "maxNewTokens": 500,
    }


def test_createTermsGenerator_local_requires_model_dir(monkeypatch):
    from core import config
    from core.dataGen import termsGeneratorFactory

    monkeypatch.setattr(
        config,
        "getTermsModelConfig",
        lambda: {
            "engine": "local",
            "local_model_dir": "",
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 500,
        },
    )
    termsGeneratorFactory.resetTermsGenerator()

    import pytest

    with pytest.raises(ValueError, match="local_model_dir"):
        termsGeneratorFactory.createTermsGenerator()


def test_createTermsGenerator_local_dispatch(monkeypatch):
    from core import config
    from core.dataGen import termsGeneratorFactory

    monkeypatch.setattr(
        config,
        "getTermsModelConfig",
        lambda: {
            "engine": "local",
            "local_model_dir": "/fake/model",
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 500,
        },
    )

    class _StubLocal:
        def __init__(self, modelDir, temperature, topP, maxNewTokens):
            self.modelDir = modelDir
            self.temperature = temperature
            self.topP = topP
            self.maxNewTokens = maxNewTokens

    import core.answerGeneration.localInference as localMod

    monkeypatch.setattr(localMod, "LocalInference", _StubLocal)
    termsGeneratorFactory.resetTermsGenerator()

    inst = termsGeneratorFactory.createTermsGenerator()
    assert isinstance(inst, _StubLocal)
    assert inst.modelDir == "/fake/model"

    # 再次调用应复用单例
    inst2 = termsGeneratorFactory.createTermsGenerator()
    assert inst2 is inst
