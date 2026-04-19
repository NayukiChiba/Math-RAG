"""
术语结构化生成：engine 分发工厂

根据 config.toml 中 [terms_gen].engine 创建对应推理实例：
    - "api"   -> ApiInference(scope="terms")
    - "local" -> LocalInference(modelDir=[terms_gen].local_model_dir)

在同一进程内复用单例，避免本地模型反复加载。
"""

from __future__ import annotations

from core import config

_localInstance = None  # 本地推理单例，按 modelDir 复用


def createTermsGenerator():
    """按 [terms_gen].engine 创建推理实例。"""
    cfg = config.getTermsModelConfig()
    engine = str(cfg.get("engine", "api")).strip().lower()

    if engine == "local":
        return _getLocalInference(
            modelDir=cfg["local_model_dir"],
            temperature=cfg["temperature"],
            topP=cfg["top_p"],
            maxNewTokens=cfg["max_tokens"],
        )

    from core.answerGeneration.apiInference import ApiInference

    return ApiInference(
        temperature=cfg["temperature"],
        topP=cfg["top_p"],
        maxNewTokens=cfg["max_tokens"],
        scope="terms",
    )


def _getLocalInference(modelDir, temperature, topP, maxNewTokens):
    """获取本地推理单例（按 modelDir 复用）。"""
    global _localInstance

    from core.answerGeneration.localInference import LocalInference

    if not modelDir:
        raise ValueError(
            "术语生成 engine=local，但 [terms_gen].local_model_dir 未配置；"
            "请在 config.toml 中设置 local_model_dir 为本地模型目录。"
        )

    if (
        _localInstance is not None
        and getattr(_localInstance, "modelDir", None) == modelDir
    ):
        return _localInstance

    _localInstance = LocalInference(
        modelDir=modelDir,
        temperature=temperature,
        topP=topP,
        maxNewTokens=maxNewTokens,
    )
    return _localInstance


def resetTermsGenerator() -> None:
    """重置单例，供引擎切换后调用。"""
    global _localInstance
    _localInstance = None
