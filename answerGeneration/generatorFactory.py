"""
推理引擎工厂模块

根据 config.toml 中的 engine 配置，动态创建对应的推理实例。
消除 webui / ragPipeline / experiments 中的重复引擎选择逻辑。
"""

from __future__ import annotations

import config


def createGenerator():
    """
    根据 config.getGenerationConfig() 中的 engine 字段
    创建并返回对应的推理引擎实例。

    Returns:
        ApiInference 或 LocalInference 实例
    """
    engine = config.getGenerationConfig().get("engine", "local")

    if engine == "api":
        from answerGeneration.apiInference import ApiInference

        return ApiInference()

    from answerGeneration.localInference import LocalInference

    return LocalInference()
