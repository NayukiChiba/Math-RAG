"""
生成模块

包含 RAG 问答流程所需的提示模板、Qwen 推理封装和端到端流程。
"""

from generation.promptTemplates import (
    SYSTEM_PROMPT,
    buildContext,
    buildMessages,
    buildPrompt,
    formatTermContext,
)
from generation.qwenInference import QwenInference


# 延迟导入 RagPipeline，避免在导入时触发 faiss 依赖
def __getattr__(name):
    if name == "RagPipeline":
        from generation.ragPipeline import RagPipeline

        return RagPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SYSTEM_PROMPT",
    "buildPrompt",
    "buildMessages",
    "buildContext",
    "formatTermContext",
    "QwenInference",
    "RagPipeline",
]
