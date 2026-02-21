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
from generation.ragPipeline import RagPipeline

__all__ = [
    "SYSTEM_PROMPT",
    "buildPrompt",
    "buildMessages",
    "buildContext",
    "formatTermContext",
    "QwenInference",
    "RagPipeline",
]
