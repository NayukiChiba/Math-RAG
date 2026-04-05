"""生成包统一导出。"""

from __future__ import annotations

from core.answerGeneration.localInference import LocalInference
from core.answerGeneration.promptTemplates import (
    SYSTEM_PROMPT,
    buildContext,
    buildMessages,
    buildPrompt,
    buildPromptJinja2,
    formatTermContext,
)


def _raise_optional_import_error(feature: str, error: Exception) -> None:
    raise RuntimeError(
        f"{feature} 依赖未安装或导入失败，请先安装可选依赖后再使用。"
    ) from error


try:
    from core.answerGeneration.ragPipeline import RagPipeline, loadQueries, saveResults

    _RAG_IMPORT_ERROR: Exception | None = None
except Exception as error:  # noqa: BLE001
    _RAG_IMPORT_ERROR = error

    class RagPipeline:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _raise_optional_import_error("RagPipeline", _RAG_IMPORT_ERROR)

    def loadQueries(*args, **kwargs):  # type: ignore[no-redef]
        _raise_optional_import_error("loadQueries", _RAG_IMPORT_ERROR)

    def saveResults(*args, **kwargs):  # type: ignore[no-redef]
        _raise_optional_import_error("saveResults", _RAG_IMPORT_ERROR)


try:
    from core.answerGeneration.webui import chat, createUI, getGeneratorInstance
    from core.answerGeneration.webui import main as run_webui

    _WEBUI_IMPORT_ERROR: Exception | None = None
except Exception as error:  # noqa: BLE001
    _WEBUI_IMPORT_ERROR = error

    def getGeneratorInstance(*args, **kwargs):  # type: ignore[no-redef]
        _raise_optional_import_error("getGeneratorInstance", _WEBUI_IMPORT_ERROR)

    def chat(*args, **kwargs):  # type: ignore[no-redef]
        _raise_optional_import_error("chat", _WEBUI_IMPORT_ERROR)

    def createUI(*args, **kwargs):  # type: ignore[no-redef]
        _raise_optional_import_error("createUI", _WEBUI_IMPORT_ERROR)

    def run_webui(*args, **kwargs):  # type: ignore[no-redef]
        _raise_optional_import_error("run_webui", _WEBUI_IMPORT_ERROR)


__all__ = [
    "SYSTEM_PROMPT",
    "buildPrompt",
    "buildMessages",
    "buildContext",
    "buildPromptJinja2",
    "formatTermContext",
    "LocalInference",
    "RagPipeline",
    "loadQueries",
    "saveResults",
    "getGeneratorInstance",
    "chat",
    "createUI",
    "run_webui",
]
