"""研究线命令统一路由。

映射自 src/research/cli/parser.py。所有命令通过任务管理器异步执行，
透传参数到对应 runner 的 main() 入口。
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from webui.backend.schemas import ResearchCommandRequest, TaskRef
from webui.backend.taskManager import getTaskManager

router = APIRouter()


# 命令 → 模块映射（与 research/cli/parser.py 保持一致）
_COMMAND_MODULES: dict[str, str] = {
    "generate-queries": "research.evaluationData.generateQueries",
    "build-term-mapping": "research.runners.buildTermMapping",
    "eval-retrieval": "research.modelEvaluation.evalRetrieval",
    "experiments": "research.runners.runExperiments",
    "eval-generation": "research.modelEvaluation.evalGeneration",
    "eval-generation-comparison": "research.runners.evalGenerationComparison",
    "significance-test": "research.runners.significanceTest",
    "report": "reports_generation.reports.generateReport",
    "full-reports": "research.runners.fullReports",
    "publish-reports": "research.runners.publishReports",
    "quick-eval": "reports_generation.quick_eval.quickEval",
    "defense-figures": "reports_generation.reports.generateDefenseFigures",
    "add-missing-terms": "research.runners.addMissingTerms",
}


@router.get("/commands")
def listCommands() -> dict[str, str]:
    """返回可用命令及其对应的底层模块。"""
    return dict(_COMMAND_MODULES)


@router.post("/{command}", response_model=TaskRef)
async def runResearchCommand(command: str, req: ResearchCommandRequest) -> TaskRef:
    """启动研究线命令。

    - `command` 必须在 _COMMAND_MODULES 中；stats 为特殊内置命令。
    - `req.args` 为透传给 runner main() 的命令行参数（列表形式）。
    """
    if command == "stats":
        return await _submitStats()

    if command not in _COMMAND_MODULES:
        raise HTTPException(status_code=404, detail=f"未知研究线命令: {command}")

    moduleName = _COMMAND_MODULES[command]
    passthrough = req.args or []

    def run() -> dict:
        from core.cli.runner import run_module_main

        run_module_main(moduleName, passthrough)
        return {"ok": True, "module": moduleName}

    taskId = await getTaskManager().submit(
        command=f"research.{command}",
        target=run,
        args={"args": passthrough, "module": moduleName},
    )
    return TaskRef(taskId=taskId)


async def _submitStats() -> TaskRef:
    """stats 命令直接调用 research.dataStat.run_statistics。"""

    def run() -> dict:
        from research.dataStat import run_statistics

        run_statistics()
        return {"ok": True}

    taskId = await getTaskManager().submit(
        command="research.stats",
        target=run,
    )
    return TaskRef(taskId=taskId)
