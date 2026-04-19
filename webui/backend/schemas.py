"""API 请求/响应 Pydantic 模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ── 通用 ────────────────────────────────────────────────────────────────


class TaskRef(BaseModel):
    """长任务启动后的引用。"""

    taskId: str = Field(description="任务唯一 ID（UUID）")


class ErrorResponse(BaseModel):
    """错误响应。"""

    detail: str


# ── RAG 问答 ────────────────────────────────────────────────────────────


class RagQueryRequest(BaseModel):
    """单次问答请求。"""

    query: str
    useRag: bool = True
    temperature: float | None = None
    topP: float | None = None
    maxNewTokens: int | None = None


class RagQueryResponse(BaseModel):
    """单次问答响应（非流式）。"""

    query: str
    answer: str
    retrievedTerms: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    latency: dict[str, int]


class RagBatchRequest(BaseModel):
    """批量问答请求。"""

    queries: list[str]
    useRag: bool = True
    temperature: float | None = None
    topP: float | None = None
    maxNewTokens: int | None = None


# ── 入库与索引 ──────────────────────────────────────────────────────────


class IngestRequest(BaseModel):
    """PDF 入库请求。"""

    pdf: str = Field(description="PDF 路径或 data/raw 中的文件名")
    ocrStartPage: int | None = None
    extractStartPage: int | None = None
    generateStartPage: int | None = None
    skipGeneration: bool = False
    skipIndex: bool = False
    rebuildIndex: bool = False
    skipBm25: bool = False
    skipBm25plus: bool = False
    skipVector: bool = False
    vectorModel: str | None = None
    batchSize: int = 32


class BuildIndexRequest(BaseModel):
    """重建索引请求。"""

    rebuild: bool = False
    skipBm25: bool = False
    skipBm25plus: bool = False
    skipVector: bool = False
    vectorModel: str | None = None
    batchSize: int = 32


class IndexStatus(BaseModel):
    """索引状态。"""

    corpusExists: bool
    corpusDocCount: int | None = None
    bm25Exists: bool
    bm25plusExists: bool
    vectorExists: bool
    retrievalDir: str


# ── 数据列表 ────────────────────────────────────────────────────────────


class RawPdfInfo(BaseModel):
    """data/raw 下的 PDF 文件信息。"""

    name: str
    sizeBytes: int
    modifiedAt: str


class ProcessedInfo(BaseModel):
    """data/processed 子目录状态。"""

    ocrBooks: list[str]
    termsBooks: list[str]
    chunkBooks: list[str]


# ── 研究线 ──────────────────────────────────────────────────────────────


class ResearchCommandRequest(BaseModel):
    """研究线通用命令启动请求。"""

    args: list[str] = Field(
        default_factory=list,
        description="透传给目标 runner 的命令行参数",
    )


# ── 任务管理 ────────────────────────────────────────────────────────────

TaskStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]


class TaskInfo(BaseModel):
    """任务详细信息。"""

    taskId: str
    command: str
    status: TaskStatus
    createdAt: str
    startedAt: str | None = None
    finishedAt: str | None = None
    progress: float | None = None
    message: str | None = None
    args: dict[str, Any] | None = None
    errorMessage: str | None = None
    logTail: list[str] = Field(default_factory=list, description="最后若干条日志")


# ── 配置 ────────────────────────────────────────────────────────────────


class ConfigPatchRequest(BaseModel):
    """配置修改请求：section.key = value 扁平 dict。"""

    section: str
    updates: dict[str, Any]


# ── 报告与日志 ──────────────────────────────────────────────────────────


class ReportRunInfo(BaseModel):
    """一次运行（outputs/log/<run_id>）的汇总信息。"""

    runId: str
    path: str
    hasFinalReport: bool
    hasComparison: bool
    hasFullEval: bool
    createdAt: str


class FigureInfo(BaseModel):
    """图表文件信息。"""

    relPath: str
    sizeBytes: int
    modifiedAt: str
