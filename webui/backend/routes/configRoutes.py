"""config.toml 读写路由。

使用最小依赖（标准库 tomllib + 自己写的 TOML 简易写出）避免引入额外包。
仅允许修改已在文件中存在的 section，避免误创字段。
"""

from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from webui.backend.schemas import ConfigPatchRequest

router = APIRouter()


EngineName = Literal["local", "api"]


class EnginesState(BaseModel):
    """三处 engine 的当前值。"""

    ocr: EngineName
    dataGen: EngineName
    rag: EngineName


class EnginesPatch(BaseModel):
    """只修改填入的字段；None 表示不动。"""

    ocr: EngineName | None = None
    dataGen: EngineName | None = None
    rag: EngineName | None = None


def _configPath() -> str:
    from core import config

    return config.CONFIG_TOML


@router.get("")
def getFullConfig() -> dict[str, Any]:
    """返回完整的 config.toml（已解析为 dict）。"""
    path = _configPath()
    if not os.path.isfile(path):
        return {}

    try:
        import tomllib
    except Exception:
        import tomli as tomllib  # type: ignore

    with open(path, "rb") as f:
        return tomllib.load(f)


@router.patch("")
def patchConfig(req: ConfigPatchRequest) -> dict[str, Any]:
    """修改某个 section 的部分字段。

    为避免 TOML 格式损失（注释、顺序），采用"逐行匹配 key 替换值"的策略：
    仅修改已存在的 key，未匹配的 key 在 section 末尾追加。
    """
    path = _configPath()
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"config.toml 不存在: {path}")

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    newLines, remaining = _replaceInSection(lines, req.section, req.updates)

    if remaining:
        # 将未匹配的 key 追加到 section 末尾
        newLines = _appendToSection(newLines, req.section, remaining)

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(newLines)

    _clearConfigCache()
    return getFullConfig()


# ── engines：快捷读写三处 engine ────────────────────────────────────


@router.get("/engines", response_model=EnginesState)
def getEngines() -> EnginesState:
    """读取 OCR / 数据生成 / RAG 回答当前各自使用的引擎。"""
    data = getFullConfig()
    ocrEngine = _normalize(str(_dig(data, "ocr", "engine", default="local")))
    # 术语生成（[model]）目前只实现 API 分支
    dataGenEngine: EngineName = "api"
    ragEngine = _normalize(str(_dig(data, "generation", "engine", default="api")))
    return EnginesState(ocr=ocrEngine, dataGen=dataGenEngine, rag=ragEngine)


@router.patch("/engines", response_model=EnginesState)
def patchEngines(req: EnginesPatch) -> EnginesState:
    """批量切换 OCR / RAG 引擎（dataGen 固定 API，忽略）。"""
    if req.ocr is not None:
        _writeKey("ocr", "engine", req.ocr)
    if req.rag is not None:
        _writeKey("generation", "engine", req.rag)
        _resetRagSingleton()
    if req.dataGen is not None and req.dataGen != "api":
        raise HTTPException(
            status_code=400,
            detail="术语结构化生成当前只支持 API 引擎",
        )
    _clearConfigCache()
    return getEngines()


# ── 内部 ─────────────────────────────────────────────────────────────


def _normalize(v: str) -> EngineName:
    vv = v.strip().lower()
    return "api" if vv == "api" else "local"


def _dig(data: dict, *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _writeKey(section: str, key: str, value: Any) -> None:
    path = _configPath()
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"config.toml 不存在: {path}")
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    newLines, remaining = _replaceInSection(lines, section, {key: value})
    if remaining:
        newLines = _appendToSection(newLines, section, remaining)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(newLines)


def _clearConfigCache() -> None:
    """让 core.config 的 lru_cache 失效。"""
    try:
        from core import config as coreConfig

        coreConfig._get_config_data.cache_clear()  # type: ignore[attr-defined]
        coreConfig.getPathsConfig.cache_clear()  # type: ignore[attr-defined]
        coreConfig.getReportsGenerationConfig.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass


def _resetRagSingleton() -> None:
    """切换 RAG 引擎后，重置 ragRoutes 的 Pipeline 单例。"""
    try:
        from webui.backend.routes import ragRoutes

        ragRoutes._ragPipeline = None  # type: ignore[attr-defined]
    except Exception:
        pass


# ── 内部：极简 TOML 行级编辑 ────────────────────────────────────────


def _isSectionHeader(line: str, section: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("[") and stripped.rstrip() == f"[{section}]"


def _isAnySectionHeader(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("[") and stripped.endswith("]")


def _tomlValue(value: Any) -> str:
    """将 Python 值序列化为 TOML 右值。仅支持基本类型与一维数组。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_tomlValue(v) for v in value) + "]"
    raise ValueError(f"config 不支持的值类型: {type(value).__name__}")


def _replaceInSection(
    lines: list[str], section: str, updates: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """在指定 section 内替换已存在的 key；返回新行与尚未匹配的更新。"""
    newLines: list[str] = []
    inSection = False
    remaining = dict(updates)

    for line in lines:
        if _isSectionHeader(line, section):
            inSection = True
            newLines.append(line)
            continue

        if inSection and _isAnySectionHeader(line):
            inSection = False

        if inSection and "=" in line and not line.lstrip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in remaining:
                newValue = _tomlValue(remaining.pop(key))
                newLines.append(f"{key} = {newValue}\n")
                continue

        newLines.append(line)

    return newLines, remaining


def _appendToSection(
    lines: list[str], section: str, updates: dict[str, Any]
) -> list[str]:
    """将剩余 key 追加到 section 末尾（下一个 section 前或文件末尾）。"""
    result: list[str] = []
    inSection = False
    inserted = False

    for line in lines:
        if _isSectionHeader(line, section):
            inSection = True
            result.append(line)
            continue

        if inSection and _isAnySectionHeader(line) and not inserted:
            for key, value in updates.items():
                result.append(f"{key} = {_tomlValue(value)}\n")
            inserted = True
            inSection = False

        result.append(line)

    if not inserted:
        # section 可能在文件末尾
        if inSection:
            for key, value in updates.items():
                result.append(f"{key} = {_tomlValue(value)}\n")
        else:
            # section 不存在，追加新 section
            result.append(f"\n[{section}]\n")
            for key, value in updates.items():
                result.append(f"{key} = {_tomlValue(value)}\n")

    return result
