"""
RAG 提示模板模块

功能：
1. 定义数学名词问答场景的 system prompt
2. 从检索结果列表构建结构化上下文（按 rank 排序，控制总长度）
3. 生成完整的 system + user prompt 对（f-string 实现）
4. 提供 Jinja2 模板字符串常量，支持灵活渲染
5. 公式以 LaTeX 格式原样保留，来源字段包含在提示中

使用方法：
    from generation.promptTemplates import buildPrompt, buildMessages

    # 检索结果需包含 term、subject、text、source、page 字段
    retrievalResults = [
        {
            "rank": 1,
            "term": "一致收敛",
            "subject": "数学分析",
            "text": "术语: 一致收敛\\n定义1[strict]: ...",
            "source": "数学分析(第5版)上",
            "page": 123,
        }
    ]

    # 构建 {system, user, used_results}
    prompt = buildPrompt(query="什么是一致收敛？", retrievalResults=retrievalResults)
    print(prompt["system"])
    print(prompt["user"])

    # 构建 OpenAI 兼容的 messages 列表
    messages = buildMessages(query="什么是一致收敛？", retrievalResults=retrievalResults)
"""

import os
import sys
from pathlib import Path
from typing import Any

# 路径调整，支持直接运行和模块导入两种方式
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# ---- 配置加载 ----


def _loadGenerationConfig() -> dict[str, Any]:
    """
    从 config.toml 加载 [generation] 配置，缺失时使用默认值

    Returns:
        生成配置字典
    """
    defaults = {
        "max_context_chars": 2000,
        "max_chars_per_term": 800,
    }

    if not os.path.isfile(config.CONFIG_TOML):
        return defaults

    try:
        # 复用 config.py 中的 _load_toml
        data = config._load_toml(config.CONFIG_TOML)
    except Exception:
        return defaults

    gen_cfg = data.get("generation", {})
    return {
        "max_context_chars": gen_cfg.get(
            "max_context_chars", defaults["max_context_chars"]
        ),
        "max_chars_per_term": gen_cfg.get(
            "max_chars_per_term", defaults["max_chars_per_term"]
        ),
    }


_GEN_CFG = _loadGenerationConfig()

# 上下文总最大字符数（1.5B 模型约 2000 token，中文约 1.5 char/token，预留 prompt 开销）
MAX_CONTEXT_CHARS: int = _GEN_CFG["max_context_chars"]

# 单条术语文本的最大字符数，避免一条术语挤占整个上下文
MAX_CHARS_PER_TERM: int = _GEN_CFG["max_chars_per_term"]


# ---- System Prompt ----

SYSTEM_PROMPT = """你是一位专业的数学教学助手，专注于数学分析、高等代数、概率论等大学数学课程。

回答要求：
1. 基于提供的参考资料作答，优先使用资料中的定义与公式
2. 数学符号和公式使用 LaTeX 格式（行内公式用 $...$，行间公式用 $$...$$）
3. 回答结构清晰：先给出核心定义，再说明公式与用法，最后标注来源
4. 来源引用格式：【书名 第X页】
5. 若参考资料不足以完整回答，说明局限性，不要编造内容"""


# ---- Jinja2 模板字符串（需安装 jinja2 才能渲染） ----

# system prompt Jinja2 模板（当前等同于常量，预留插槽供后续扩展 few-shot 等）
JINJA2_SYSTEM_TEMPLATE = """你是一位专业的数学教学助手，专注于数学分析、高等代数、概率论等大学数学课程。

回答要求：
1. 基于提供的参考资料作答，优先使用资料中的定义与公式
2. 数学符号和公式使用 LaTeX 格式（行内公式用 $...$，行间公式用 $$...$$）
3. 回答结构清晰：先给出核心定义，再说明公式与用法，最后标注来源
4. 来源引用格式：【书名 第X页】
5. 若参考资料不足以完整回答，说明局限性，不要编造内容
{% if extra_instructions %}{{ extra_instructions }}{% endif %}"""

# user prompt Jinja2 模板，占位符：{{ context }}、{{ query }}
JINJA2_USER_TEMPLATE_WITH_CONTEXT = """\
请根据以下参考资料回答问题。

===参考资料===
{{ context }}
===参考资料结束===

问题：{{ query }}"""

JINJA2_USER_TEMPLATE_NO_CONTEXT = """\
问题：{{ query }}
（注：未检索到相关参考资料，请根据已知知识回答。）"""


# ---- 核心函数 ----


def formatTermContext(item: dict[str, Any]) -> str:
    """
    将单条检索结果格式化为上下文文本块

    Args:
        item: 检索结果，字段说明：
            - term (str): 术语名称
            - subject (str, 可选): 学科
            - text (str, 可选): 术语完整文本（由 buildCorpus 生成）
            - source (str, 可选): 来源书名
            - page (int, 可选): 页码

    Returns:
        格式化后的文本块字符串
    """
    term = item.get("term", "（未知术语）")
    subject = item.get("subject", "")
    source = item.get("source", "")
    page = item.get("page", None)
    text = item.get("text", "")

    # 构建来源标注
    if source and page is not None:
        sourceLabel = f"【{source} 第{page}页】"
    elif source:
        sourceLabel = f"【{source}】"
    else:
        sourceLabel = ""

    # 首行：学科 + 术语名
    header = f"[{subject}] {term}" if subject else term
    if sourceLabel:
        header = f"{header}  {sourceLabel}"

    lines = [header]

    # 追加术语文本内容
    if text:
        # 截断超长文本，避免单条挤占整个上下文
        if len(text) > MAX_CHARS_PER_TERM:
            truncatedText = text[:MAX_CHARS_PER_TERM] + "…（已截断）"
        else:
            truncatedText = text
        lines.append(truncatedText)

    return "\n".join(lines)


def buildContext(
    retrievalResults: list[dict[str, Any]],
    maxChars: int = MAX_CONTEXT_CHARS,
) -> tuple[str, list[dict[str, Any]]]:
    """
    从检索结果列表构建上下文字符串

    按 rank 升序拼接（rank 越小越优先），超出 maxChars 时停止追加。

    Args:
        retrievalResults: 检索结果列表
        maxChars: 上下文最大字符数（默认 MAX_CONTEXT_CHARS）

    Returns:
        (contextStr, usedResults)
        - contextStr: 拼接后的上下文字符串
        - usedResults: 实际纳入上下文的检索结果列表
    """
    # 按 rank 排序，rank 缺失时排在最后
    sortedResults = sorted(retrievalResults, key=lambda r: r.get("rank", 9999))

    SEPARATOR = "\n\n---\n\n"
    separatorLen = len(SEPARATOR)

    contextParts: list[str] = []
    usedResults: list[dict[str, Any]] = []
    totalChars = 0

    for item in sortedResults:
        block = formatTermContext(item)
        # 非首条需要加分隔符
        neededChars = len(block) + (separatorLen if contextParts else 0)

        if totalChars + neededChars > maxChars:
            break

        contextParts.append(block)
        usedResults.append(item)
        totalChars += neededChars

    contextStr = SEPARATOR.join(contextParts)
    return contextStr, usedResults


def buildPrompt(
    query: str,
    retrievalResults: list[dict[str, Any]],
    maxContextChars: int = MAX_CONTEXT_CHARS,
    systemPrompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """
    构建完整的 system + user prompt（f-string 实现）

    Args:
        query: 用户查询文本
        retrievalResults: 检索结果列表（需包含 term、text、source、page 等字段）
        maxContextChars: 上下文最大字符数
        systemPrompt: 系统提示词（默认 SYSTEM_PROMPT）

    Returns:
        dict，包含：
            - "system" (str): 系统提示词
            - "user" (str): 用户消息内容
            - "used_results" (list): 实际纳入上下文的检索结果（便于调试和溯源）
    """
    contextStr, usedResults = buildContext(retrievalResults, maxContextChars)

    if contextStr:
        userContent = (
            f"请根据以下参考资料回答问题。\n\n"
            f"===参考资料===\n"
            f"{contextStr}\n"
            f"===参考资料结束===\n\n"
            f"问题：{query}"
        )
    else:
        # 无检索结果时退化为直接问答
        userContent = (
            f"问题：{query}\n（注：未检索到相关参考资料，请根据已知知识回答。）"
        )

    return {
        "system": systemPrompt,
        "user": userContent,
        "used_results": usedResults,
    }


def buildMessages(
    query: str,
    retrievalResults: list[dict[str, Any]],
    maxContextChars: int = MAX_CONTEXT_CHARS,
    systemPrompt: str = SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """
    构建 OpenAI/HuggingFace Chat 兼容的 messages 列表

    Args:
        query: 用户查询文本
        retrievalResults: 检索结果列表
        maxContextChars: 上下文最大字符数
        systemPrompt: 系统提示词

    Returns:
        [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
    """
    prompt = buildPrompt(query, retrievalResults, maxContextChars, systemPrompt)
    return [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]


def buildPromptJinja2(
    query: str,
    retrievalResults: list[dict[str, Any]],
    maxContextChars: int = MAX_CONTEXT_CHARS,
    extraInstructions: str = "",
) -> dict[str, Any]:
    """
    使用 Jinja2 模板构建 prompt（需安装 jinja2）

    与 buildPrompt 功能相同，但通过 Jinja2 渲染，便于后续扩展（few-shot、条件块等）。

    Args:
        query: 用户查询文本
        retrievalResults: 检索结果列表
        maxContextChars: 上下文最大字符数
        extraInstructions: 额外指令（追加到 system prompt 末尾）

    Returns:
        dict，包含 "system"、"user"、"used_results"

    Raises:
        ImportError: 未安装 jinja2 时抛出
    """
    try:
        from jinja2 import Template
    except ImportError as e:
        raise ImportError(
            "buildPromptJinja2 需要 jinja2，请执行 pip install jinja2"
        ) from e

    contextStr, usedResults = buildContext(retrievalResults, maxContextChars)

    # 渲染 system prompt
    systemTmpl = Template(JINJA2_SYSTEM_TEMPLATE)
    systemContent = systemTmpl.render(extra_instructions=extraInstructions).strip()

    # 根据是否有上下文选择 user 模板
    if contextStr:
        userTmpl = Template(JINJA2_USER_TEMPLATE_WITH_CONTEXT)
        userContent = userTmpl.render(context=contextStr, query=query)
    else:
        userTmpl = Template(JINJA2_USER_TEMPLATE_NO_CONTEXT)
        userContent = userTmpl.render(query=query)

    return {
        "system": systemContent,
        "user": userContent,
        "used_results": usedResults,
    }


# ---- 命令行演示 ----


def _demoWithCorpus(query: str, topk: int = 3) -> None:
    """
    从语料文件加载示例，演示提示模板效果

    Args:
        query: 演示查询
        topk: 加载前 topk 条语料作为模拟检索结果
    """
    import json

    corpusFile = os.path.join(config.PROCESSED_DIR, "retrieval", "corpus.jsonl")

    if not os.path.isfile(corpusFile):
        print(f"❌ 语料文件不存在: {corpusFile}")
        return

    # 加载前 topk 条作为模拟检索结果
    items: list[dict[str, Any]] = []
    with open(corpusFile, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= topk:
                break
            try:
                item = json.loads(line.strip())
                item["rank"] = i + 1
                items.append(item)
            except json.JSONDecodeError:
                continue

    if not items:
        print("❌ 语料文件为空")
        return

    print("=" * 60)
    print(f"演示查询：{query}")
    print(f"检索结果数：{len(items)} 条（来自语料前 {topk} 条）")
    print("=" * 60)

    prompt = buildPrompt(query=query, retrievalResults=items)

    print("\n【System Prompt】")
    print(prompt["system"])
    print(f"\n【User Prompt】（实际纳入 {len(prompt['used_results'])} 条）")
    print(prompt["user"])
    print("\n【Messages 格式】")
    messages = buildMessages(query=query, retrievalResults=items)
    for msg in messages:
        print(f"  role={msg['role']}, content 长度={len(msg['content'])} 字符")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 提示模板演示")
    parser.add_argument(
        "--query", type=str, default="什么是一致收敛？", help="演示查询"
    )
    parser.add_argument("--topk", type=int, default=3, help="加载语料前 topk 条")
    args = parser.parse_args()

    _demoWithCorpus(query=args.query, topk=args.topk)
