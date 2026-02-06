"""
JSON 生成模块 - 为术语生成结构化 JSON 数据
"""

import json
import sys
from pathlib import Path

# 规范模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common import call_model, extract_json_from_response

# ============================================================
# JSON 生成提示词
# ============================================================

JSON_GENERATION_PROMPT = """你是一个数学术语定义专家。请为以下数学术语生成结构化的 JSON 定义。

术语：{term}
学科：{subject}

上下文参考（来自教材 OCR）：
{context}

请生成以下格式的 JSON：
```json
{{
  "sense_id": "{term}_1",
  "term": "{term}",
  "subject": "{subject}",
  "definitions": [
    {{
      "type": "strict",
      "text": "严格的数学定义，使用规范的数学语言",
      "conditions": ["适用条件1", "适用条件2"],
      "notation": "相关数学符号表示"
    }},
    {{
      "type": "alternative",
      "text": "通俗易懂的解释，帮助理解概念"
    }}
  ],
  "formulas": ["相关公式1", "相关公式2"],
  "usage": "该术语在数学中的典型用法和应用场景",
  "applications": "实际应用举例",
  "sources": {sources}
}}
```

要求：
1. strict 定义必须准确、规范，使用数学语言
2. alternative 定义要通俗易懂，便于理解
3. 公式使用 LaTeX 格式
4. 如果上下文中有相关内容，优先参考上下文
5. 只返回 JSON，不要其他文字"""

# ============================================================
# 质量检查
# ============================================================


def quality_check(
    data: dict,
    min_strict_defs: int = 1,
    min_alt_defs: int = 1,
    min_formulas: int = 0,
    min_def_length: int = 20,
) -> tuple[bool, str]:
    """
    检查生成的 JSON 数据质量。

    Args:
        data: JSON 数据
        min_strict_defs: 最少 strict 定义数
        min_alt_defs: 最少 alternative 定义数
        min_formulas: 最少公式数
        min_def_length: 定义最小长度

    Returns:
        (是否通过, 失败原因)
    """
    # 检查必要字段
    required_fields = ["sense_id", "term", "subject", "definitions"]
    for field in required_fields:
        if field not in data:
            return False, f"缺少必要字段: {field}"

    # 检查 definitions
    definitions = data.get("definitions", [])
    if not isinstance(definitions, list) or len(definitions) == 0:
        return False, "definitions 为空或格式错误"

    # 统计定义类型
    strict_count = 0
    alt_count = 0
    for d in definitions:
        if not isinstance(d, dict):
            continue
        def_type = d.get("type", "")
        def_text = d.get("text", "")

        if def_type == "strict":
            strict_count += 1
            if len(def_text) < min_def_length:
                return False, f"strict 定义过短: {len(def_text)} < {min_def_length}"
        elif def_type == "alternative":
            alt_count += 1

    if strict_count < min_strict_defs:
        return False, f"strict 定义不足: {strict_count} < {min_strict_defs}"

    if alt_count < min_alt_defs:
        return False, f"alternative 定义不足: {alt_count} < {min_alt_defs}"

    # 检查公式数量
    formulas = data.get("formulas", [])
    if len(formulas) < min_formulas:
        return False, f"公式不足: {len(formulas)} < {min_formulas}"

    return True, ""


# ============================================================
# JSON 生成函数
# ============================================================


def generate_json_for_term(
    term: str,
    client,
    model: str,
    subject: str = "数学分析",
    context: str = "",
    sources: list[dict] | None = None,
    max_tokens: int = 2048,
    max_attempts: int = 3,
    quality_config: dict | None = None,
) -> dict | None:
    """
    为单个术语生成 JSON 数据。

    Args:
        term: 术语
        client: OpenAI 兼容客户端
        model: 模型名称
        subject: 学科
        context: 上下文参考
        sources: 来源列表
        max_tokens: 最大 token 数
        max_attempts: 最大重试次数
        quality_config: 质量检查配置

    Returns:
        生成的 JSON 数据，失败返回 None
    """
    if quality_config is None:
        quality_config = {}

    # 格式化来源
    sources_str = json.dumps(sources or [], ensure_ascii=False)

    # 构建提示词
    prompt = JSON_GENERATION_PROMPT.format(
        term=term,
        subject=subject,
        context=context[:2000] if context else "无上下文",
        sources=sources_str,
    )

    for attempt in range(max_attempts):
        try:
            # 调用模型
            response = call_model(
                client=client,
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=0.3,
            )

            # 解析 JSON
            data = extract_json_from_response(response)
            if data is None:
                print(f"  尝试 {attempt + 1}: JSON 解析失败")
                continue

            # 质量检查
            passed, reason = quality_check(data, **quality_config)
            if not passed:
                print(f"  尝试 {attempt + 1}: 质量检查失败 - {reason}")
                # 尝试修复
                prompt = _build_repair_prompt(term, subject, response, reason)
                continue

            return data

        except Exception as e:
            print(f"  尝试 {attempt + 1}: 生成失败 - {e}")
            continue

    return None


def _build_repair_prompt(term: str, subject: str, original: str, reason: str) -> str:
    """构建修复提示词"""
    return f"""之前为术语 "{term}" 生成的 JSON 存在问题：{reason}

原始输出：
{original[:1500]}

请修复上述问题，重新生成符合要求的 JSON。
学科：{subject}

要求：
1. 必须包含至少 1 个 strict 类型定义
2. 必须包含至少 1 个 alternative 类型定义
3. strict 定义长度至少 20 字符
4. 只返回 JSON，不要其他文字"""


# ============================================================
# 批量生成
# ============================================================


def generate_json_batch(
    terms: list[str],
    client,
    model: str,
    subject: str = "数学分析",
    context_provider=None,
    sources_provider=None,
    output_path: str | Path | None = None,
    max_attempts: int = 3,
    quality_config: dict | None = None,
    on_progress=None,
) -> list[dict]:
    """
    批量为术语生成 JSON 数据。

    Args:
        terms: 术语列表
        client: OpenAI 兼容客户端
        model: 模型名称
        subject: 学科
        context_provider: 上下文提供函数 (term) -> str
        sources_provider: 来源提供函数 (term) -> list[dict]
        output_path: 输出文件路径（增量保存）
        max_attempts: 最大重试次数
        quality_config: 质量检查配置
        on_progress: 进度回调函数 (current, total, term, success)

    Returns:
        生成的 JSON 数据列表
    """
    results = []

    # 加载已有结果（支持断点续传）
    existing_terms = set()
    if output_path:
        output_path = Path(output_path)
        if output_path.exists():
            try:
                with open(output_path, encoding="utf-8") as f:
                    existing_data = json.load(f)
                    results = existing_data
                    existing_terms = {d.get("term") for d in existing_data}
            except (json.JSONDecodeError, KeyError):
                pass

    total = len(terms)
    for i, term in enumerate(terms):
        # 跳过已处理的术语
        if term in existing_terms:
            if on_progress:
                on_progress(i + 1, total, term, True)
            continue

        print(f"[{i + 1}/{total}] 处理术语: {term}")

        # 获取上下文和来源
        context = context_provider(term) if context_provider else ""
        sources = sources_provider(term) if sources_provider else None

        # 生成 JSON
        data = generate_json_for_term(
            term=term,
            client=client,
            model=model,
            subject=subject,
            context=context,
            sources=sources,
            max_attempts=max_attempts,
            quality_config=quality_config,
        )

        success = data is not None
        if success:
            results.append(data)
            # 增量保存
            if output_path:
                _save_results(results, output_path)

        if on_progress:
            on_progress(i + 1, total, term, success)

    return results


def _save_results(results: list[dict], path: Path):
    """保存结果到文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
