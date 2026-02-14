"""
使用 OpenAI 兼容接口在线生成数学名词数据（JSON）。

使用方法：
    python data_gen.py                    # 按顺序处理所有已提取术语的书
    python data_gen.py "书名"              # 只处理指定的书
    python data_gen.py "书名" 100    # 只处理第 100 页及之后首次出现的术语

输出：每本书的术语 JSON 保存到 processed/chunk/<book_name>/ 下，每个术语一个独立 JSON 文件。
配置项在 config.toml 的 [model] 部分。
"""

import json
import os
import re
import sys
import time
from pathlib import Path

# 规范模块搜索路径，保证能定位项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def _load_toml(path):
    """读取 TOML 配置。"""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def _default_prompts():
    """默认提示词模板（建议放在 config.toml 中显式配置）。"""
    return {
        "prompt_system": (
            "你是数学术语编纂助手。请为指定术语生成标准化 JSON。"
            "必须严格输出 JSON 对象，不要输出多余解释。"
        ),
        "prompt_user": (
            "术语：{{term}}\n"
            "学科：{{subject_label}}\n"
            "来源：{{sources}}\n"
            "上下文（来自 OCR）：\n"
            "{{context}}\n"
            "要求：\n"
            "1) 输出字段必须包含：id, term, aliases, sense_id, subject, definitions, notation, formula, "
            "usage, applications, disambiguation, related_terms, sources, search_keys, lang, confidence。\n"
            "2) definitions 至少包含 2 条（strict + alternative），必要时再补充 informal，"
            "且每条定义包含 reference 字段。\n"
            "3) 数学符号与公式必须使用 LaTeX。\n"
            "4) related_terms 至少 3 个。\n"
            "5) 仅输出 JSON 对象。\n"
            "6) 内容必须更具体，避免空泛表述。\n"
            "示例（格式参考，内容需针对当前术语生成）：\n"
            "{{example}}"
        ),
        "prompt_example": (
            "{\n"
            '  "id": "ma-柯西列",\n'
            '  "term": "柯西列",\n'
            '  "aliases": ["Cauchy 列"],\n'
            '  "sense_id": "1",\n'
            '  "subject": "数学分析",\n'
            '  "definitions": [\n'
            "    {\n"
            '      "type": "strict",\n'
            '      "text": "在度量空间 $(X,d)$ 中，序列 $\\{x_n\\}$ 称为柯西列，若对任意 $\\epsilon>0$，存在 $N$ 使得当 $m,n\\ge N$ 时，有 $d(x_m,x_n)<\\epsilon$。",\n'
            '      "conditions": "$(X,d)$ 为度量空间",\n'
            '      "notation": "$d(x_m,x_n)$",\n'
            '      "reference": "教材/讲义"\n'
            "    }\n"
            "  ],\n"
            '  "notation": "$d(x_m,x_n)$",\n'
            '  "formula": ["\\\\forall\\\\epsilon>0\\\\,\\\\exists N\\\\,\\\\forall m,n\\\\ge N:\\\\,d(x_m,x_n)<\\\\epsilon"],\n'
            '  "usage": "用于刻画序列在度量空间中的内在收敛性，不依赖极限点是否存在。",\n'
            '  "applications": "用于证明完备性与收敛性等价性质；在函数空间中用于一致收敛的判别。",\n'
            '  "disambiguation": "不同于一般收敛，柯西列只要求后项彼此接近。",\n'
            '  "related_terms": ["完备性", "收敛", "度量空间"],\n'
            '  "sources": ["教材/讲义"],\n'
            '  "search_keys": ["柯西列", "cauchy列", "cauchy"],\n'
            '  "lang": "zh",\n'
            '  "confidence": "medium"\n'
            "}\n"
        ),
        "prompt_repair_system": (
            "你是数学术语编纂助手。请修复并补全 JSON。"
            "必须严格输出 JSON 对象，不要输出多余解释。"
        ),
        "prompt_repair_user": (
            "术语：{{term}}\n"
            "学科：{{subject_label}}\n"
            "**上次生成失败原因**：{{reason}}\n\n"
            "请根据上述错误原因，修复以下 JSON 的问题：\n"
            "- 如果缺少必需字段，请补齐\n"
            "- 如果内容过短或不符合要求，请扩充\n"
            "- 至少包含 1 条 strict 定义和 1 条 alternative 定义\n"
            "- 补齐公式（formula 字段至少 1 个 LaTeX 公式）\n"
            "- 补齐 usage、applications（各至少 20 字符）\n"
            "- 补齐 related_terms（至少 3 个）\n"
            "- 给出 sources 和 definitions[].reference\n\n"
            "仅输出修复后的完整 JSON 对象，不要输出解释。\n\n"
            "当前 JSON（需要修复）：\n"
            "{{bad_json}}\n"
        ),
    }


def _load_config():
    """从 config.toml 读取配置。"""
    data = _load_toml(config.CONFIG_TOML)
    paths = data.get("paths", {})
    model_cfg = data.get("model", {})

    processed_dir = os.path.abspath(
        os.path.join(config.PROJECT_ROOT, paths.get("processed_dir", ""))
    )
    output_json = os.path.abspath(
        os.path.join(config.PROJECT_ROOT, paths.get("model_output_json", ""))
    )

    cfg = {
        "root_dir": config.PROJECT_ROOT,
        "processed_dir": processed_dir,
        "output_json": output_json,
        "api_base": model_cfg.get("api_base", "").rstrip("/"),
        "model": model_cfg.get("model", ""),
        "api_key_env": model_cfg.get("api_key_env", "API-KEY"),
        "max_tokens": model_cfg.get("max_tokens", 900),
        "temperature": model_cfg.get("temperature", 0.3),
        "top_p": model_cfg.get("top_p", 0.9),
        "max_retries": model_cfg.get("max_retries", 2),
        "max_attempts": model_cfg.get("max_attempts", 5),
        "min_definition_chars": model_cfg.get("min_definition_chars", 40),
        "min_usage_chars": model_cfg.get("min_usage_chars", 20),
        "min_applications_chars": model_cfg.get("min_applications_chars", 20),
        "min_formula_count": model_cfg.get("min_formula_count", 1),
        "endpoint": model_cfg.get("endpoint", "/chat/completions"),
        "request_timeout": model_cfg.get("request_timeout", 60),
        # 不配置 rpm/tpm 视为不限速
        "rpm": model_cfg.get("rpm", 0),
        "tpm": model_cfg.get("tpm", 0),
        "retry_wait_seconds": model_cfg.get("retry_wait_seconds", 2),
        "clear_context_each_request": model_cfg.get("clear_context_each_request", True),
        "stream": model_cfg.get("stream", False),
        "start_index": model_cfg.get("start_index", 1),
        "prompt_system": model_cfg.get("prompt_system", ""),
        "prompt_user": model_cfg.get("prompt_user", ""),
        "prompt_example": model_cfg.get("prompt_example", ""),
        "prompt_repair_system": model_cfg.get("prompt_repair_system", ""),
        "prompt_repair_user": model_cfg.get("prompt_repair_user", ""),
        "ocr_max_context_chars": model_cfg.get("ocr_max_context_chars", 800),
    }

    defaults = _default_prompts()
    for key, value in defaults.items():
        if not cfg.get(key):
            cfg[key] = value
    return cfg


def _load_env_value(root_dir, key):
    """从环境变量或 .env 读取密钥（支持含 '-' 的键名）。"""
    if key in os.environ:
        return os.environ.get(key)

    env_path = os.path.join(root_dir, ".env")
    if not os.path.isfile(env_path):
        return None

    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k != key:
                continue
            v = v.strip().strip('"').strip("'")
            return v
    return None


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_key(text):
    if not text:
        return ""
    return re.sub(r"\s+", "", str(text)).lower()


def _estimate_tokens(text):
    """粗略估算 token 数量（按 2 字符≈1 token）。"""
    if not text:
        return 0
    return max(1, len(text) // 2)


def _render_prompt(template, **kwargs):
    """用 {{key}} 形式替换模板占位符。"""
    if not template:
        return ""
    text = template
    for key, value in kwargs.items():
        text = text.replace(f"{{{{{key}}}}}", str(value))
    return text


def _infer_subject_label(book_name):
    """从书名目录名推断学科标签。

    例如：
        '数学分析(第5版)上(华东师范大学数学系)' → '数学分析'
        '概率论与数理统计教程第三版(茆诗松)'   → '概率论与数理统计'
        '高等代数(第五版)(王萼芳石生明)'       → '高等代数'
    """
    # 取第一个括号前的部分
    name = re.split(r"[（(]", book_name, maxsplit=1)[0]
    # 去掉版次信息（如 "第5版"、"第三版"）
    name = re.sub(r"第.+?版", "", name)
    # 去掉 "教程"、"上"、"下" 等后缀
    name = re.sub(r"(教程|上|下)\s*$", "", name).strip()
    return name or book_name


def _build_id(term):
    key = _normalize_key(term)
    if not key:
        return "ma-unknown"
    return f"ma-{key}"


def _extract_json_block(text):
    """从模型输出中提取 JSON 对象。"""
    if not text:
        return None

    code_block = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if code_block:
        return code_block.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _quality_check(record, cfg):
    """检查输出质量，决定是否需要重试。"""
    if not isinstance(record, dict):
        return False, "输出不是 JSON 对象"

    definitions = _ensure_list(record.get("definitions"))
    if not definitions:
        return False, "definitions 为空"

    strict_defs = [
        d for d in definitions if isinstance(d, dict) and d.get("type") == "strict"
    ]
    if not strict_defs:
        return False, "缺少 strict 定义"

    alt_defs = [
        d for d in definitions if isinstance(d, dict) and d.get("type") == "alternative"
    ]
    if not alt_defs:
        return False, "缺少 alternative 定义"

    strict_text = strict_defs[0].get("text", "")
    if len(strict_text) < cfg["min_definition_chars"]:
        return False, "strict 定义过短"

    formulas = _ensure_list(record.get("formula"))
    if (
        len([f for f in formulas if isinstance(f, str) and f.strip()])
        < cfg["min_formula_count"]
    ):
        return False, "公式数量不足"

    usage = record.get("usage") or ""
    applications = record.get("applications") or ""
    if len(usage) < cfg["min_usage_chars"]:
        return False, "usage 过短"
    if len(applications) < cfg["min_applications_chars"]:
        return False, "applications 过短"

    related_terms = _ensure_list(record.get("related_terms"))
    if len([t for t in related_terms if isinstance(t, str) and t.strip()]) < 3:
        return False, "related_terms 过少"

    sources = _ensure_list(record.get("sources"))
    if not sources:
        return False, "sources 为空"

    return True, ""


def _normalize_record(record, term, model_id, subject_label, sources):
    """补齐字段，确保符合 plan.md 规定的格式。"""
    record = record or {}

    record_id = record.get("id") or _build_id(term)
    aliases = _ensure_list(record.get("aliases"))
    definitions = _ensure_list(record.get("definitions"))

    if not definitions:
        definitions = [
            {
                "type": "informal",
                "text": f"{term} 是数学中的一个概念。",
                "conditions": "",
                "notation": "",
                "reference": sources[0] if sources else "",
            }
        ]

    fixed_definitions = []
    for item in definitions:
        if not isinstance(item, dict):
            continue
        fixed_definitions.append(
            {
                "type": item.get("type") or "informal",
                "text": item.get("text") or "",
                "conditions": item.get("conditions") or "",
                "notation": item.get("notation") or "",
                "reference": item.get("reference") or (sources[0] if sources else ""),
            }
        )

    keys = set()
    if term:
        keys.add(term)
        keys.add(_normalize_key(term))
    for a in aliases:
        if a:
            keys.add(a)
            keys.add(_normalize_key(a))
    search_keys = [k for k in keys if k]

    return {
        "id": record_id,
        "term": record.get("term") or term,
        "aliases": aliases,
        "sense_id": record.get("sense_id") or "1",
        "subject": record.get("subject") or subject_label,
        "definitions": fixed_definitions,
        "notation": record.get("notation") or "",
        "formula": _ensure_list(record.get("formula")),
        "usage": record.get("usage") or "",
        "applications": record.get("applications") or "",
        "disambiguation": record.get("disambiguation") or "",
        "related_terms": _ensure_list(record.get("related_terms")),
        "sources": sources,
        "search_keys": search_keys,
        "lang": record.get("lang") or "zh",
        "confidence": record.get("confidence") or "low",
    }


def _build_prompt(cfg, term, context, sources):
    """从配置构造提示词。"""
    example = cfg["prompt_example"]
    sources_text = "；".join(sources) if sources else ""
    system = _render_prompt(
        cfg["prompt_system"],
        term=term,
        subject_label=cfg["subject_label"],
        example=example,
        context=context,
        sources=sources_text,
    )
    user = _render_prompt(
        cfg["prompt_user"],
        term=term,
        subject_label=cfg["subject_label"],
        example=example,
        context=context,
        sources=sources_text,
    )
    return system, user


def _build_repair_prompt(cfg, term, bad_json, reason, context, sources):
    """从配置构造修复提示词。"""
    sources_text = "；".join(sources) if sources else ""
    system = _render_prompt(
        cfg["prompt_repair_system"],
        term=term,
        subject_label=cfg["subject_label"],
        reason=reason,
        bad_json=bad_json,
        context=context,
        sources=sources_text,
    )
    user = _render_prompt(
        cfg["prompt_repair_user"],
        term=term,
        subject_label=cfg["subject_label"],
        reason=reason,
        bad_json=bad_json,
        context=context,
        sources=sources_text,
    )
    return system, user


def _call_model(cfg, api_key, messages):
    """调用 OpenAI 兼容接口。"""
    import requests

    url = f"{cfg['api_base']}{cfg['endpoint']}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "max_tokens": cfg["max_tokens"],
        "stream": cfg["stream"],
    }

    # 发送请求，包含连接和读取超时
    timeout_val = cfg.get("request_timeout", 60)
    try:
        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=(10, timeout_val),  # (连接超时, 读取超时)
            stream=cfg["stream"],
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("请求超时：连接或读取超时")
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"连接错误：{e}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"请求异常：{e}")

    # 检查HTTP状态码
    if resp.status_code in (429, 503):
        raise RuntimeError(f"HTTP {resp.status_code}")
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP错误 {resp.status_code}：{e}")

    # 非流式模式：直接解析JSON
    if not cfg["stream"]:
        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"JSON解析失败：{e}，响应内容：{resp.text[:200]}")

        # 验证响应结构
        if not isinstance(data, dict):
            raise RuntimeError(f"响应格式错误：期望dict，得到{type(data)}")
        choices = data.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            raise RuntimeError(f"响应缺少choices字段，完整响应：{data}")
        message = choices[0].get("message")
        if not message or not isinstance(message, dict):
            raise RuntimeError(f"响应缺少message字段，完整响应：{data}")
        content = message.get("content")
        if content is None:
            raise RuntimeError(f"响应缺少content字段，完整响应：{data}")
        return content

    # 流式模式：逐行解析
    parts = []
    line_count = 0
    max_lines = 10000  # 防止无限循环

    try:
        for line in resp.iter_lines(decode_unicode=True):
            line_count += 1
            if line_count > max_lines:
                raise RuntimeError(
                    f"流式响应行数超过限制（{max_lines}行），可能存在异常"
                )

            if not line:
                continue

            # 提取data字段
            if line.startswith("data:"):
                data_str = line[len("data:") :].strip()
            else:
                data_str = line.strip()

            # 跳过空行和结束标记
            if not data_str or data_str == "[DONE]":
                continue

            # 解析JSON chunk
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                # 跳过无法解析的行
                continue

            # 提取内容
            if not isinstance(chunk, dict):
                continue
            choices = chunk.get("choices")
            if not choices or not isinstance(choices, list):
                continue
            if len(choices) == 0:
                continue
            delta = choices[0].get("delta")
            if not delta or not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if content:
                parts.append(content)
    except requests.exceptions.ChunkedEncodingError as e:
        # 流式传输中断，返回已接收的内容
        if not parts:
            raise RuntimeError(f"流式传输中断且无内容：{e}")
        # 如果已有部分内容，则返回
        pass
    except Exception as e:
        # 其他异常
        if not parts:
            raise RuntimeError(f"流式解析失败：{e}")
        # 如果已有部分内容，则返回
        pass

    result = "".join(parts)
    if not result:
        raise RuntimeError("流式响应为空")
    return result


def _collect_book_dirs():
    """扫描术语目录，返回有 map.json 的书名列表（按名称排序）。"""
    if not os.path.isdir(config.TERMS_DIR):
        return []
    books = []
    for name in sorted(os.listdir(config.TERMS_DIR)):
        terms_book_dir = os.path.join(config.TERMS_DIR, name)
        terms_map_path = os.path.join(terms_book_dir, "map.json")
        if os.path.isfile(terms_map_path):
            books.append(name)
    return books


def _load_term_pages_map_for_book(book_name):
    """读取指定书的术语-页码映射（JSON）。"""
    terms_map_path = os.path.join(config.TERMS_DIR, book_name, "map.json")
    if not os.path.isfile(terms_map_path):
        return {}
    try:
        with open(terms_map_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    term_pages = {}
    for k, v in data.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, list):
            pages = [p for p in v if isinstance(p, int)]
        else:
            pages = []
        if pages:
            term_pages[k] = pages
    return term_pages


def _load_terms_for_book(book_name):
    """读取指定书的术语列表（保持顺序）。"""
    terms_all_path = os.path.join(config.TERMS_DIR, book_name, "all.json")
    if not os.path.isfile(terms_all_path):
        # 回退：从 map.json 的 key 中获取
        term_pages = _load_term_pages_map_for_book(book_name)
        return sorted(term_pages.keys())
    try:
        with open(terms_all_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if isinstance(data, list):
        return [t for t in data if isinstance(t, str)]
    return []


def _load_pages_context_for_book(book_name, pages):
    """从指定书的分页 OCR 输出中读取上下文。"""
    pages_dir = os.path.join(config.OCR_DIR, book_name, "pages")
    if not os.path.isdir(pages_dir) or not pages:
        return []

    contexts = []
    for page in pages:
        page_file = os.path.join(pages_dir, f"page_{page:04d}.md")
        if not os.path.isfile(page_file):
            continue
        try:
            with open(page_file, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            continue
        # 去掉页码标记
        if content.startswith("<!-- page:"):
            content = content.split("\n", 2)[-1].lstrip()
        contexts.append(f"[第{page}页]\n{content.strip()}")
    return contexts


def _build_sources_for_book(book_name, pages):
    """构造来源列表（书名 + 页码）。"""
    sources = []
    if book_name and pages:
        for page in pages:
            sources.append(f"{book_name} 第{page}页")
    if sources:
        return sources
    if book_name:
        return [book_name]
    return ["未知来源"]


def _pick_context(contexts, max_chars):
    """合并上下文并限制长度。"""
    if not contexts:
        return ""
    merged = "\n\n".join(contexts)
    if len(merged) <= max_chars:
        return merged
    return merged[:max_chars] + "..."


def _safe_filename(term):
    """将术语转换为安全的文件名。"""
    safe = re.sub(r'[<>:"/\\|?*]', "_", term)
    return safe[:50]


def _filter_terms_by_start_page(terms, term_pages_map, start_page):
    """根据起始页码过滤术语，只保留首次出现在 start_page 及之后的术语。"""
    filtered = []
    for term in terms:
        pages = term_pages_map.get(term, [])
        if not pages:
            continue
        # 术语首次出现的页码
        first_page = min(pages)
        if first_page >= start_page:
            filtered.append(term)
    return filtered


def _generate_for_book(book_name, cfg, api_key, start_page=None):
    """
    为单本书的术语生成 JSON 数据。

    Args:
        book_name: 书名（OCR 目录名）
        cfg: 模型配置
        api_key: API 密钥
        start_page: 起始页码（可选），只处理首次出现在该页及之后的术语

    Returns:
        (成功数, 失败数)
    """
    # 按书名自动推断学科标签（如 "数学分析"、"高等代数"）
    cfg["subject_label"] = _infer_subject_label(book_name)

    terms = _load_terms_for_book(book_name)
    if not terms:
        print(f"  未找到术语: {book_name}")
        return 0, 0

    term_pages_map = _load_term_pages_map_for_book(book_name)

    # 按起始页码过滤术语
    if start_page is not None:
        terms = _filter_terms_by_start_page(terms, term_pages_map, start_page)
        if not terms:
            print(f"  第 {start_page} 页及之后无术语: {book_name}")
            return 0, 0
    else:
        # 使用 config.toml 的 start_index 做术语序号偏移
        try:
            start_index = int(cfg["start_index"])
        except (TypeError, ValueError):
            start_index = 1
        if start_index < 1:
            start_index = 1
        if start_index > len(terms):
            print(f"  start_index 超出术语数量，已无可处理项: {book_name}")
            return 0, 0
        terms = terms[start_index - 1 :]

    # 输出目录：processed/chunk/{书名}/
    chunk_dir = os.path.join(config.CHUNK_DIR, book_name)
    os.makedirs(chunk_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"生成 JSON: {book_name}")
    if start_page is not None:
        print(f"起始页码: {start_page}")
    print(f"术语数量: {len(terms)}")
    print(f"输出目录: {chunk_dir}")
    print(f"{'=' * 60}")

    success_terms = []
    skipped_terms = []  # 已存在且质量合格，直接跳过
    failed_terms = []
    failed_reasons = {}

    total = len(terms)
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    iterator = tqdm(terms, desc=f"生成({book_name[:15]})") if tqdm else terms
    last_request_at = 0.0
    current_term = None
    interrupted = False

    try:
        for idx, term in enumerate(iterator, start=1):
            current_term = term

            # 检查是否已生成（跳过已有且质量合格的 JSON）
            term_json_path = os.path.join(chunk_dir, f"{_safe_filename(term)}.json")
            if os.path.isfile(term_json_path):
                try:
                    with open(term_json_path, encoding="utf-8") as f:
                        existing_record = json.load(f)
                    # 对已有文件做质量检查，合格才跳过
                    quality_ok, quality_reason = _quality_check(existing_record, cfg)
                    if quality_ok:
                        skipped_terms.append(term)
                        print(f"  [跳过] {idx}/{total}: {term}（已有且质量合格）")
                        continue
                    # 质量不合格，重新生成
                    print(
                        f"  [重生成] {idx}/{total}: {term}（已有文件质量不合格: {quality_reason}）"
                    )
                except Exception as e:
                    # 文件损坏，重新生成
                    print(f"  [重生成] {idx}/{total}: {term}（已有文件损坏: {e}）")
            else:
                print(f"  [生成] {idx}/{total}: {term}")

            record = None
            last_json = ""
            last_reason = ""
            ok = False

            pages = term_pages_map.get(term, [])
            sources = _build_sources_for_book(book_name, pages)

            page_contexts = _load_pages_context_for_book(book_name, pages)
            context = _pick_context(page_contexts, cfg["ocr_max_context_chars"])
            if not context:
                context = "未找到明确上下文，请基于通用知识补全。"

            attempt = 0
            max_attempts = cfg["max_attempts"]
            while attempt < max_attempts:
                attempt += 1

                if attempt == 1 or not last_json:
                    system, user = _build_prompt(cfg, term, context, sources)
                else:
                    system, user = _build_repair_prompt(
                        cfg, term, last_json, last_reason, context, sources
                    )

                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]

                # 限速控制
                now = time.time()
                min_interval = 0.0
                if cfg["rpm"] and cfg["rpm"] > 0:
                    min_interval = 60.0 / cfg["rpm"]
                prompt_tokens = _estimate_tokens(system + user)
                token_budget = prompt_tokens + cfg["max_tokens"]
                if cfg["tpm"] and cfg["tpm"] > 0:
                    min_interval_tpm = 60.0 * token_budget / cfg["tpm"]
                else:
                    min_interval_tpm = 0.0
                min_interval = max(min_interval, min_interval_tpm)
                sleep_for = min_interval - (now - last_request_at)
                if sleep_for > 0:
                    time.sleep(sleep_for)

                # 调用模型
                try:
                    gen_text = _call_model(cfg, api_key, messages)
                    last_request_at = time.time()
                except Exception as e:
                    err = str(e)
                    last_reason = f"请求失败：{err}"
                    print(
                        f"    [请求失败] {term}（第 {attempt}/{max_attempts} 次）: {err}"
                    )
                    if "HTTP 429" in err or "HTTP 503" in err:
                        time.sleep(max(cfg["retry_wait_seconds"], 10))
                    else:
                        time.sleep(cfg["retry_wait_seconds"])
                    continue

                # 提取 JSON
                json_block = _extract_json_block(gen_text)
                if not json_block:
                    last_json = gen_text
                    last_reason = "未找到 JSON 对象"
                    print(
                        f"    [重试] {term}（第 {attempt}/{max_attempts} 次）: 未找到 JSON 对象"
                    )
                    time.sleep(cfg["retry_wait_seconds"])
                    continue

                last_json = json_block
                try:
                    record = json.loads(json_block)
                except json.JSONDecodeError:
                    record = None
                    last_reason = "JSON 解析失败"
                    print(
                        f"    [重试] {term}（第 {attempt}/{max_attempts} 次）: JSON 解析失败"
                    )
                    time.sleep(cfg["retry_wait_seconds"])
                    continue

                # 质量检查
                ok, reason = _quality_check(record, cfg)
                if ok:
                    print(f"    [质量通过] {term}（第 {attempt} 次尝试）")
                    break
                last_reason = reason
                print(f"    [重试] {term}（第 {attempt}/{max_attempts} 次）: {reason}")
                time.sleep(cfg["retry_wait_seconds"])

            if not ok:
                print(
                    f"  [失败] {term}: 达到最大重试 {max_attempts} 次（{last_reason}）"
                )

            # 质量合格才保存，不合格不写文件（下次运行会重新生成）
            if ok:
                item = _normalize_record(
                    record, term, cfg["model"], cfg["subject_label"], sources
                )
                with open(term_json_path, "w", encoding="utf-8") as f:
                    json.dump(item, f, ensure_ascii=False, indent=2)
                success_terms.append(term)
                print(f"  [成功] {idx}/{total}: {term} -> {term_json_path}")
            else:
                # 质量不合格，不保存文件；如果之前有残留的不合格文件，也删掉
                if os.path.isfile(term_json_path):
                    os.remove(term_json_path)
                failed_terms.append(term)
                failed_reasons[term] = last_reason or "质量未达标"
                print(
                    f"  [未保存] {idx}/{total}: {term}（质量不合格: {last_reason}，下次运行将重试）"
                )
    except KeyboardInterrupt:
        interrupted = True
        print(f"已中断，当前词条: {current_term}")
    except Exception as e:
        interrupted = True
        print(f"异常中断: {e}，当前词条: {current_term}")

    print(f"  生成完成: {chunk_dir}")
    print(
        f"  跳过（已有且合格）: {len(skipped_terms)}, 新生成成功: {len(success_terms)}, 失败: {len(failed_terms)}"
    )
    if failed_terms:
        print("  失败词条:")
        for t in failed_terms:
            print(f"    - {t}（原因: {failed_reasons.get(t, '未知')}）")
    if interrupted:
        print(f"  中断位置: {current_term}")

    return len(skipped_terms) + len(success_terms), len(failed_terms)


def main():
    cfg = _load_config()
    if not cfg["api_base"]:
        print("config.toml 中未配置 model.api_base。")
        return
    if not cfg["model"]:
        print("config.toml 中未配置 model.model。")
        return

    api_key = _load_env_value(cfg["root_dir"], cfg["api_key_env"])
    if not api_key:
        print("未找到 API Key，请检查 .env 或环境变量。")
        return

    # 确定要处理的书目列表和起始页码
    start_page = None
    if len(sys.argv) > 1:
        # 指定了书名参数
        book_name = sys.argv[1]
        if book_name.lower().endswith(".pdf"):
            book_name = book_name[:-4]

        # 检查是否提供了起始页码参数
        if len(sys.argv) > 2:
            try:
                start_page = int(sys.argv[2])
            except ValueError:
                print(f"起始页码参数无效: {sys.argv[2]}，必须是整数")
                return

        terms_book_dir = os.path.join(config.TERMS_DIR, book_name)
        if not os.path.isdir(terms_book_dir):
            print(f"未找到术语目录: {terms_book_dir}")
            return
        book_list = [book_name]
    else:
        # 未指定，按顺序处理所有有术语的书
        book_list = _collect_book_dirs()
        if not book_list:
            print(f"术语目录下未找到已提取术语的书: {config.TERMS_DIR}")
            return
        print(f"找到 {len(book_list)} 本有术语的书:")
        for b in book_list:
            print(f"  - {b}")

    # 按顺序处理每本书
    total_success = 0
    total_failed = 0

    for book_name in book_list:
        success, failed = _generate_for_book(
            book_name, cfg, api_key, start_page=start_page
        )
        total_success += success
        total_failed += failed

    print(f"\n{'=' * 60}")
    print(f"全部完成: 处理 {len(book_list)} 本书")
    print(f"总成功（含跳过）: {total_success}, 总失败: {total_failed}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
