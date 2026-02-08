"""
使用 OpenAI 兼容接口在线生成数学名词数据（JSON）。

使用方法：
    python data_gen.py                    # 按顺序处理所有已提取术语的书
    python data_gen.py "书名"              # 只处理指定的书

输出：每本书的 JSON 保存到 ocr/<book_name>/terms_json/ 下，
      并合并到 ocr/<book_name>/terms_all.json。
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
            "问题：{{reason}}\n"
            "请在保持结构完整的前提下补全内容，至少包含 1 条 strict 定义，并补齐公式、usage、applications、related_terms，"
            "且给出 sources 和 definitions.reference。\n"
            "仅输出修复后的 JSON 对象。\n"
            "当前 JSON：\n"
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
        "subject_label": model_cfg.get("subject_label", ""),
        "seed_terms": model_cfg.get("seed_terms", []),
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
        "ocr_context_path": model_cfg.get("ocr_context_path", ""),
        "ocr_pages_dir": model_cfg.get("ocr_pages_dir", ""),
        "ocr_terms_with_pages_path": model_cfg.get("ocr_terms_with_pages_path", ""),
        "ocr_book_title": model_cfg.get("ocr_book_title", ""),
        "ocr_source_label": model_cfg.get("ocr_source_label", ""),
        "ocr_keywords": model_cfg.get("ocr_keywords", []),
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
    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=cfg["request_timeout"],
        stream=cfg["stream"],
    )
    if resp.status_code in (429, 503):
        raise RuntimeError(f"HTTP {resp.status_code}")
    resp.raise_for_status()

    if not cfg["stream"]:
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    parts = []
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data:"):
            data_str = line[len("data:") :].strip()
        else:
            data_str = line.strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            parts.append(content)
    return "".join(parts)


def _collect_book_dirs():
    """扫描 OCR 输出目录，返回有 terms_map.json 的书名列表（按名称排序）。"""
    if not os.path.isdir(config.OCR_DIR):
        return []
    books = []
    for name in sorted(os.listdir(config.OCR_DIR)):
        book_dir = os.path.join(config.OCR_DIR, name)
        terms_map_path = os.path.join(book_dir, "terms_map.json")
        if os.path.isfile(terms_map_path):
            books.append(name)
    return books


def _load_term_pages_map_for_book(book_name):
    """读取指定书的术语-页码映射（JSON）。"""
    terms_map_path = os.path.join(config.OCR_DIR, book_name, "terms_map.json")
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
    terms_all_path = os.path.join(config.OCR_DIR, book_name, "terms_all.json")
    if not os.path.isfile(terms_all_path):
        # 回退：从 terms_map.json 的 key 中获取
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


def _generate_for_book(book_name, cfg, api_key):
    """
    为单本书的术语生成 JSON 数据。

    Args:
        book_name: 书名（OCR 目录名）
        cfg: 模型配置
        api_key: API 密钥

    Returns:
        (成功数, 失败数)
    """
    terms = _load_terms_for_book(book_name)
    if not terms:
        print(f"  未找到术语: {book_name}")
        return 0, 0

    term_pages_map = _load_term_pages_map_for_book(book_name)

    # 应用 start_index（仅当使用 config.toml 的 start_index）
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

    book_dir = os.path.join(config.OCR_DIR, book_name)
    json_dir = os.path.join(book_dir, "terms_json")
    os.makedirs(json_dir, exist_ok=True)

    # 合并输出文件
    output_json = os.path.join(book_dir, "terms_all.json")

    print(f"\n{'=' * 60}")
    print(f"生成 JSON: {book_name}")
    print(f"术语数量: {len(terms)}")
    print(f"输出目录: {json_dir}")
    print(f"{'=' * 60}")

    success_terms = []
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

    # 收集所有生成结果
    all_items = []

    try:
        for idx, term in enumerate(iterator, start=1):
            current_term = term

            # 检查是否已生成（跳过已有的单个 JSON）
            term_json_path = os.path.join(json_dir, f"{_safe_filename(term)}.json")
            if os.path.isfile(term_json_path):
                try:
                    with open(term_json_path, encoding="utf-8") as f:
                        existing = json.load(f)
                    all_items.append(existing)
                    success_terms.append(term)
                    continue
                except Exception:
                    pass  # 文件损坏，重新生成

            if not tqdm:
                print(f"  正在生成 {idx}/{total}: {term}")

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
            while True:
                if attempt == 0:
                    system, user = _build_prompt(cfg, term, context, sources)
                else:
                    system, user = _build_repair_prompt(
                        cfg, term, last_json, last_reason, context, sources
                    )

                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]

                try:
                    attempt += 1
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

                    gen_text = _call_model(cfg, api_key, messages)
                    last_request_at = time.time()
                except Exception as e:
                    err = str(e)
                    last_reason = f"请求失败：{err}"
                    if not tqdm:
                        print(f"  请求失败，准备重试: {term}（第 {attempt} 次）")
                    if "HTTP 429" in err or "HTTP 503" in err:
                        time.sleep(max(cfg["retry_wait_seconds"], 10))
                    else:
                        time.sleep(cfg["retry_wait_seconds"])
                    continue

                json_block = _extract_json_block(gen_text)
                if json_block:
                    last_json = json_block
                    try:
                        record = json.loads(json_block)
                    except json.JSONDecodeError:
                        record = None
                        last_reason = "JSON 解析失败"
                        if not tqdm:
                            print(
                                f"  JSON 解析失败，准备重试: {term}（第 {attempt} 次）"
                            )
                        time.sleep(cfg["retry_wait_seconds"])
                        continue
                else:
                    record = None
                    last_json = gen_text
                    last_reason = "未找到 JSON 对象"
                    if not tqdm:
                        print(
                            f"  未找到 JSON 对象，准备重试: {term}（第 {attempt} 次）"
                        )
                    time.sleep(cfg["retry_wait_seconds"])
                    continue

                ok, reason = _quality_check(record, cfg)
                if ok:
                    break
                last_reason = reason
                if not tqdm:
                    print(
                        f"  质量未达标，准备重试: {term}（第 {attempt} 次，原因: {reason}）"
                    )
                time.sleep(cfg["retry_wait_seconds"])
                if attempt >= max_attempts:
                    break
            if attempt >= max_attempts and not ok:
                if not tqdm:
                    print(f"  已达到最大重试次数: {term}（{max_attempts} 次）")

            item = _normalize_record(
                record, term, cfg["model"], cfg["subject_label"], sources
            )
            all_items.append(item)

            # 保存单个术语 JSON
            with open(term_json_path, "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False, indent=2)

            if ok:
                success_terms.append(term)
            else:
                failed_terms.append(term)
                failed_reasons[term] = last_reason or "质量未达标"

            if not tqdm:
                print(f"  完成 {idx}/{total}: {term}")
    except KeyboardInterrupt:
        interrupted = True
        print(f"已中断，当前词条: {current_term}")
    except Exception as e:
        interrupted = True
        print(f"异常中断: {e}，当前词条: {current_term}")

    # 合并所有 JSON 到 terms_all.json
    if all_items:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_items, f, ensure_ascii=False, indent=2)

    print(f"  生成完成: {output_json}")
    print(f"  成功: {len(success_terms)}, 失败: {len(failed_terms)}")
    if failed_terms:
        print("  失败词条:")
        for t in failed_terms:
            print(f"    - {t}（原因: {failed_reasons.get(t, '未知')}）")
    if interrupted:
        print(f"  中断位置: {current_term}")

    return len(success_terms), len(failed_terms)


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

    # 确定要处理的书目列表
    if len(sys.argv) > 1:
        # 指定了书名参数
        book_name = sys.argv[1]
        if book_name.lower().endswith(".pdf"):
            book_name = book_name[:-4]
        book_dir = os.path.join(config.OCR_DIR, book_name)
        if not os.path.isdir(book_dir):
            print(f"未找到 OCR 输出目录: {book_dir}")
            return
        book_list = [book_name]
    else:
        # 未指定，按顺序处理所有有术语的书
        book_list = _collect_book_dirs()
        if not book_list:
            print(f"OCR 目录下未找到已提取术语的书: {config.OCR_DIR}")
            return
        print(f"找到 {len(book_list)} 本有术语的书:")
        for b in book_list:
            print(f"  - {b}")

    # 按顺序处理每本书
    total_success = 0
    total_failed = 0

    for book_name in book_list:
        success, failed = _generate_for_book(book_name, cfg, api_key)
        total_success += success
        total_failed += failed

    print(f"\n{'=' * 60}")
    print(f"全部完成: 处理 {len(book_list)} 本书")
    print(f"总成功: {total_success}, 总失败: {total_failed}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
