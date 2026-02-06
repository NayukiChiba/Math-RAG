"""
全流程：遍历 raw 下所有 PDF -> OCR 分页 -> AI 抽取术语 -> AI 生成 JSON。
说明：不使用命令行参数，直接运行即可。
配置项在 config.toml 的 [ocr] 部分。
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

# 从配置文件加载 OCR 配置
_ocr_cfg = config.get_ocr_config()
PAGE_START = _ocr_cfg["page_start"]
PAGE_END = _ocr_cfg["page_end"]
SKIP_EXISTING_OCR = _ocr_cfg["skip_existing"]
OCR_MAX_IMAGE_SIZE = _ocr_cfg["max_image_size"]
MAX_PAGE_CHARS = _ocr_cfg["max_page_chars"]
MAX_TERM_LEN = _ocr_cfg["max_term_len"]
TERM_MAX_TOKENS = _ocr_cfg["term_max_tokens"]
MAX_PAGES_PER_TERM = _ocr_cfg["max_pages_per_term"]

# 常见噪声词（非术语）
STOPWORDS = {
    "我们",
    "它们",
    "这是",
    "因为",
    "因此",
    "于是",
    "可以",
    "可能",
    "若",
    "则",
    "设",
    "可",
    "应",
    "由此",
    "这里",
    "中学",
    "课程",
    "例",
    "证明",
    "注意",
    "这个",
    "那个",
    "这些",
    "那些",
    "它",
    "其",
}

# 数学术语关键词，用于过滤噪声
TERM_HINTS = [
    "函数",
    "序列",
    "数列",
    "列",
    "级数",
    "极限",
    "连续",
    "可导",
    "可微",
    "可积",
    "收敛",
    "一致",
    "有界",
    "无界",
    "上界",
    "下界",
    "开集",
    "闭集",
    "区间",
    "空间",
    "范数",
    "距离",
    "测度",
    "积分",
    "定理",
    "引理",
    "命题",
    "推论",
    "矩阵",
    "行列式",
    "向量",
    "线性",
    "多项式",
    "方程",
    "概率",
    "随机",
    "分布",
    "期望",
    "方差",
    "群",
    "环",
    "域",
]

# 泛称与编号型定理过滤
GENERIC_MATH_TITLES = {"定理", "引理", "命题", "推论"}
NUMBERED_TITLE_PATTERN = re.compile(r"^(定理|引理|命题|推论)\\s*\\d+(?:\\.\\d+)*$")


def _load_toml(path):
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def _default_prompts():
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
            '      "text": "在度量空间 $(X,d)$ 中，序列 $\\\\{x_n\\\\}$ 称为柯西列，若对任意 $\\\\epsilon>0$，存在 $N$ 使得当 $m,n\\\\ge N$ 时，有 $d(x_m,x_n)<\\\\epsilon$。",\n'
            '      "conditions": "$(X,d)$ 为度量空间",\n'
            '      "notation": "$d(x_m,x_n)$",\n'
            '      "reference": "教材/讲义 第1页"\n'
            "    }\n"
            "  ],\n"
            '  "notation": "$d(x_m,x_n)$",\n'
            '  "formula": ["\\\\\\\\forall\\\\\\\\epsilon>0\\\\\\\\,\\\\\\\\exists N\\\\\\\\,\\\\\\\\forall m,n\\\\\\\\ge N:\\\\\\\\,d(x_m,x_n)<\\\\\\\\epsilon"],\n'
            '  "usage": "用于刻画序列在度量空间中的内在收敛性，不依赖极限点是否存在。",\n'
            '  "applications": "用于证明完备性与收敛性等价性质；在函数空间中用于一致收敛的判别。",\n'
            '  "disambiguation": "不同于一般收敛，柯西列只要求后项彼此接近。",\n'
            '  "related_terms": ["完备性", "收敛", "度量空间"],\n'
            '  "sources": ["教材/讲义 第1页"],\n'
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
    data = _load_toml(config.CONFIG_TOML) if os.path.isfile(config.CONFIG_TOML) else {}
    model_cfg = data.get("model", {})

    cfg = {
        "api_base": model_cfg.get("api_base", "").rstrip("/"),
        "model": model_cfg.get("model", ""),
        "api_key_env": model_cfg.get("api_key_env", "API-KEY"),
        "subject_label": model_cfg.get("subject_label", "数学分析"),
        "max_tokens": model_cfg.get("max_tokens", 900),
        "temperature": model_cfg.get("temperature", 0.3),
        "top_p": model_cfg.get("top_p", 0.9),
        "max_attempts": model_cfg.get("max_attempts", 5),
        "min_definition_chars": model_cfg.get("min_definition_chars", 40),
        "min_usage_chars": model_cfg.get("min_usage_chars", 20),
        "min_applications_chars": model_cfg.get("min_applications_chars", 20),
        "min_formula_count": model_cfg.get("min_formula_count", 1),
        "endpoint": model_cfg.get("endpoint", "/chat/completions"),
        "request_timeout": model_cfg.get("request_timeout", 60),
        "rpm": model_cfg.get("rpm", 0),
        "tpm": model_cfg.get("tpm", 0),
        "retry_wait_seconds": model_cfg.get("retry_wait_seconds", 2),
        "stream": model_cfg.get("stream", False),
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


def _estimate_tokens(text):
    if not text:
        return 0
    return max(1, len(text) // 2)


class _RateLimiter:
    def __init__(self):
        self.last_request_at = 0.0

    def wait(self, cfg, token_budget):
        min_interval = 0.0
        if cfg["rpm"] and cfg["rpm"] > 0:
            min_interval = 60.0 / cfg["rpm"]

        if cfg["tpm"] and cfg["tpm"] > 0:
            min_interval_tpm = 60.0 * token_budget / cfg["tpm"]
        else:
            min_interval_tpm = 0.0

        min_interval = max(min_interval, min_interval_tpm)
        sleep_for = min_interval - (time.time() - self.last_request_at)
        if sleep_for > 0:
            time.sleep(sleep_for)

    def mark(self):
        self.last_request_at = time.time()


def _call_model(cfg, api_key, messages, max_tokens, stream):
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
        "max_tokens": max_tokens,
        "stream": stream,
    }
    resp = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=cfg["request_timeout"],
        stream=stream,
    )
    if resp.status_code in (429, 503):
        raise RuntimeError(f"HTTP {resp.status_code}")
    resp.raise_for_status()

    if not stream:
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


def _get_pdf_page_count(pdf_path):
    try:
        from pypdf import PdfReader
    except ImportError:
        return None

    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception:
        return None


def _iter_pages(total_pages):
    if PAGE_START is None:
        start = 0
    else:
        start = max(0, PAGE_START)

    if PAGE_END is None:
        if total_pages is None:
            return []
        end = total_pages - 1
    else:
        end = PAGE_END

    if total_pages is not None:
        end = min(end, total_pages - 1)

    if end < start:
        return []
    return list(range(start, end + 1))


def _ensure_ocr(pdf_path, pages_dir):
    """生成器函数：逐页 OCR，每完成一页就 yield (page_no, page_file)"""
    if not os.path.isfile(pdf_path):
        print(f"未找到 PDF：{pdf_path}")
        return

    os.makedirs(pages_dir, exist_ok=True)

    try:
        from pix2text import Pix2Text
    except ImportError:
        print("未检测到 pix2text，请先安装：pip install pix2text")
        return

    total_pages = _get_pdf_page_count(pdf_path)
    if total_pages is None and PAGE_END is None:
        print("未检测到 PDF 页数，请安装 pypdf 或手动设置 PAGE_END。")
        return

    pages = _iter_pages(total_pages)
    if not pages:
        print("未找到可处理的页码范围，请检查 PAGE_START/PAGE_END。")
        return

    p2t = Pix2Text.from_config()

    for page in pages:
        page_no = page + 1
        page_file = os.path.join(pages_dir, f"page_{page_no:04d}.md")
        if SKIP_EXISTING_OCR and os.path.isfile(page_file):
            yield page_no, page_file, False  # False 表示跳过（已存在）
            continue

        print(f"OCR 页码：{page_no}")
        doc = p2t.recognize_pdf(
            pdf_path,
            page_numbers=[page],
            table_as_image=True,
            resized_shape=OCR_MAX_IMAGE_SIZE,  # 从配置文件读取，限制最大尺寸
        )

        temp_output = os.path.join(os.path.dirname(pages_dir), "output.md")
        doc.to_markdown(os.path.dirname(pages_dir))
        if not os.path.isfile(temp_output):
            print(f"未生成 output.md：{temp_output}")
            continue

        with open(temp_output, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        with open(page_file, "w", encoding="utf-8") as f:
            f.write(f"<!-- page: {page_no} -->\n\n")
            f.write(content)

        del doc
        try:
            import gc

            gc.collect()
        except Exception:
            pass

        yield page_no, page_file, True  # True 表示新处理的页面


def _read_text(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def _split_into_chunks(text, max_chars=MAX_PAGE_CHARS):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    buf = []
    size = 0
    for p in paragraphs:
        if size + len(p) + 2 > max_chars and buf:
            chunks.append("\n\n".join(buf))
            buf = []
            size = 0
        buf.append(p)
        size += len(p) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def _build_term_prompt(chunk):
    system = (
        "你是数学术语抽取助手。"
        "请从给定文本中抽取数学相关术语（概念/定义/定理/引理/命题/推论名称）。"
        "仅输出 JSON 数组，元素为术语字符串，不要输出任何额外说明。"
    )
    user = (
        "文本如下（OCR）：\n"
        f"{chunk}\n\n"
        "要求：\n"
        "1) 只输出 JSON 数组。\n"
        "2) 术语必须与数学相关，非数学内容不要输出。\n"
        "3) 术语尽量为名词短语，避免代词/泛指词。\n"
        "4) 不要输出句子。\n"
        "5) 不要重复。\n"
    )
    return system, user


def _extract_json_array(text):
    if not text:
        return None
    code_block = re.search(r"```json\s*(\[.*?\])\s*```", text, re.S)
    if code_block:
        return code_block.group(1)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _clean_term(term):
    term = term.strip()
    term = term.strip("，,。.;:：()（）[]{}")
    if not term:
        return ""
    if any(ch in term for ch in ["$", "\\", "=", "^"]):
        return ""
    if len(term) < 2 or len(term) > MAX_TERM_LEN:
        return ""
    # 过滤泛称与编号型定理/推论等
    if term in GENERIC_MATH_TITLES:
        return ""
    if NUMBERED_TITLE_PATTERN.match(term):
        return ""
    if term in STOPWORDS:
        return ""
    for w in STOPWORDS:
        if w and w in term:
            return ""
    return term


def _is_likely_term(term):
    if not term:
        return False
    if any(hint in term for hint in TERM_HINTS):
        return True
    if len(term) >= 6:
        return False
    return True


def _parse_terms_from_text(text):
    json_block = _extract_json_array(text)
    if not json_block:
        return []
    try:
        data = json.loads(json_block)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict) and "terms" in data:
        data = data["terms"]
    if not isinstance(data, list):
        return []
    terms = []
    for item in data:
        if isinstance(item, str):
            term = _clean_term(item)
        elif isinstance(item, dict) and "term" in item:
            term = _clean_term(item.get("term", ""))
        else:
            continue
        if term and _is_likely_term(term):
            terms.append(term)
    return terms


def _post_clean_terms(terms):
    if not terms:
        return []

    alias_map = {
        "sup": "上确界",
        "inf": "下确界",
    }

    bad_char_pattern = re.compile(r"[\\$=\[\]\(\){}]")
    ascii_only_pattern = re.compile(r"^[A-Za-z0-9._\-]+$")

    cleaned = []
    seen = set()
    for term in terms:
        t = term.strip()
        if not t:
            continue
        if bad_char_pattern.search(t):
            continue
        t_lower = t.lower()
        if t_lower in alias_map:
            t = alias_map[t_lower]
        if ascii_only_pattern.match(t) and t.lower() not in alias_map:
            continue
        t = _clean_term(t)
        if not t or not _is_likely_term(t):
            continue
        if t in seen:
            continue
        seen.add(t)
        cleaned.append(t)
    return cleaned


def _collect_page_files(pages_dir):
    if not os.path.isdir(pages_dir):
        return []
    files = []
    for name in os.listdir(pages_dir):
        if not name.endswith(".md"):
            continue
        if not name.startswith("page_"):
            continue
        files.append(name)
    return sorted(files)


def _parse_page_no(filename):
    m = re.search(r"page_(\d+)", filename)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _extract_terms_for_page(cfg, api_key, page_file, page_no, rate_limiter):
    """提取单页的术语，返回术语列表"""
    text = _read_text(page_file)
    chunks = _split_into_chunks(text)

    page_terms = []
    for chunk in chunks:
        system, user = _build_term_prompt(chunk)
        prompt_tokens = _estimate_tokens(system + user)
        token_budget = prompt_tokens + TERM_MAX_TOKENS

        attempt = 0
        while True:
            attempt += 1
            try:
                rate_limiter.wait(cfg, token_budget)
                result_text = _call_model(
                    cfg,
                    api_key,
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    TERM_MAX_TOKENS,
                    False,
                )
                rate_limiter.mark()
            except Exception as e:
                err = str(e)
                print(f"请求失败，准备重试（第 {attempt} 次）：{err}")
                if "HTTP 429" in err or "HTTP 503" in err:
                    time.sleep(max(cfg["retry_wait_seconds"], 10))
                else:
                    time.sleep(cfg["retry_wait_seconds"])
                if attempt >= cfg["max_attempts"]:
                    break
                continue

            chunk_terms = _parse_terms_from_text(result_text)
            if not chunk_terms:
                print(f"解析失败，准备重试（第 {attempt} 次）")
                time.sleep(cfg["retry_wait_seconds"])
                if attempt >= cfg["max_attempts"]:
                    break
                continue

            page_terms.extend(chunk_terms)
            break

    return _post_clean_terms(page_terms)


def _extract_terms_for_book(cfg, api_key, pages_dir, rate_limiter):
    page_files = _collect_page_files(pages_dir)
    if not page_files:
        print(f"未找到分页 OCR 输出：{pages_dir}")
        return {}

    term_pages = {}
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    iterator = tqdm(page_files, desc="抽取术语") if tqdm else page_files

    for idx, fname in enumerate(iterator, start=1):
        page_no = _parse_page_no(fname)
        if page_no is None:
            continue
        if not tqdm:
            print(f"抽取术语：{idx}/{len(page_files)} 页码 {page_no}")

        path = os.path.join(pages_dir, fname)
        text = _read_text(path)
        chunks = _split_into_chunks(text)

        page_terms = []
        for chunk in chunks:
            system, user = _build_term_prompt(chunk)
            prompt_tokens = _estimate_tokens(system + user)
            token_budget = prompt_tokens + TERM_MAX_TOKENS

            attempt = 0
            while True:
                attempt += 1
                try:
                    rate_limiter.wait(cfg, token_budget)
                    result_text = _call_model(
                        cfg,
                        api_key,
                        [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        TERM_MAX_TOKENS,
                        False,
                    )
                    rate_limiter.mark()
                except Exception as e:
                    err = str(e)
                    if not tqdm:
                        print(f"请求失败，准备重试（第 {attempt} 次）：{err}")
                    if "HTTP 429" in err or "HTTP 503" in err:
                        time.sleep(max(cfg["retry_wait_seconds"], 10))
                    else:
                        time.sleep(cfg["retry_wait_seconds"])
                    if attempt >= cfg["max_attempts"]:
                        break
                    continue

                chunk_terms = _parse_terms_from_text(result_text)
                if not chunk_terms:
                    if not tqdm:
                        print(f"解析失败，准备重试（第 {attempt} 次）")
                    time.sleep(cfg["retry_wait_seconds"])
                    if attempt >= cfg["max_attempts"]:
                        break
                    continue

                page_terms.extend(chunk_terms)
                break

        page_terms = _post_clean_terms(page_terms)
        for term in page_terms:
            term_pages.setdefault(term, set()).add(page_no)

    return {k: sorted(v) for k, v in term_pages.items()}


def _render_prompt(template, **kwargs):
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


def _normalize_key(text):
    if not text:
        return ""
    return re.sub(r"\s+", "", str(text)).lower()


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _extract_json_block(text):
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


def _normalize_record(record, term, subject_label, sources):
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


def _load_page_context(pages_dir, page):
    page_file = os.path.join(pages_dir, f"page_{page:04d}.md")
    if not os.path.isfile(page_file):
        return ""
    try:
        content = _read_text(page_file)
    except Exception:
        return ""
    if content.startswith("<!-- page:"):
        content = content.split("\n", 2)[-1].lstrip()
    return f"[第{page}页]\n{content.strip()}"


def _pick_context(contexts, max_chars):
    if not contexts:
        return ""
    merged = "\n\n".join(contexts)
    if len(merged) <= max_chars:
        return merged
    return merged[:max_chars] + "..."


def _build_sources(term_sources):
    sources = []
    for book_title, page in term_sources:
        sources.append(f"{book_title} 第{page}页")
    if sources:
        return sources
    return ["未知来源"]


def _collect_pdfs(raw_dir):
    if not os.path.isdir(raw_dir):
        return []
    files = []
    for name in os.listdir(raw_dir):
        if name.lower().endswith(".pdf"):
            files.append(name)
    return sorted(files)


def main():
    cfg = _load_config()
    if not cfg["api_base"]:
        print("config.toml 中未配置 model.api_base。")
        return
    if not cfg["model"]:
        print("config.toml 中未配置 model.model。")
        return

    api_key = _load_env_value(config.PROJECT_ROOT, cfg["api_key_env"])
    if not api_key:
        print("未找到 API Key，请检查 .env 或环境变量。")
        return

    pdf_files = _collect_pdfs(config.RAW_DIR)
    if not pdf_files:
        print(f"未找到 PDF：{config.RAW_DIR}")
        return

    os.makedirs(config.OCR_DIR, exist_ok=True)

    rate_limiter = _RateLimiter()

    # 用于跟踪已生成 JSON 的术语（避免重复生成）
    generated_terms = set()

    # 输出文件
    output_path = os.path.join(config.OCR_DIR, "terms_json_all.json")
    out_f = open(output_path, "w", encoding="utf-8")
    out_f.write("[\n")
    first_item = True

    for pdf_name in pdf_files:
        pdf_path = os.path.join(config.RAW_DIR, pdf_name)
        book_title = os.path.splitext(pdf_name)[0]
        book_dir = os.path.join(config.OCR_DIR, book_title)
        pages_dir = os.path.join(book_dir, "pages")
        os.makedirs(book_dir, exist_ok=True)

        print(f"处理书籍：{book_title}")

        # 用于保存该书籍的术语映射
        book_term_pages = {}

        # 逐页处理：OCR -> 提取术语 -> 生成 JSON
        for page_no, page_file, is_new in _ensure_ocr(pdf_path, pages_dir):
            # 提取该页术语
            print(f"  提取术语：页码 {page_no}")
            page_terms = _extract_terms_for_page(
                cfg, api_key, page_file, page_no, rate_limiter
            )

            if not page_terms:
                print(f"  页码 {page_no} 未提取到术语")
                continue

            print(f"  页码 {page_no} 提取到 {len(page_terms)} 个术语")

            # 更新书籍术语映射
            for term in page_terms:
                book_term_pages.setdefault(term, set()).add(page_no)

            # 为新术语生成 JSON（跳过已生成的）
            new_terms = [t for t in page_terms if t not in generated_terms]
            for term in new_terms:
                generated_terms.add(term)

                # 构建来源和上下文
                sources = [f"{book_title} 第{page_no}页"]
                context = _load_page_context(pages_dir, page_no)
                if not context:
                    context = "未找到明确上下文，请基于通用知识补全。"

                print(f"  生成 JSON：{term}")

                record = None
                last_json = ""
                last_reason = ""

                attempt = 0
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
                    prompt_tokens = _estimate_tokens(system + user)
                    token_budget = prompt_tokens + cfg["max_tokens"]

                    try:
                        attempt += 1
                        rate_limiter.wait(cfg, token_budget)
                        gen_text = _call_model(
                            cfg,
                            api_key,
                            messages,
                            cfg["max_tokens"],
                            cfg["stream"],
                        )
                        rate_limiter.mark()
                    except Exception as e:
                        err = str(e)
                        last_reason = f"请求失败：{err}"
                        print(f"  请求失败，准备重试：{term}（第 {attempt} 次）")
                        if "HTTP 429" in err or "HTTP 503" in err:
                            time.sleep(max(cfg["retry_wait_seconds"], 10))
                        else:
                            time.sleep(cfg["retry_wait_seconds"])
                        if attempt >= cfg["max_attempts"]:
                            break
                        continue

                    json_block = _extract_json_block(gen_text)
                    if json_block:
                        last_json = json_block
                        try:
                            record = json.loads(json_block)
                        except json.JSONDecodeError:
                            record = None
                            last_reason = "JSON 解析失败"
                            time.sleep(cfg["retry_wait_seconds"])
                            if attempt >= cfg["max_attempts"]:
                                break
                            continue
                    else:
                        record = None
                        last_json = gen_text
                        last_reason = "未找到 JSON 对象"
                        time.sleep(cfg["retry_wait_seconds"])
                        if attempt >= cfg["max_attempts"]:
                            break
                        continue

                    ok, reason = _quality_check(record, cfg)
                    if ok:
                        break
                    last_reason = reason
                    time.sleep(cfg["retry_wait_seconds"])
                    if attempt >= cfg["max_attempts"]:
                        break

                item = _normalize_record(record, term, cfg["subject_label"], sources)
                if not first_item:
                    out_f.write(",\n")
                out_f.write(json.dumps(item, ensure_ascii=False, indent=2))
                out_f.flush()
                first_item = False

        # 保存该书籍的术语映射
        if book_term_pages:
            term_map_path = os.path.join(book_dir, "terms_map.json")
            with open(term_map_path, "w", encoding="utf-8") as f:
                json.dump(
                    {k: sorted(v) for k, v in book_term_pages.items()},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"  术语映射已保存：{term_map_path}")

    out_f.write("\n]\n")
    out_f.close()
    print(f"输出完成：{output_path}")


if __name__ == "__main__":
    main()
