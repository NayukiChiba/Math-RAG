"""
从 OCR 每页 Markdown 中抽取术语，并写入 config.toml 的 seed_terms。
说明：不使用命令行参数，直接运行即可。
策略：使用模型抽取术语，并记录术语所在页码。
配置项在 config.toml 的 [ocr] 和 [model] 部分。
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
MAX_TERM_LEN = _ocr_cfg["max_term_len"]
MAX_PAGE_CHARS = _ocr_cfg["max_page_chars"]

# 书名与输出目录
PDF_NAME = "数学分析(第5版) 上 (华东师范大学数学系).pdf"
BOOK_DIR = os.path.join(config.OCR_DIR, os.path.splitext(PDF_NAME)[0])
PAGES_DIR = os.path.join(BOOK_DIR, "pages")

TERMS_OUT_PATH = os.path.join(config.PROCESSED_DIR, "terms_from_ocr.txt")
TERMS_MAP_PATH = os.path.join(config.PROCESSED_DIR, "terms_from_ocr_map.json")
CONFIG_PATH = config.CONFIG_TOML

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

# 数学术语常见后缀/关键词，用于过滤噪声
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
]

# 泛称与编号型定理过滤
GENERIC_MATH_TITLES = {"定理", "引理", "命题", "推论"}
NUMBERED_TITLE_PATTERN = re.compile(r"^(定理|引理|命题|推论)\\s*\\d+(?:\\.\\d+)*$")

MAX_TERM_LEN = 16
MAX_PAGE_CHARS = 1800


def _load_toml(path):
    """读取 TOML 配置。"""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


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


def _load_config():
    """从 config.toml 读取模型配置。"""
    data = _load_toml(CONFIG_PATH)
    model_cfg = data.get("model", {})
    return {
        "api_base": model_cfg.get("api_base", "").rstrip("/"),
        "model": model_cfg.get("model", ""),
        "api_key_env": model_cfg.get("api_key_env", "API-KEY"),
        "request_timeout": model_cfg.get("request_timeout", 60),
        "endpoint": model_cfg.get("endpoint", "/chat/completions"),
        "temperature": model_cfg.get("temperature", 0.2),
        "top_p": model_cfg.get("top_p", 0.9),
        "max_tokens": model_cfg.get("max_tokens", 600),
        "retry_wait_seconds": model_cfg.get("retry_wait_seconds", 2),
        "max_attempts": model_cfg.get("max_attempts", 5),
        "rpm": model_cfg.get("rpm", 0),
        "tpm": model_cfg.get("tpm", 0),
    }


def _read_text(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到文件：{path}")
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


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


def _split_into_chunks(text, max_chars=MAX_PAGE_CHARS):
    """按段落切分并合并成固定长度的块。"""
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


def _build_prompt(chunk):
    system = (
        "你是数学术语抽取助手。"
        "请从给定文本中抽取数学分析相关术语（概念/定义/定理/引理/命题/推论名称）。"
        "仅输出 JSON 数组，元素为术语字符串，不要输出任何额外说明。"
    )
    user = (
        "文本如下（OCR）：\n"
        f"{chunk}\n\n"
        "要求：\n"
        "1) 只输出 JSON 数组。\n"
        "2) 术语尽量为名词短语，避免代词/泛指词。\n"
        "3) 不要输出句子。\n"
        "4) 不要重复。\n"
    )
    return system, user


def _estimate_tokens(text):
    """粗略估算 token 数量（按 2 字符≈1 token）。"""
    if not text:
        return 0
    return max(1, len(text) // 2)


def _call_model(cfg, api_key, system, user):
    """调用 OpenAI 兼容接口。"""
    import requests

    url = f"{cfg['api_base']}{cfg['endpoint']}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "max_tokens": cfg["max_tokens"],
        "stream": False,
    }
    resp = requests.post(
        url, headers=headers, json=payload, timeout=cfg["request_timeout"]
    )
    if resp.status_code in (429, 503):
        raise RuntimeError(f"HTTP {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


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
    """对术语列表进行二次清洗与归一化。"""
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


def _replace_seed_terms(config_text, terms):
    lines = config_text.splitlines()
    out = []
    in_model = False
    skipping = False
    replaced = False

    seed_block = ["seed_terms = ["]
    for t in terms:
        seed_block.append(f'  "{t}",')
    seed_block.append("]")

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[model]"):
            in_model = True
            out.append(line)
            continue
        if (
            in_model
            and stripped.startswith("[")
            and stripped.endswith("]")
            and not stripped.startswith("[model]")
        ):
            in_model = False

        if in_model and stripped.startswith("seed_terms"):
            out.extend(seed_block)
            replaced = True
            if "]" not in line:
                skipping = True
            continue

        if skipping:
            if "]" in line:
                skipping = False
            continue

        out.append(line)

    if not replaced:
        out2 = []
        inserted = False
        for line in out:
            out2.append(line)
            if not inserted and line.strip().startswith("[model]"):
                out2.extend(seed_block)
                inserted = True
        out = out2

    return "\n".join(out) + "\n"


def _collect_page_files():
    if not os.path.isdir(PAGES_DIR):
        return []
    files = []
    for name in os.listdir(PAGES_DIR):
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

    page_files = _collect_page_files()
    if not page_files:
        print("未找到分页 OCR 输出，请先运行 pix2text_ocr.py。")
        return

    term_pages = {}
    last_request_at = 0.0

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
            print(f"处理中：{idx}/{len(page_files)} 页码 {page_no}")

        path = os.path.join(PAGES_DIR, fname)
        text = _read_text(path)
        chunks = _split_into_chunks(text)

        page_terms = []
        for chunk in chunks:
            system, user = _build_prompt(chunk)
            prompt_tokens = _estimate_tokens(system + user)
            token_budget = prompt_tokens + cfg["max_tokens"]

            attempt = 0
            max_attempts = cfg["max_attempts"]
            while True:
                attempt += 1
                try:
                    now = time.time()
                    min_interval = 0.0
                    if cfg["rpm"] and cfg["rpm"] > 0:
                        min_interval = 60.0 / cfg["rpm"]
                    if cfg["tpm"] and cfg["tpm"] > 0:
                        min_interval_tpm = 60.0 * token_budget / cfg["tpm"]
                    else:
                        min_interval_tpm = 0.0
                    min_interval = max(min_interval, min_interval_tpm)
                    sleep_for = min_interval - (now - last_request_at)
                    if sleep_for > 0:
                        time.sleep(sleep_for)

                    result_text = _call_model(cfg, api_key, system, user)
                    last_request_at = time.time()
                except Exception as e:
                    err = str(e)
                    if not tqdm:
                        print(f"请求失败，准备重试（第 {attempt} 次）：{err}")
                    if "HTTP 429" in err or "HTTP 503" in err:
                        time.sleep(max(cfg["retry_wait_seconds"], 10))
                    else:
                        time.sleep(cfg["retry_wait_seconds"])
                    if attempt >= max_attempts:
                        break
                    continue

                chunk_terms = _parse_terms_from_text(result_text)
                if not chunk_terms:
                    if not tqdm:
                        print(f"解析失败，准备重试（第 {attempt} 次）")
                    time.sleep(cfg["retry_wait_seconds"])
                    if attempt >= max_attempts:
                        break
                    continue

                page_terms.extend(chunk_terms)
                break

        page_terms = _post_clean_terms(page_terms)
        for term in page_terms:
            term_pages.setdefault(term, set()).add(page_no)

    if not term_pages:
        print("未抽取到术语，请检查 OCR 输出质量或模型配置。")
        return

    # 生成术语列表与映射
    all_terms = sorted(term_pages.keys())
    term_pages_sorted = {k: sorted(v) for k, v in term_pages.items()}

    os.makedirs(os.path.dirname(TERMS_OUT_PATH), exist_ok=True)
    with open(TERMS_OUT_PATH, "w", encoding="utf-8") as f:
        for term in all_terms:
            pages = ",".join(str(p) for p in term_pages_sorted[term])
            f.write(f"{term}\tpages={pages}\n")

    with open(TERMS_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(term_pages_sorted, f, ensure_ascii=False, indent=2)

    # 写回 config.toml
    config_text = _read_text(CONFIG_PATH)
    new_text = _replace_seed_terms(config_text, all_terms)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(new_text)

    print(f"抽取术语数量：{len(all_terms)}")
    print(f"已写入：{TERMS_OUT_PATH}")
    print(f"页码映射：{TERMS_MAP_PATH}")


if __name__ == "__main__":
    main()
