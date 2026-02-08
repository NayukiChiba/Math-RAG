"""
从 OCR 每页 Markdown 中抽取术语。

使用方法：
    python extract_terms_from_ocr.py                    # 按顺序处理所有已 OCR 的书
    python extract_terms_from_ocr.py "书名"              # 只处理指定的书
    python extract_terms_from_ocr.py "书名" 100          # 从第 100 页开始处理指定的书

输出：每本书的 all.json / map.json 保存到 processed/terms/<book_name>/ 目录。
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
NUMBERED_TITLE_PATTERN = re.compile(r"^(定理|引理|命题|推论)\s*\d+(?:\.\d+)*$")


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


def _flush_terms_to_disk(term_pages, terms_out_path, terms_map_path):
    """将当前术语数据增量写入磁盘。每页处理完后调用，防止中断丢失进度。"""
    all_terms = sorted(term_pages.keys())
    term_pages_sorted = {k: sorted(v) for k, v in term_pages.items()}

    with open(terms_out_path, "w", encoding="utf-8") as f:
        json.dump(all_terms, f, ensure_ascii=False, indent=2)
    with open(terms_map_path, "w", encoding="utf-8") as f:
        json.dump(term_pages_sorted, f, ensure_ascii=False, indent=2)


def _collect_page_files(pages_dir):
    """收集指定目录下的分页 MD 文件。"""
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


def _collect_book_dirs():
    """扫描 OCR 输出目录，返回按名称排序的书名列表（仅包含有 pages/ 的目录）。"""
    if not os.path.isdir(config.OCR_DIR):
        return []
    books = []
    for name in sorted(os.listdir(config.OCR_DIR)):
        book_dir = os.path.join(config.OCR_DIR, name)
        pages_dir = os.path.join(book_dir, "pages")
        if os.path.isdir(pages_dir):
            books.append(name)
    return books


def _parse_page_no(filename):
    m = re.search(r"page_(\d+)", filename)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _extract_terms_for_book(book_name, cfg, api_key, start_page=None):
    """
    为单本书抽取术语。

    Args:
        book_name: 书名（OCR 目录名）
        cfg: 模型配置
        api_key: API 密钥
        start_page: 起始页码（可选），控制从第几页开始处理

    Returns:
        (术语列表, 术语-页码映射字典) 或 (None, None) 如果失败
    """
    # OCR 分页输入路径
    ocr_book_dir = os.path.join(config.OCR_DIR, book_name)
    pages_dir = os.path.join(ocr_book_dir, "pages")

    # 输出路径：processed/terms/{书名}/
    terms_book_dir = os.path.join(config.TERMS_DIR, book_name)
    os.makedirs(terms_book_dir, exist_ok=True)
    terms_out_path = os.path.join(terms_book_dir, "all.json")
    terms_map_path = os.path.join(terms_book_dir, "map.json")

    page_files = _collect_page_files(pages_dir)
    if not page_files:
        print(f"  未找到分页 OCR 输出: {pages_dir}")
        return None, None

    print(f"\n{'=' * 60}")
    print(f"抽取术语: {book_name}")
    print(f"页数: {len(page_files)}")
    if start_page is not None:
        print(f"起始页: {start_page}")
    print(f"{'=' * 60}")

    # 加载已有数据，支持断点续写
    term_pages = {}
    if os.path.isfile(terms_map_path):
        try:
            with open(terms_map_path, encoding="utf-8") as f:
                existing = json.load(f)
            # 将已有页码列表还原为 set
            for t, pages in existing.items():
                term_pages[t] = set(pages)
            print(f"  已加载已有术语 {len(term_pages)} 个，将在此基础上续写")
        except (json.JSONDecodeError, Exception) as e:
            print(f"  加载已有术语文件失败，将从零开始: {e}")
            term_pages = {}

    last_request_at = 0.0

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    iterator = (
        tqdm(page_files, desc=f"抽取术语({book_name[:15]})") if tqdm else page_files
    )

    for idx, fname in enumerate(iterator, start=1):
        page_no = _parse_page_no(fname)
        if page_no is None:
            continue

        # 跳过起始页之前的页面
        if start_page is not None and page_no < start_page:
            continue

        if not tqdm:
            print(f"  处理中：{idx}/{len(page_files)} 页码 {page_no}")

        path = os.path.join(pages_dir, fname)
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
                        print(f"  请求失败，准备重试（第 {attempt} 次）：{err}")
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
                        print(f"  解析失败，准备重试（第 {attempt} 次）")
                    time.sleep(cfg["retry_wait_seconds"])
                    if attempt >= max_attempts:
                        break
                    continue

                page_terms.extend(chunk_terms)
                break

        page_terms = _post_clean_terms(page_terms)
        for term in page_terms:
            term_pages.setdefault(term, set()).add(page_no)

        # 打印本页抽取到的术语
        if page_terms:
            print(
                f"  页码 {page_no} 抽取到 {len(page_terms)} 个术语: {', '.join(page_terms)}"
            )
        else:
            print(f"  页码 {page_no} 未抽取到术语")

        # 每处理完一页立即写入 JSON，防止中断丢失进度
        if page_terms:
            _flush_terms_to_disk(term_pages, terms_out_path, terms_map_path)

    if not term_pages:
        print(f"  未抽取到术语: {book_name}")
        return None, None

    # 最终写入一次，确保数据完整
    _flush_terms_to_disk(term_pages, terms_out_path, terms_map_path)

    all_terms = sorted(term_pages.keys())
    term_pages_sorted = {k: sorted(v) for k, v in term_pages.items()}

    print(f"  抽取术语数量: {len(all_terms)}")
    print(f"  术语文件: {terms_out_path}")
    print(f"  页码映射: {terms_map_path}")

    return all_terms, term_pages_sorted


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

    # 确定要处理的书目列表和起始页
    start_page = None
    if len(sys.argv) > 1:
        # 指定了书名参数
        book_name = sys.argv[1]
        # 去掉可能的 .pdf 后缀
        if book_name.lower().endswith(".pdf"):
            book_name = book_name[:-4]

        # 检查是否提供了起始页参数
        if len(sys.argv) > 2:
            try:
                start_page = int(sys.argv[2])
            except ValueError:
                print(f"起始页参数无效: {sys.argv[2]}，必须是整数")
                return

        book_dir = os.path.join(config.OCR_DIR, book_name)
        if not os.path.isdir(book_dir):
            print(f"未找到 OCR 输出目录: {book_dir}")
            return
        book_list = [book_name]
    else:
        # 未指定，按顺序处理所有已 OCR 的书
        book_list = _collect_book_dirs()
        if not book_list:
            print(f"OCR 目录下未找到已处理的书: {config.OCR_DIR}")
            return
        print(f"找到 {len(book_list)} 本已 OCR 的书:")
        for b in book_list:
            print(f"  - {b}")

    total_books = 0
    total_terms = 0

    for book_name in book_list:
        terms, _ = _extract_terms_for_book(
            book_name, cfg, api_key, start_page=start_page
        )
        if terms:
            total_terms += len(terms)
            total_books += 1

    print(f"\n{'=' * 60}")
    print(f"全部完成: 处理 {total_books} 本书, 总术语 {total_terms} 个")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
