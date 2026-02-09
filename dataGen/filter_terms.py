"""
使用 AI 过滤术语，移除无意义、OCR 错误、纯符号等噪声。

使用方法：
    python filter_terms.py                    # 过滤所有书的术语
    python filter_terms.py "书名"              # 只过滤指定的书

输出：过滤后的 all_filtered.json / map_filtered.json 保存到同目录。
配置项在 config.toml 的 [model] 部分。
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

# 规范模块搜索路径，保证能定位项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

CONFIG_PATH = config.CONFIG_TOML


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
    }


# 全局配置
_CFG = _load_config()
API_BASE = _CFG["api_base"]
MODEL_NAME = _CFG["model"]
API_KEY = _load_env_value(config.PROJECT_ROOT, _CFG["api_key_env"])
ENDPOINT = _CFG["endpoint"]
TIMEOUT = _CFG["request_timeout"]
TEMPERATURE = _CFG["temperature"]
TOP_P = _CFG["top_p"]
MAX_TOKENS = _CFG["max_tokens"]
RETRY_WAIT = _CFG["retry_wait_seconds"]
MAX_ATTEMPTS = _CFG["max_attempts"]
RPM = _CFG["rpm"]

# RPM 限速
_request_interval = 60.0 / RPM if RPM > 0 else 0
_last_request_time = 0

# 批量处理大小
BATCH_SIZE = 50


def _wait_for_rate_limit():
    """RPM 限速等待。"""
    global _last_request_time
    if _request_interval <= 0:
        return
    elapsed = time.time() - _last_request_time
    if elapsed < _request_interval:
        time.sleep(_request_interval - elapsed)
    _last_request_time = time.time()


def call_llm(prompt: str, system: str = None) -> str:
    """调用 LLM API，支持重试。"""
    if not API_KEY:
        raise ValueError("未找到 API_KEY，请在 .env 或环境变量中配置")

    url = f"{API_BASE}{ENDPOINT}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
    }

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            _wait_for_rate_limit()
            response = requests.post(
                url, json=payload, headers=headers, timeout=TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"  [尝试 {attempt}/{MAX_ATTEMPTS}] API 调用失败: {e}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_WAIT)
            else:
                raise RuntimeError(f"API 调用失败，已重试 {MAX_ATTEMPTS} 次") from e


def filter_terms_batch(terms: list[str]) -> list[str]:
    """
    批量过滤术语，返回有效的术语列表。

    使用 AI 判断哪些术语是有意义的数学概念、定理、公式名称。
    """
    if not terms:
        return []

    system_prompt = """你是一个数学术语专家。你的任务是从给定的术语列表中筛选出有效的数学术语。

**保留标准**：
- 是明确的数学概念、定理、公式、方法的名称
- 是数学分析、高等代数、概率统计等领域的专业术语
- 可以独立理解其含义，不依赖特定上下文

**移除标准**：
- 纯数学符号或表达式（如 "x→0", "+∞", "2/3"）
- 单个变量或参数（如 "x", "y", "n"）
- 明显的 OCR 错误（如 "Q冰间"）
- 无意义的短语片段（如 "下类无端", "两点"）
- 泛指词（如 "例", "证明", "注意"）
- 不完整的术语（如 "的"、"性"）
- 过于通用的词（如 "函数", "定理"单独出现时）

返回格式：只返回保留的术语，每行一个，不要添加编号或其他内容。"""

    user_prompt = f"""请从以下术语列表中筛选出有效的数学术语：

{chr(10).join(terms)}

请只返回保留的术语，每行一个。"""

    try:
        result = call_llm(user_prompt, system_prompt)
        # 解析结果，提取有效术语
        filtered = []
        for line in result.strip().split("\n"):
            line = line.strip()
            if line and line in terms:  # 确保返回的术语在原列表中
                filtered.append(line)
        return filtered
    except Exception as e:
        print(f"  批量过滤失败: {e}")
        return []


def filter_terms_file(book_name: str):
    """过滤指定书的术语文件。"""
    terms_dir = os.path.join(config.PROCESSED_DIR, "terms", book_name)
    all_json = os.path.join(terms_dir, "all.json")
    map_json = os.path.join(terms_dir, "map.json")

    if not os.path.isfile(all_json):
        print(f"[跳过] {book_name}: 未找到 all.json")
        return

    print(f"\n{'=' * 60}")
    print(f"正在过滤: {book_name}")
    print(f"{'=' * 60}")

    # 加载原始术语
    with open(all_json, encoding="utf-8") as f:
        all_terms = json.load(f)

    with open(map_json, encoding="utf-8") as f:
        term_map = json.load(f)

    print(f"原始术语数量: {len(all_terms)}")

    # 分批过滤
    filtered_terms = []
    total_batches = (len(all_terms) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(all_terms), BATCH_SIZE):
        batch = all_terms[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(
            f"  处理批次 {batch_num}/{total_batches} ({len(batch)} 个术语)...",
            end=" ",
            flush=True,
        )

        filtered_batch = filter_terms_batch(batch)
        filtered_terms.extend(filtered_batch)
        print(f"保留 {len(filtered_batch)} 个")

    print(f"过滤后术语数量: {len(filtered_terms)}")
    print(f"移除比例: {(1 - len(filtered_terms) / len(all_terms)) * 100:.1f}%")

    # 构建新的 map
    filtered_map = {}
    for term in filtered_terms:
        if term in term_map:
            filtered_map[term] = term_map[term]

    # 备份原始文件（仅在首次运行时备份，保留最原始数据）
    all_original_json = os.path.join(terms_dir, "all_original.json")
    map_original_json = os.path.join(terms_dir, "map_original.json")

    import shutil

    if not os.path.exists(all_original_json):
        shutil.copy2(all_json, all_original_json)
        shutil.copy2(map_json, map_original_json)
        print("✓ 已备份原始文件:")
        print(f"  - {all_original_json}")
        print(f"  - {map_original_json}")
    else:
        print("✓ 原始备份已存在，跳过备份步骤")

    # 保存过滤后的结果，覆盖原文件
    with open(all_json, "w", encoding="utf-8") as f:
        json.dump(filtered_terms, f, ensure_ascii=False, indent=2)

    with open(map_json, "w", encoding="utf-8") as f:
        json.dump(filtered_map, f, ensure_ascii=False, indent=2)

    print("✓ 已保存过滤结果:")
    print(f"  - {all_json}")
    print(f"  - {map_json}")


def main():
    """主函数。"""
    # 检查配置
    if not API_KEY:
        print("错误: 未找到 API_KEY，请在 .env 或环境变量中配置")
        return

    print(f"使用模型: {MODEL_NAME}")
    print(f"API Base: {API_BASE}")
    print(f"批量大小: {BATCH_SIZE}")

    # 获取所有已处理的书
    terms_root = os.path.join(config.PROCESSED_DIR, "terms")
    if not os.path.isdir(terms_root):
        print(f"错误: terms 目录不存在: {terms_root}")
        return

    books = [
        d for d in os.listdir(terms_root) if os.path.isdir(os.path.join(terms_root, d))
    ]

    # 如果指定了书名，只处理该书
    if len(sys.argv) > 1:
        target_book = sys.argv[1]
        if target_book not in books:
            print(f"错误: 未找到书籍: {target_book}")
            print(f"可用的书籍: {', '.join(books)}")
            return
        books = [target_book]

    if not books:
        print("未找到任何术语文件")
        return

    print(f"\n将处理 {len(books)} 本书:")
    for book in books:
        print(f"  - {book}")

    # 逐本处理
    for book in books:
        try:
            filter_terms_file(book)
        except Exception as e:
            print(f"[错误] {book}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("过滤完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
