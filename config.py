"""
项目路径配置（统一使用 os.path）。
"""

import os


def _load_toml(path):
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_TOML = os.path.join(PROJECT_ROOT, "config.toml")


def _get_processed_dir():
    default_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    if not os.path.isfile(CONFIG_TOML):
        return default_dir

    try:
        data = _load_toml(CONFIG_TOML)
    except Exception:
        return default_dir

    paths_cfg = data.get("paths", {})
    processed_dir = paths_cfg.get("processed_dir") or ""
    if not processed_dir:
        return default_dir

    if os.path.isabs(processed_dir):
        return os.path.abspath(processed_dir)
    return os.path.abspath(os.path.join(PROJECT_ROOT, processed_dir))


RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = _get_processed_dir()
OCR_DIR = os.path.join(PROCESSED_DIR, "ocr")
# 术语列表与映射：processed/terms/{书名}/all.json, map.json
TERMS_DIR = os.path.join(PROCESSED_DIR, "terms")
# 术语定义数据（每个术语一个 JSON）：processed/chunk/{书名}/{term}.json
CHUNK_DIR = os.path.join(PROCESSED_DIR, "chunk")


def get_ocr_config():
    """获取 OCR 相关配置"""
    defaults = {
        "page_start": 0,
        "page_end": None,  # -1 在 toml 中表示 None
        "skip_existing": True,
        "max_image_size": (1280, 1280),
        "max_page_chars": 1800,
        "max_term_len": 16,
        "term_max_tokens": 300,
        "max_pages_per_term": 6,
    }

    if not os.path.isfile(CONFIG_TOML):
        return defaults

    try:
        data = _load_toml(CONFIG_TOML)
    except Exception:
        return defaults

    ocr_cfg = data.get("ocr", {})

    result = {}
    result["page_start"] = ocr_cfg.get("page_start", defaults["page_start"])

    page_end = ocr_cfg.get("page_end", -1)
    result["page_end"] = None if page_end == -1 else page_end

    result["skip_existing"] = ocr_cfg.get("skip_existing", defaults["skip_existing"])

    max_image_size = ocr_cfg.get("max_image_size", list(defaults["max_image_size"]))
    result["max_image_size"] = (
        tuple(max_image_size) if isinstance(max_image_size, list) else max_image_size
    )

    result["max_page_chars"] = ocr_cfg.get("max_page_chars", defaults["max_page_chars"])
    result["max_term_len"] = ocr_cfg.get("max_term_len", defaults["max_term_len"])
    result["term_max_tokens"] = ocr_cfg.get(
        "term_max_tokens", defaults["term_max_tokens"]
    )
    result["max_pages_per_term"] = ocr_cfg.get(
        "max_pages_per_term", defaults["max_pages_per_term"]
    )

    return result


def _getQwenModelDir() -> str:
    """从 config.toml 读取 Qwen 模型目录，默认为项目上级目录下的 Qwen-model（与 Math-RAG 同级）"""
    defaultDir = os.path.join(PROJECT_ROOT, "..", "Qwen-model")

    if not os.path.isfile(CONFIG_TOML):
        return os.path.abspath(defaultDir)

    try:
        data = _load_toml(CONFIG_TOML)
    except Exception:
        return os.path.abspath(defaultDir)

    gen_cfg = data.get("generation", {})
    qwenDir = gen_cfg.get("qwen_model_dir", "").strip()

    if not qwenDir:
        return os.path.abspath(defaultDir)

    if os.path.isabs(qwenDir):
        return os.path.abspath(qwenDir)
    return os.path.abspath(os.path.join(PROJECT_ROOT, qwenDir))


# Qwen 模型本地路径
QWEN_MODEL_DIR = _getQwenModelDir()


def getGenerationConfig() -> dict:
    """
    获取生成层相关配置

    Returns:
        dict，包含 max_context_chars、max_chars_per_term、temperature、
        top_p、max_new_tokens 等字段
    """
    defaults = {
        "max_context_chars": 2000,
        "max_chars_per_term": 800,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_new_tokens": 512,
    }

    if not os.path.isfile(CONFIG_TOML):
        return defaults

    try:
        data = _load_toml(CONFIG_TOML)
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
        "temperature": gen_cfg.get("temperature", defaults["temperature"]),
        "top_p": gen_cfg.get("top_p", defaults["top_p"]),
        "max_new_tokens": gen_cfg.get("max_new_tokens", defaults["max_new_tokens"]),
    }
