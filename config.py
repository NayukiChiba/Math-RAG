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
# 评测数据目录（查询集在 data/evaluation 而非 processed）
EVALUATION_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation")
# 报告输出根目录
REPORTS_BASE_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")

# 缓存：同一进程内只生成一次时间戳目录
_reportsDir = None


def getReportsDir() -> str:
    """
    获取当前运行对应的报告输出目录

    每次运行自动在 outputs/reports/ 下创建以当前时间命名的子目录，
    格式为 YYYYMMDD_HHMMSS。同一进程中多次调用返回同一个目录。

    Returns:
        str: 报告输出目录的绝对路径
    """
    global _reportsDir
    if _reportsDir is None:
        import time

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        _reportsDir = os.path.join(REPORTS_BASE_DIR, timestamp)
        os.makedirs(_reportsDir, exist_ok=True)
    return _reportsDir


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


def getRetrievalConfig() -> dict:
    """
    获取检索模块配置

    Returns:
        dict，包含 recall_factor、rrf_k、bm25_default_weight、
        default_vector_model、default_reranker_model 等字段
    """
    defaults = {
        "recall_factor": 5,
        "advanced_recall_topk": 100,
        "rerank_candidates": 50,
        "rrf_k": 60,
        "rrf_min_k": 30,
        "rrf_max_k": 100,
        "bm25_default_weight": 0.7,
        "vector_default_weight": 0.3,
        "overlap_threshold": 0.5,
        "bm25_difficult_threshold_low": 0.5,
        "bm25_difficult_threshold_high": 2.0,
        "rewrite_query_count": 3,
        "rewrite_max_terms": 10,
        "default_normalization": "percentile",
        "default_reranker_model": "BAAI/bge-reranker-v2-mixed",
        "default_vector_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "use_hybrid_tokenization": True,
        "eval_num_queries": 20,
        "eval_topk": 10,
        "eval_hybrid_alpha": 0.85,
        "eval_hybrid_beta": 0.15,
    }

    if not os.path.isfile(CONFIG_TOML):
        return defaults

    try:
        data = _load_toml(CONFIG_TOML)
    except Exception:
        return defaults

    ret_cfg = data.get("retrieval", {})
    result = {}
    for key, defaultVal in defaults.items():
        result[key] = ret_cfg.get(key, defaultVal)
    return result
