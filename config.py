"""项目统一配置入口。"""

import os
from functools import lru_cache


def _load_toml(path):
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_TOML = os.path.join(PROJECT_ROOT, "config.toml")


def _resolve_path(path_value: str) -> str:
    if not path_value:
        raise ValueError("配置路径不能为空")
    if os.path.isabs(path_value):
        return os.path.abspath(path_value)
    return os.path.abspath(os.path.join(PROJECT_ROOT, path_value))


@lru_cache(maxsize=1)
def _get_config_data() -> dict:
    if not os.path.isfile(CONFIG_TOML):
        raise FileNotFoundError(f"配置文件不存在: {CONFIG_TOML}")
    return _load_toml(CONFIG_TOML)


@lru_cache(maxsize=1)
def getPathsConfig() -> dict[str, str]:
    """读取并解析 [paths] 配置。"""
    data = _get_config_data()
    paths_cfg = data.get("paths", {})
    required_keys = [
        "raw_dir",
        "processed_dir",
        "ocr_dir",
        "terms_dir",
        "chunk_dir",
        "evaluation_dir",
        "stats_dir",
        "outputs_dir",
        "reports_base_dir",
        "figures_dir",
        "logs_dir",
        "rag_results_file",
    ]
    missing = [key for key in required_keys if not paths_cfg.get(key)]
    if missing:
        joined = ", ".join(missing)
        raise KeyError(f"config.toml [paths] 缺少必填项: {joined}")

    resolved = {}
    for key, value in paths_cfg.items():
        if key.endswith("_dir") or key.endswith("_file"):
            resolved[key] = _resolve_path(str(value).strip())
        else:
            resolved[key] = value
    return resolved


def getPath(name: str) -> str:
    """按名称获取已解析的路径配置。"""
    return getPathsConfig()[name]


_PATHS = getPathsConfig()

RAW_DIR = _PATHS["raw_dir"]
PROCESSED_DIR = _PATHS["processed_dir"]
OCR_DIR = _PATHS["ocr_dir"]
TERMS_DIR = _PATHS["terms_dir"]
CHUNK_DIR = _PATHS["chunk_dir"]
EVALUATION_DIR = _PATHS["evaluation_dir"]
STATS_DIR = _PATHS["stats_dir"]
OUTPUTS_DIR = _PATHS["outputs_dir"]
REPORTS_BASE_DIR = _PATHS["reports_base_dir"]
FIGURES_DIR = _PATHS["figures_dir"]
LOGS_DIR = _PATHS["logs_dir"]
RAG_RESULTS_FILE = _PATHS["rag_results_file"]

# 缓存：同一进程内只生成一次时间戳目录
_reportsDir = None


def getReportsDir() -> str:
    """
    获取当前运行对应的报告输出目录

    每次运行自动在 reports_base_dir 下创建以当前时间命名的子目录，
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

    try:
        data = _get_config_data()
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

    # 加速相关配置
    result["device"] = ocr_cfg.get("device", "") or None
    result["mfr_batch_size"] = ocr_cfg.get("mfr_batch_size", 1)
    result["render_dpi"] = ocr_cfg.get("render_dpi", 300)
    result["resized_shape"] = ocr_cfg.get("resized_shape", 768)
    result["text_contain_formula"] = ocr_cfg.get("text_contain_formula", True)
    result["batch_pages"] = ocr_cfg.get("batch_pages", 0)
    result["ocr_workers"] = ocr_cfg.get("ocr_workers", 0)

    return result


def _getQwenModelDir() -> str:
    """从 config.toml 读取 Qwen 模型目录。"""
    try:
        data = _get_config_data()
    except Exception:
        return ""

    gen_cfg = data.get("generation", {})
    qwenDir = gen_cfg.get("qwen_model_dir", "").strip()

    if not qwenDir:
        return ""
    return _resolve_path(qwenDir)


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

    try:
        data = _get_config_data()
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
        "default_vector_model": "BAAI/bge-base-zh-v1.5",
        "use_hybrid_tokenization": True,
        "bm25_char_ngram_max": 3,
        "eval_num_queries": 20,
        "eval_topk": 10,
        "eval_hybrid_alpha": 0.85,
        "eval_hybrid_beta": 0.15,
    }

    try:
        data = _get_config_data()
    except Exception:
        return defaults

    ret_cfg = data.get("retrieval", {})
    result = {}
    for key, defaultVal in defaults.items():
        result[key] = ret_cfg.get(key, defaultVal)
    return result
