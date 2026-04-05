"""项目统一配置入口。"""

import os
from functools import lru_cache

from utils import OutputManager, getFileLoader, getOutputManager


def _load_toml(path):
    return getFileLoader().toml(path)


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
        return {}
    try:
        return _load_toml(CONFIG_TOML)
    except Exception:
        return {}


@lru_cache(maxsize=1)
def getPathsConfig() -> dict[str, str]:
    """读取并解析 [paths] 配置。"""
    data = _get_config_data()
    paths_cfg = data.get("paths", {}) if isinstance(data, dict) else {}
    defaults = {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "ocr_dir": "data/processed/ocr",
        "terms_dir": "data/processed/terms",
        "chunk_dir": "data/processed/chunks",
        "evaluation_dir": "data/evaluation",
        "stats_dir": "data/stats",
        "outputs_dir": "outputs",
        "log_base_dir": "outputs/log",
        "reports_base_dir": "outputs/log",
        "figures_dir": "outputs/figures",
        "logs_dir": "outputs/log",
        "rag_results_file": "outputs/rag_results.jsonl",
    }
    supported_keys = [
        "raw_dir",
        "processed_dir",
        "ocr_dir",
        "terms_dir",
        "chunk_dir",
        "evaluation_dir",
        "stats_dir",
        "outputs_dir",
        "log_base_dir",
        "reports_base_dir",
        "figures_dir",
        "logs_dir",
        "rag_results_file",
    ]

    resolved = {}
    for key in supported_keys:
        if key == "log_base_dir":
            rawValue = paths_cfg.get(
                "log_base_dir",
                paths_cfg.get("reports_base_dir", defaults["log_base_dir"]),
            )
        else:
            rawValue = paths_cfg.get(key, defaults[key])
        value = str(rawValue).strip()
        if key.endswith("_dir") or key.endswith("_file"):
            resolved[key] = _resolve_path(value)
        else:
            resolved[key] = value

    # 兼容旧代码中对 REPORTS_BASE_DIR 的读取，统一映射到 log_base_dir
    resolved["reports_base_dir"] = resolved["log_base_dir"]
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
LOG_BASE_DIR = _PATHS["log_base_dir"]
REPORTS_BASE_DIR = _PATHS["reports_base_dir"]
FIGURES_DIR = _PATHS["figures_dir"]
LOGS_DIR = _PATHS["logs_dir"]
RAG_RESULTS_FILE = _PATHS["rag_results_file"]

# 缓存：同一进程内只生成一次时间戳目录
_runLogDir = None
_jsonLogDir = None
_textLogDir = None
_outputController = None


def _get_output_controller() -> OutputManager:
    """获取全局输出控制器（若基础目录变化则自动刷新）。"""
    global _outputController
    _outputController = getOutputManager(LOG_BASE_DIR)
    return _outputController


def getOutputController() -> OutputManager:
    """对外暴露统一输出控制器。"""
    return _get_output_controller()


def getRunLogDir() -> str:
    """获取当前运行目录：outputs/log/YYYYMMDD_HHMMSS。"""
    global _runLogDir
    _runLogDir = _get_output_controller().get_run_dir()
    return _runLogDir


def getJsonLogDir() -> str:
    """获取当前运行 JSON 输出目录：outputs/log/YYYYMMDD_HHMMSS/json。"""
    global _jsonLogDir
    _jsonLogDir = _get_output_controller().get_json_dir()
    return _jsonLogDir


def getTextLogDir() -> str:
    """获取当前运行文本日志目录：outputs/log/YYYYMMDD_HHMMSS/text。"""
    global _textLogDir
    _textLogDir = _get_output_controller().get_text_dir()
    return _textLogDir


def normalizeJsonOutputPath(path: str, defaultName: str) -> str:
    """
    将任意输出路径归一化到当前运行 JSON 目录。

    规则：outputs/log/YYYYMMDD_HHMMSS/json/<filename>
    """
    return _get_output_controller().normalize_json_path(path, defaultName)


def normalizeTextLogPath(path: str, defaultName: str) -> str:
    """将任意日志路径归一化到当前运行 text 目录。"""
    return _get_output_controller().normalize_text_path(path, defaultName)


def getReportsDir() -> str:
    """
    兼容旧接口：返回当前运行 JSON 输出目录

    统一输出规范：outputs/log/YYYYMMDD_HHMMSS/json

    Returns:
        str: JSON 输出目录的绝对路径
    """
    return getJsonLogDir()


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


def _getLocalModelDir() -> str:
    """从 config.toml 读取本地模型目录。"""
    try:
        data = _get_config_data()
    except Exception:
        return ""

    gen_cfg = data.get("generation", {})
    # 优先读取 local_model_dir，向后兼容 qwen_model_dir
    modelDir = gen_cfg.get("local_model_dir", gen_cfg.get("qwen_model_dir", "")).strip()

    if not modelDir:
        return ""
    return _resolve_path(modelDir)


# 本地模型路径
LOCAL_MODEL_DIR = _getLocalModelDir()
# 向后兼容旧代码
QWEN_MODEL_DIR = LOCAL_MODEL_DIR


def getGenerationConfig() -> dict:
    """
    获取生成层相关配置

    Returns:
        dict，包含 engine, max_context_chars、max_chars_per_term、temperature、
        top_p、max_new_tokens 等字段
    """
    defaults = {
        "engine": "api",
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
        "engine": gen_cfg.get("engine", defaults["engine"]),
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


def getApiConfig() -> dict:
    """读取 RAG 生成层的 API 配置（优先从 [generation]，回退到 [model]）。"""
    try:
        data = _get_config_data()
    except Exception:
        data = {}

    gen_cfg = data.get("generation", {})
    model_cfg = data.get("model", {})

    # 优先从 [generation] 读取专属字段，否则回退到 [model]
    return {
        "api_base": gen_cfg.get(
            "api_base", model_cfg.get("api_base", "https://api.deepseek.com/v1")
        ),
        "model": gen_cfg.get("api_model", model_cfg.get("model", "deepseek-chat")),
        "api_key_env": gen_cfg.get(
            "api_key_env", model_cfg.get("api_key_env", "API-KEY")
        ),
        "stream": gen_cfg.get("api_stream", False),
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
        "out_of_scope_score_threshold": 0.80,
        "no_overlap_strict_score_threshold": 0.88,
        "overlap_min_chars": 2,
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
