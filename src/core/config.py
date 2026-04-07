"""项目统一配置入口。"""

import copy
import logging
import os
from functools import lru_cache

from core.utils import OutputManager, getFileLoader, getOutputManager

logger = logging.getLogger(__name__)


def _load_toml(path):
    return getFileLoader().toml(path)


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
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
        logger.warning(
            "读取 config.toml 失败，将使用空字典作为原始配置（各 get*Config 仍合并内置默认值）",
            exc_info=True,
        )
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
        "reports_base_dir": "outputs/reports",
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
        rawValue = paths_cfg.get(key, defaults[key])
        value = str(rawValue).strip()
        if key.endswith("_dir") or key.endswith("_file"):
            resolved[key] = _resolve_path(value)
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
LOG_BASE_DIR = _PATHS["log_base_dir"]
REPORTS_BASE_DIR = _PATHS["reports_base_dir"]
REPORTS_PUBLISH_DIR = REPORTS_BASE_DIR
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


def _deep_merge_reports_generation(defaults: dict, override: dict) -> dict:
    """合并 reports_generation 配置：嵌套 dict 递归合并，其余以 override 为准。"""
    keys = set(defaults) | set(override)
    out: dict = {}
    for key in keys:
        dv, ov = defaults.get(key), override.get(key)
        if (
            key in override
            and key in defaults
            and isinstance(dv, dict)
            and isinstance(ov, dict)
        ):
            out[key] = _deep_merge_reports_generation(dv, ov)
        elif key in override:
            out[key] = ov
        elif key in defaults:
            out[key] = dv
    return out


_REPORTS_GENERATION_DEFAULTS: dict = {
    "defense_output_subdir": "defense",
    "defense_save_dpi": 200,
    "defense_matplotlib_font_size": 12,
    "defense_cjk_font_file": "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "chunk_statistics_basename": "chunkStatistics.json",
    "corpus_relpath": "retrieval/corpus.jsonl",
    "queries_basename": "queries.jsonl",
    "queries_full_basename": "queries_full.jsonl",
    "golden_set_basename": "golden_set.jsonl",
    "term_mapping_basename": "term_mapping.json",
    "report_json_all_methods_relpath": "full_eval/all_methods.json",
    "report_json_ablation_relpath": "ablation_study.json",
    "report_json_significance_relpath": "significance_test.json",
    "report_json_final_report_relpath": "final_report.md",
    "report_json_comparison_relpath": "comparison_results.json",
    "report_doc_figures_display_prefix": "outputs/figures",
    "report_case_examples_count": 3,
    "report_figure_save_pdf_dpi": 300,
    "report_figure_save_png_dpi": 200,
    "report_method_comparison_recall_ks": [1, 3, 5, 10],
    "report_method_comparison_figsize": [10.0, 5.5],
    "report_topk_ablation_figsize": [7.0, 4.5],
    "report_alpha_sensitivity_figsize": [7.0, 4.5],
    "report_subject_breakdown_figsize": [9.0, 5.0],
    "report_chart_colors": [
        "#4472C4",
        "#ED7D31",
        "#A9D18E",
        "#FF6B6B",
        "#9B59B6",
    ],
    "report_chart_hatches": ["", "//", "\\\\", "xx", ".."],
    "report_footer_note": (
        "*本报告由 `reports_generation.reports.generateReport` 自动生成。"
        "定稿与图表见 `outputs/reports/`；原始跑次与完整 JSON 见 `outputs/log/<run_id>/`。*"
    ),
    "fig_method_comparison_basename": "method_comparison",
    "fig_topk_ablation_basename": "topk_ablation",
    "fig_alpha_sensitivity_basename": "alpha_sensitivity",
    "fig_subject_breakdown_basename": "subject_breakdown",
    "chart_comparison_filename": "retrieval_comparison.png",
    "chart_comparison_figsize": [14.0, 10.0],
    "chart_comparison_save_dpi": 300,
    "chart_comparison_suptitle": "检索评测指标对比",
    "chart_comparison_suptitle_fontsize": 16,
    "chart_recall_ks": [1, 3, 5, 10],
    "chart_ndcg_ks": [3, 5, 10],
    "viz_output_subdir": "visualizations",
    "viz_figure_dpi": 100,
    "viz_savefig_dpi": 300,
    "viz_figure_figsize": [12.0, 8.0],
    "defense_matplotlib_fallback_fonts": [
        "SimHei",
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "DejaVu Sans",
    ],
    "report_matplotlib_fallback_fonts": [
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
    ],
    "chart_font_sans_serif": ["SimHei"],
    "viz_windows_font_candidates": [
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/msyhbd.ttc",
        "/mnt/c/Windows/Fonts/msyhl.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
        "/mnt/c/Windows/Fonts/simsun.ttc",
        "/mnt/c/Windows/Fonts/Deng.ttf",
    ],
    "viz_preferred_cn_fonts": [
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "Droid Sans Fallback",
    ],
    "report_subject_breakdown_subjects_zh": ["数学分析", "概率论", "高等代数"],
    "report_subject_breakdown_labels_en": [
        "Math Analysis",
        "Probability",
        "Linear Algebra",
    ],
    "defense_palette": [
        "#3B82F6",
        "#8B5CF6",
        "#10B981",
        "#F59E0B",
        "#EF4444",
        "#EC4899",
        "#06B6D4",
        "#64748B",
    ],
    "defense_gradient_blues": [
        "#DBEAFE",
        "#93C5FD",
        "#60A5FA",
        "#3B82F6",
        "#2563EB",
        "#1D4ED8",
    ],
    "defense_gradient_multi": [
        "#3B82F6",
        "#8B5CF6",
        "#EC4899",
        "#EF4444",
        "#F59E0B",
        "#10B981",
    ],
    "defense_colors": {
        "primary": "#3B82F6",
        "secondary": "#8B5CF6",
        "accent": "#10B981",
        "warm": "#F59E0B",
        "danger": "#EF4444",
        "rose": "#EC4899",
        "cyan": "#06B6D4",
        "slate": "#64748B",
    },
    "quick_eval": {
        "default_mode": "basic",
        "basic_methods": ["bm25", "bm25plus", "hybrid_plus"],
        "optimized_methods": [
            "bm25_heavy",
            "hybrid_more_recall",
            "optimized_hybrid",
            "optimized_rrf",
            "optimized_advanced",
            "extreme_rrf",
        ],
        "all_methods": [
            "bm25",
            "bm25plus",
            "vector",
            "hybrid_plus",
            "hybrid_rrf",
            "advanced",
            "optimized_hybrid",
            "hybrid_more_recall",
            "bm25_heavy",
            "bm25_ultra",
            "optimized_rrf",
            "extreme_rrf",
            "optimized_advanced",
            "advanced_no_rerank",
            "advanced_more_rewrite",
            "bm25plus_only",
            "bm25plus_aggressive",
            "vector_only",
            "direct_lookup_hybrid",
            "direct_lookup_rrf",
            "direct_lookup_bm25_only",
        ],
    },
    "viz_filenames": {
        "book_distribution": "1_书籍术语分布.png",
        "subject_distribution": "2_学科分布.png",
        "field_coverage": "3_字段覆盖率.png",
        "term_length": "4_长度分布.png",
        "definition_type": "5_定义类型分布.png",
        "dashboard": "0_综合统计面板.png",
    },
}


def default_quick_eval_config() -> dict:
    """[reports_generation.quick_eval] 内置默认方法清单，供导入期回退。"""
    return copy.deepcopy(_REPORTS_GENERATION_DEFAULTS["quick_eval"])


@lru_cache(maxsize=1)
def getReportsGenerationConfig() -> dict:
    """
    读取 [reports_generation] 及子表（defense_colors、quick_eval、viz_filenames），
    与内置默认值深度合并。
    """
    data = _get_config_data()
    raw = data.get("reports_generation", {}) if isinstance(data, dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    return _deep_merge_reports_generation(_REPORTS_GENERATION_DEFAULTS, raw)


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
