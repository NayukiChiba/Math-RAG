# config.py — 全局配置访问层

## 概述

项目所有模块通过 `import config` 统一获取路径与运行参数。  
内部使用 `@lru_cache` 避免重复解析 `config.toml`。

---

## 路径常量（模块级，直接 import 使用）

| 常量 | 说明 |
|------|------|
| `RAW_DIR` | 原始 PDF 目录 |
| `PROCESSED_DIR` | 处理后数据根目录 |
| `OCR_DIR` | OCR 分页 Markdown 目录 |
| `TERMS_DIR` | 术语映射目录 |
| `CHUNK_DIR` | 术语 JSON 目录 |
| `EVALUATION_DIR` | 评测数据目录 |
| `STATS_DIR` | 统计报告与图表目录 |
| `OUTPUTS_DIR` | 实验输出根目录 |
| `LOG_BASE_DIR` | 运行日志与报告根目录 |
| `REPORTS_BASE_DIR` | 同 `LOG_BASE_DIR`（兼容旧代码） |
| `FIGURES_DIR` | 论文图表目录 |
| `LOGS_DIR` | 日志目录 |
| `RAG_RESULTS_FILE` | RAG 问答结果文件 |

所有路径均为绝对路径，由 `config.toml [paths]` 配置并相对项目根解析。

---

## 函数

### `getPathsConfig() -> dict[str, str]`

读取并解析 `[paths]` section，返回所有路径的 dict。  
结果有 `@lru_cache` 缓存，同进程只解析一次。

### `getPath(name: str) -> str`

按名称获取已解析的路径。

- **参数**：`name` — 路径键名，如 `"chunk_dir"`
- **返回**：绝对路径字符串
- **注意**：键名不存在时抛 `KeyError`

### `get_ocr_config() -> dict`

返回 `[ocr]` section 的配置 dict。  
包含字段：`page_start / page_end / skip_existing / max_image_size / device / mfr_batch_size / render_dpi / resized_shape / text_contain_formula / batch_pages / ocr_workers / max_page_chars / max_term_len / term_max_tokens / max_pages_per_term`

### `getRetrievalConfig() -> dict`

返回 `[retrieval]` section 的配置 dict。  
包含字段：`recall_factor / rrf_k / bm25_default_weight / vector_default_weight / default_vector_model / default_reranker_model / use_hybrid_tokenization / bm25_char_ngram_max` 等。

### `getGenerationConfig() -> dict`

返回 `[generation]` section 的配置 dict。  
包含字段：`temperature / top_p / max_new_tokens / qwen_model_dir / max_context_chars / max_chars_per_term`

### `getOutputController() -> OutputManager`

获取全局 `OutputManager` 实例（基于 `LOG_BASE_DIR`）。

### `getRunLogDir() -> str`

获取当前运行目录：`outputs/log/YYYYMMDD_HHMMSS`  
同一进程内只生成一次，调用多次返回相同目录。

### `getJsonLogDir() -> str`

获取当前运行 JSON 输出目录：`outputs/log/YYYYMMDD_HHMMSS/json`

### `getTextLogDir() -> str`

获取当前运行文本日志目录：`outputs/log/YYYYMMDD_HHMMSS/text`

### `normalizeJsonOutputPath(path: str, defaultName: str) -> str`

将任意路径归一化到当前运行 JSON 目录。

- **参数**：
  - `path`：原始路径（可为空字符串）
  - `defaultName`：默认文件名
- **返回**：归一化后的绝对路径
- **规则**：取 `path` 的 basename，若为空则用 `defaultName`，拼到 `getJsonLogDir()`

### `normalizeTextLogPath(path: str, defaultName: str) -> str`

同上，目标目录为 `getTextLogDir()`。

### `getReportsDir() -> str`

兼容旧接口，等价于 `getJsonLogDir()`。

---

## 依赖

- `utils.FileLoader`（读取 toml）
- `utils.OutputManager`（管理输出目录）
