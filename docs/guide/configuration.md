# 配置说明

项目采用 `config.toml` + `src/core/config.py` 双层配置。

## 配置文件位置

| 文件 | 职责 |
|------|------|
| `config.toml` | 参数数据源（TOML 格式，人工编辑） |
| `src/core/config.py` | Python 读取接口与路径常量，供各模块统一调用 |

## 主要配置段

### `[paths]` — 目录路径

```toml
[paths]
raw_dir = "data/raw"
processed_dir = "data/processed"
evaluation_dir = "data/evaluation"
log_base_dir = "outputs/log"          # 实验跑次目录
reports_base_dir = "outputs/reports"  # 定稿发布目录
figures_dir = "outputs/figures"
```

### `[ocr]` — OCR 配置

```toml
[ocr]
device = "cuda"        # 推理设备："cuda" / "cpu"
mfr_batch_size = 8    # 公式识别批量大小
render_dpi = 200      # PDF 渲染 DPI
```

### `[terms_gen]` — 术语结构化生成

```toml
[terms_gen]
engine = "api"   # "api" / "local"
api_base = "https://api.deepseek.com/v1"
model = "deepseek-chat"
api_key_env = "API-KEY-TERMS"
local_model_dir = ""
temperature = 0.3
```

### `[rag_gen]` — RAG 回答生成

```toml
[rag_gen]
engine = "api"   # "api" / "local"
api_base = "https://api.deepseek.com/v1"
api_model = "deepseek-chat"
api_key_env = "API-KEY-RAG"
local_model_dir = "../Qwen3.5-4B"
temperature = 0.7
```

### `[retrieval]` — 检索参数

```toml
[retrieval]
recall_factor = 10
rrf_k = 60
bm25_default_weight = 0.7
vector_default_weight = 0.3
default_vector_model = "BAAI/bge-base-zh-v1.5"
```

### `[reports_generation]` — 报告生成

```toml
[reports_generation]
report_footer_note = "..."
defense_output_subdir = "defense"
```

## Python 配置接口

```python
from core import config

# 路径常量
config.RAW_DIR
config.PROCESSED_DIR
config.LOG_BASE_DIR
config.REPORTS_BASE_DIR

# 配置读取函数
config.getPathsConfig()
config.get_ocr_config()
config.getRetrievalConfig()
config.getGenerationConfig()
config.getOutputController()
```

详细 API 说明见 [config.py 文档](/core/config)。
