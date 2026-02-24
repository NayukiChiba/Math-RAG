# dataGen - 数据生成模块

从 PDF 教材生成结构化数学术语数据的完整流水线。

## 模块结构

```
dataGen/
├── pix2text_ocr.py             # PDF → OCR Markdown
├── extract_terms_from_ocr.py   # OCR → 术语映射
├── data_gen.py                 # 术语 → 结构化 JSON（调用 LLM）
├── filter_terms.py             # 术语过滤与质量清洗
├── clean_failed_ocr.py         # 清理 OCR 失败数据
└── __init__.py
```

## 处理流程

```
data/raw/**/*.pdf
        ↓
  [pix2text_ocr.py]             PDF → 分页 OCR Markdown
        ↓
data/processed/ocr/<书名>/pages/*.md
        ↓
  [extract_terms_from_ocr.py]   OCR → 术语-页码映射
        ↓
data/processed/terms/<书名>/all.json + map.json
        ↓
  [data_gen.py]                 术语 → 结构化定义 JSON（DeepSeek API）
        ↓
data/processed/chunk/<书名>/<term>.json
        ↓
  [filter_terms.py]             可选：过滤低质量术语
```

## 各脚本说明

### pix2text_ocr.py

将 `data/raw/` 下的 PDF 教材进行 OCR，输出分页 Markdown 文件。

**依赖**：`pix2text`

**配置**（`config.toml [ocr]`）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `page_start` | 起始页（0-indexed） | 0 |
| `page_end` | 结束页（-1 表示不限） | -1 |
| `skip_existing` | 跳过已处理页 | true |
| `max_image_size` | 图像最大尺寸 | [1280, 1280] |
| `max_page_chars` | 每页最大字符数 | 1800 |

**输出**：`data/processed/ocr/<书名>/pages/<page>.md`

```bash
python dataGen/pix2text_ocr.py
```

---

### extract_terms_from_ocr.py

从 OCR 结果中提取数学术语，生成术语-页码映射。

**输入**：`data/processed/ocr/<书名>/pages/*.md`

**输出**：
- `data/processed/terms/<书名>/all.json`（术语列表）
- `data/processed/terms/<书名>/map.json`（术语-页码映射）

```bash
python dataGen/extract_terms_from_ocr.py
```

---

### data_gen.py

调用 DeepSeek API（OpenAI 兼容接口），为每个术语生成结构化定义 JSON。

**输入**：`data/processed/terms/<书名>/map.json`

**输出**：`data/processed/chunk/<书名>/<term>.json`（每个术语一个文件）

**配置**（`config.toml [model]`）：

| 参数 | 说明 |
|------|------|
| `api_base` | API 地址（默认 DeepSeek） |
| `model` | 模型名称 |
| `max_tokens` | 最大生成 token 数 |
| `temperature` | 生成温度 |

**生成字段**：`id`, `term`, `aliases`, `definitions`, `formula`, `usage`, `applications`, `related_terms`, `sources`, `search_keys` 等

```bash
python dataGen/data_gen.py
```

---

### filter_terms.py

过滤低质量或非数学术语，清理异常数据。

**输入**：`data/processed/chunk/`

**功能**：
- 过滤过短的定义
- 过滤缺少必要字段的术语
- 过滤非数学相关术语

```bash
python dataGen/filter_terms.py
```

---

### clean_failed_ocr.py

清理 OCR 失败或质量极差的分页文件，避免污染后续流程。

**输入**：`data/processed/ocr/`

```bash
python dataGen/clean_failed_ocr.py
```

## 环境配置

在运行前，需在 `.env` 文件中配置 API 密钥：

```
API-KEY=sk-your-deepseek-api-key
```

`config.toml` 中 `api_key_env = "API-KEY"` 指定环境变量名。

## 数据规模（当前）

| 书籍 | 术语数 |
|------|--------|
| 数学分析（第5版）上/下 | ~1,547 |
| 高等代数（第五版） | ~645 |
| 概率论与数理统计 | ~909 |
| **合计** | **~3,102** |
