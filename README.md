# Math-RAG

基于 Qwen-Math 的数学名词检索增强生成（RAG）系统。

## 项目概述

本项目构建面向数学名词的高精度 RAG 系统，覆盖数学分析、高等代数、概率论与数理统计三大学科。

**核心目标**：
- 检索准确率优先
- 可复现、可对比的实验流程
- 代码清晰易读

**基础模型**：Qwen2.5-Math 系列本地模型（具体目录由 `config.toml [generation].qwen_model_dir` 指定）

## 项目结构

```
Math-RAG/
├── mathRag.py                 # 统一 CLI 入口（math-rag）
├── config.py                  # Python 配置接口
├── config.toml                # TOML 配置文件（全局参数）
├── requirements.txt           # 项目依赖
│
├── dataGen/                   # 数据生成模块
│   ├── pix2text_ocr.py       # PDF → OCR Markdown
│   ├── extract_terms_from_ocr.py  # OCR → 术语映射
│   ├── data_gen.py            # 术语 → 结构化 JSON
│   ├── filter_terms.py        # 术语过滤与清洗
│   └── clean_failed_ocr.py    # 清理 OCR 失败数据
│
├── dataStat/                  # 数据统计模块
│   └── chunkStatistics.py     # 术语统计与可视化
│
├── retrieval/                 # 检索模块
│   ├── buildCorpus.py         # 构建检索语料
│   ├── retrievers.py          # 全部检索器（7 种）
│   └── queryRewrite.py        # 查询改写与同义词扩展
│
├── answerGeneration/          # 回答生成模块
│   ├── promptTemplates.py     # RAG 提示模板
│   ├── qwenInference.py       # Qwen 推理封装
│   ├── ragPipeline.py         # 端到端 RAG 流程
│   └── webui.py               # Gradio WebUI
│
├── evaluationData/            # 评测数据构建模块
│   ├── generateQueries.py     # 评测查询生成
│   └── queryGeneration/       # 查询生成子模块
│
├── modelEvaluation/           # 评测模块
│   ├── evalRetrieval.py       # 检索评测
│   ├── evalGeneration.py      # 生成质量评测
│   └── quickEval.py           # 快速检索评测
│
├── scripts/                   # 脚本入口
│   ├── runRag.py              # RAG 问答
│   ├── runExperiments.py      # 对比实验
│   ├── experimentWebUI.py     # 实验 WebUI
│   └── buildEvalTermMapping.py  # 评测术语映射构建
│
├── tests/                     # 测试
│   └── testRetrievalWeights.py  # 检索权重测试
│
├── data/                      # 数据目录
│   ├── raw/                   # 原始 PDF 教材
│   ├── processed/             # 处理后数据
│   │   ├── ocr/              # OCR 结果（分页 Markdown）
│   │   ├── terms/            # 术语映射
│   │   ├── chunk/            # 术语级 JSON
│   │   └── retrieval/        # 检索语料与索引
│   ├── evaluation/            # 评测数据集
│   └── stats/                 # 统计报告与可视化
│
├── docs/                      # 文档
│   ├── plan.md               # 项目规划与路线图
│   └── task.md               # 任务进度跟踪
│
└── outputs/                   # 实验输出
    └── reports/              # 评测报告（按运行时间自动分目录）
        └── YYYYMMDD_HHMMSS/ # 每次运行的报告子目录
```

## 快速开始

### 1. 环境配置

```bash
conda create -n MathRag python=3.11
conda activate MathRag
pip install -r requirements.txt
pip install -e .
```

安装后可直接使用统一命令：`math-rag`。

### 2. 统一 CLI

Issue #51 对应的主工作流现已统一到一个入口下：

```bash
# PDF 入库：OCR → 术语抽取 → 结构化生成 → 语料/索引构建
math-rag ingest data/raw/数学分析.pdf

# 重建检索语料与索引
math-rag build-index

# 正式检索评测
math-rag eval-retrieval --visualize

# 生成质量评测
math-rag eval-generation --results outputs/rag_results.jsonl

# 生成论文/汇总报告
math-rag report

# 启动 WebUI
math-rag serve --port 7860
```

补充命令：

```bash
# 生成评测查询集与术语映射
math-rag generate-queries
math-rag build-term-mapping

# 快速检索评测 / RAG 问答 / 端到端实验
math-rag quick-eval --all-queries
math-rag rag --query "什么是一致收敛？"
math-rag experiments --limit 10

# 数据统计
math-rag stats
```

### 3. 数据处理流程

```bash
# 一键入库推荐走统一入口
math-rag ingest data/raw/数学分析.pdf

# 如需拆分执行，保留原模块脚本调用方式
python dataGen/pix2text_ocr.py
python dataGen/extract_terms_from_ocr.py
python dataGen/data_gen.py
math-rag stats
```

### 4. 检索系统

```bash
# 构建检索语料与索引
math-rag build-index

# 生成评测查询集
math-rag generate-queries

# 构建术语映射（评测用）
math-rag build-term-mapping
```

### 5. 评测与实验

详见下方 [评测与实验指南](#评测与实验指南) 章节。

## 评测与实验指南

项目包含 4 套评测体系，按以下顺序执行可完成从检索调优到论文出图的全流程。

> 每次运行的报告会自动保存到 `outputs/reports/YYYYMMDD_HHMMSS/` 时间戳子目录下，不会覆盖历史结果。

### 推荐执行顺序

| 步骤 | 脚本 | 目的 | 是否需要 Qwen 模型 | 预估耗时 |
|------|------|------|:------------------:|----------|
| 1 | `modelEvaluation/quickEval.py` | 快速对比 20+ 种检索策略，找最优配置 | 否 | 几分钟 |
| 2 | `modelEvaluation/evalRetrieval.py` | 正式检索评测，出论文数据 + 图表 | 否 | 几分钟 |
| 3 | `scripts/runExperiments.py` | 端到端对比实验（RAG vs 无检索） | **是** | 较长 |
| 4 | `modelEvaluation/evalGeneration.py` | 生成质量细粒度分析 | 否（依赖步骤 3 输出） | 几分钟 |

### 步骤 1：快速检索评测（策略选型）

快速对比多种检索策略的 Recall@K / MRR，确定最优检索方法及参数。

```bash
# 快速抽样（20 条查询，调试用）
math-rag quick-eval

# 全量评测（论文数据用这个）
math-rag quick-eval --all-queries

# 只测指定方法
math-rag quick-eval --methods bm25plus hybrid_rrf advanced --all-queries
```

### 步骤 2：正式检索评测（出论文表格和图）

在全量查询集上计算 Recall@K、MRR、nDCG@K、MAP 四个标准指标，可生成对比图表。

```bash
# 全量评测 + 生成对比图表
math-rag eval-retrieval --visualize

# 只测指定方法
math-rag eval-retrieval --methods bm25plus vector hybrid-plus-weighted hybrid-plus-rrf --visualize
```

### 步骤 3：端到端对比实验（核心消融实验）

同时测检索 + 生成，运行完整 RAG pipeline。对比 norag / bm25 / vector / hybrid / hybrid-rrf 五组，计算检索指标和生成指标（术语命中率、来源引用率）。

```bash
# 运行所有实验组（⚠ 需要 Qwen 模型，耗时较长）
math-rag experiments

# 指定实验组
math-rag experiments --groups norag bm25 vector hybrid hybrid-rrf

# 限制查询数量（调试用）
math-rag experiments --limit 10
```

### 步骤 4：生成质量评测（事后分析）

对步骤 3 产出的 RAG 问答结果做生成质量细粒度评测。

```bash
# 基本评测
math-rag eval-generation

# 加 BLEU + ROUGE 分数
math-rag eval-generation --bleu --rouge
```

### 补充：检索权重测试 / WebUI

```bash
# 检索权重调参测试
python tests/testRetrievalWeights.py

# RAG 问答界面
math-rag serve

# 对比实验界面
math-rag serve --target experiment-webui
```

## 主要模块

### dataGen - 数据生成

从 PDF 教材生成结构化数学术语数据。

| 脚本 | 功能 |
|------|------|
| `pix2text_ocr.py` | PDF 转图片，Pix2Text OCR，输出分页 Markdown |
| `extract_terms_from_ocr.py` | 从 OCR 结果提取术语-页码映射 |
| `data_gen.py` | 调用 LLM 生成术语定义 JSON（DeepSeek API） |
| `filter_terms.py` | 术语过滤与质量清洗 |
| `clean_failed_ocr.py` | 清理 OCR 失败的数据 |

### dataStat - 数据统计

| 脚本 | 功能 |
|------|------|
| `chunkStatistics.py` | 字段覆盖率、长度分布、学科分布统计 + 6 张可视化图表 |

### retrieval - 检索模块

7 种检索器统一实现在 `retrievers.py` 中：

| 检索器 | 说明 | 依赖 |
|--------|------|------|
| `BM25Retriever` | BM25 稀疏检索基线 | `rank_bm25` |
| `BM25PlusRetriever` | BM25 + 查询扩展 + 混合分词 | `rank_bm25` |
| `VectorRetriever` | 语义向量检索（FAISS） | `sentence-transformers`, `faiss` |
| `HybridRetriever` | BM25 + 向量加权融合 / RRF | 同上 |
| `HybridPlusRetriever` | 改进混合检索（百分位归一化、自适应权重） | 同上 |
| `RerankerRetriever` | 两阶段检索 + Cross-Encoder 重排序 | `sentence-transformers` |
| `AdvancedRetriever` | 多路召回 + 查询改写 + 重排序 | 同上 |

`queryRewrite.py`：查询改写模块，包含 144 条数学同义词映射（覆盖全部 102 条评测查询）。

**详见**：[retrieval/README.md](retrieval/README.md)

### modelEvaluation - 评测模块

| 脚本 | 功能 |
|------|------|
| `evalRetrieval.py` | 检索评测（Recall@K、MRR、nDCG@K、MAP） |
| `evalGeneration.py` | 生成质量评测（术语命中率、来源引用率） |
| `quickEval.py` | 快速检索评测（多方法对比） |

**详见**：[modelEvaluation/README.md](modelEvaluation/README.md)

### evaluationData - 评测数据构建模块

| 脚本 | 功能 |
|------|------|
| `generateQueries.py` | 从术语库自动生成评测查询集 |

**详见**：[evaluationData/generateQueries.py](evaluationData/generateQueries.py)

### answerGeneration - 生成模块

| 脚本 | 功能 |
|------|------|
| `promptTemplates.py` | RAG 提示模板（f-string + Jinja2） |
| `qwenInference.py` | Qwen2.5-Math 本地推理封装 |
| `ragPipeline.py` | 端到端 RAG 流程（查询 → 检索 → 生成） |
| `webui.py` | Gradio 交互界面 |

### scripts - 脚本入口

| 脚本 | 功能 |
|------|------|
| `runRag.py` | RAG 问答命令行入口 |
| `runExperiments.py` | 四组对比实验（norag/bm25/vector/hybrid） |
| `experimentWebUI.py` | 实验配置与可视化 WebUI |
| `buildEvalTermMapping.py` | 构建评测术语映射 |

## 配置说明

项目使用 `config.toml` + `config.py` 双层配置：

- **`config.toml`**：所有参数的数据源，包含 `[paths]`、`[ocr]`、`[model]`、`[generation]`、`[retrieval]` 五个配置段
- **`config.py`**：Python 接口，提供路径常量和配置读取函数

主要配置函数：

| 函数 | 说明 |
|------|------|
| `get_ocr_config()` | OCR 相关参数 |
| `getGenerationConfig()` | 生成层参数（temperature、max_new_tokens 等） |
| `getRetrievalConfig()` | 检索参数（权重、模型、RRF k 值等） |
| `getReportsDir()` | 当前运行的报告输出目录（`outputs/reports/YYYYMMDD_HHMMSS/`） |

路径常量：`PROJECT_ROOT`、`RAW_DIR`、`PROCESSED_DIR`、`OCR_DIR`、`TERMS_DIR`、`CHUNK_DIR`、`EVALUATION_DIR`、`QWEN_MODEL_DIR`

## 数据概览

| 指标 | 数值 |
|------|------|
| 总术语数 | 3,102 |
| 数学分析 | 1,547（49.9%） |
| 概率论与数理统计 | 909（29.3%） |
| 高等代数 | 645（20.8%） |
| 平均定义数/术语 | 3.0 |
| 核心字段覆盖率 | 95%+ |
| 评测查询集 | 105 条（快速）/ 3,102 条（全量） |

## 开发规范

- 驼峰命名法（文件名、函数名、变量名）
- 路径统一通过 `config.py` 管理
- 中文注释
- Git commit 规范：英文 type + 中文描述
- 代码风格：[Ruff](https://github.com/astral-sh/ruff)（`ruff check .` / `ruff format .`）
- Pre-commit hooks：ruff + pipreqs

```bash
pip install pre-commit
pre-commit install
pre-commit run -a
```

## 相关文档

- [项目规划与路线图](docs/plan.md)
- [任务进度跟踪](docs/task.md)
- [检索模块详细文档](retrieval/README.md)
- [评测模块详细文档](modelEvaluation/README.md)