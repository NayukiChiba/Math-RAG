# Math-RAG

毕业论文：一个基于 Qwen-Math 的数学名词 RAG 系统

## 项目概述

本项目构建面向数学名词的高精度检索增强生成（RAG）系统，覆盖数学分析、高等代数、概率论三大学科。

**核心目标**：
- 检索准确率优先
- 可复现、可对比的实验流程
- 代码清晰易读

**基础模型**：
- Qwen2.5-Math-1.5B-Instruct（本地运行）
- Qwen2.5-Math-7B-Instruct（可选）

## 项目结构

```
Math-RAG/
├── config.py              # 统一路径配置
├── config.toml            # 配置文件
├── AGENTS.md              # 开发规范与协作指南
├── dataGen/               # 数据生成模块
│   ├── pix2text_ocr.py   # OCR 处理
│   ├── extract_terms_from_ocr.py  # 术语提取
│   ├── data_gen.py       # JSON 数据生成
│   └── filter_terms.py   # 术语过滤
├── dataStat/              # 数据统计模块
│   ├── chunkStatistics.py  # 术语数据统计与可视化
│   └── README.md         # 模块使用说明
├── retrieval/             # 检索模块
│   ├── buildCorpus.py    # 构建检索语料
│   ├── retrievalBM25.py  # BM25 检索
│   ├── retrievalVector.py  # 向量检索
│   ├── retrievalHybrid.py  # 混合检索
│   └── README.md         # 模块使用说明
├── evaluation/            # 评测模块
│   ├── evalRetrieval.py  # 检索评测
│   └── README.md         # 模块使用说明
├── data/                  # 数据目录
│   ├── raw/              # 原始 PDF 教材
│   ├── processed/        # 处理后数据
│   │   ├── ocr/         # OCR 结果
│   │   ├── terms/       # 术语映射
│   │   ├── chunk/       # 术语级 JSON
│   │   └── retrieval/   # 检索语料与索引
│   ├── evaluation/       # 评测数据集
│   │   └── queries.jsonl  # 评测查询
│   └── stats/           # 统计报告与可视化
├── docs/                 # 文档
│   ├── plan.md          # 项目规划
│   └── task.md          # 当前任务计划
└── outputs/             # 实验输出
    ├── reports/         # 评测报告
    └── bm25/            # BM25 索引
```

## 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 数据处理

```bash
# OCR 处理（将 PDF 放入 data/raw/）
python dataGen/pix2text_ocr.py

# 提取术语
python dataGen/extract_terms_from_ocr.py

# 生成术语 JSON
python dataGen/data_gen.py
```

### 3. 数据统计

```bash
# 生成统计报告和可视化图表
python dataStat/chunkStatistics.py

# 输出：
# - data/stats/chunkStatistics.json
# - data/stats/visualizations/*.png
```

### 4. 构建检索系统

```bash
# 构建检索语料
python retrieval/buildCorpus.py

# BM25 检索（自动构建索引）
python retrieval/retrievalBM25.py --query "泰勒公式" --topk 5

# 向量检索（需要安装 faiss 和 sentence-transformers）
python retrieval/retrievalVector.py --query "泰勒公式" --topk 5

# 混合检索
python retrieval/retrievalHybrid.py --query "泰勒公式" --topk 5 --strategy rrf
```

### 5. 运行评测

```bash
# 评测 BM25（无需额外依赖）
python evaluation/evalRetrieval.py --methods bm25

# 评测所有方法（需要 faiss 和 sentence-transformers）
python evaluation/evalRetrieval.py --visualize

# 输出：
# - outputs/reports/retrieval_metrics.json
# - outputs/reports/retrieval_comparison.png（可选）
```

## 主要模块

### dataGen - 数据生成

负责教材 OCR、术语提取、JSON 数据生成。

**功能**：
- PDF 转图片 OCR
- 数学术语识别与提取
- 结构化 JSON 生成
- 术语-页码映射

**详见**：[dataGen/README.md](dataGen/README.md)（待补充）

### dataStat - 数据统计

负责数据质量评估、统计分析、可视化。

**功能**：
- 字段覆盖率统计
- 长度分布分析
- 学科分布统计
- 重复术语识别
- 自动生成可视化图表（6张高清图表）

**详见**：[dataStat/README.md](dataStat/README.md)

### retrieval - 检索模块

实现多种检索方法用于术语相似度匹配。

**功能**：
- **语料构建**：从 chunk JSON 生成检索语料（corpus.jsonl）
- **BM25 检索**：基于词频的稀疏检索（rank-bm25）
- **向量检索**：基于语义的密集检索（sentence-transformers + FAISS）
- **混合检索**：加权融合 + RRF（Reciprocal Rank Fusion）

**使用示例**：
```bash
python retrieval/retrievalBM25.py --query "泰勒公式" --topk 5
python retrieval/retrievalHybrid.py --query "泰勒公式" --strategy rrf
```

**详见**：[retrieval/README.md](retrieval/README.md)

### evaluation - 评测模块

评估不同检索方法的性能，生成对比报告。

**评测指标**：
- Recall@K（K=1,3,5,10）：召回率
- MRR：Mean Reciprocal Rank
- nDCG@K：Normalized Discounted Cumulative Gain
- MAP：Mean Average Precision

**评测数据集**：35 条手工标注查询（data/evaluation/queries.jsonl）

**使用示例**：
```bash
python evaluation/evalRetrieval.py --methods bm25 vector hybrid-rrf --visualize
```

**详见**：[evaluation/README.md](evaluation/README.md)

## 当前进度

- ✅ Plan-1：任务定义与评测标准
- ✅ Plan-3：教材 OCR + LLM 构建数学名词数据
  - 已处理 4 本教材，生成 3,102 个术语
- ✅ Plan-2：数据准备与检索系统
  - ✅ Task-1：数据核验与统计
  - ✅ Task-2：构建检索语料（3,102 条文档）
  - ✅ Task-3：BM25 基线检索
  - ✅ Task-4：向量检索基线（FAISS + Sentence Transformers）
  - ✅ Task-5：混合检索（加权融合 + RRF）
  - ✅ Task-6：评测集与指标（35 条查询，4 种指标）
- 🔄 Plan-4：RAG 系统集成（待开始）

**详见**：[docs/plan.md](docs/plan.md) 和 [docs/task.md](docs/task.md)

## 开发规范

详见 [AGENTS.md](AGENTS.md)，核心原则：

- ✅ 代码清晰易读优先（显式优于隐式）
- ✅ 实验可复现
- ✅ 使用驼峰命名法（文件名、函数名）
- ✅ 路径统一通过 `config.py` 管理
- ✅ 中文注释
- ✅ Git commit 规范（英文 type + 中文描述）

## Code Style

本仓库使用 [Ruff](https://github.com/astral-sh/ruff) 统一 Python 代码风格：

- Lint: `ruff check .`
- 自动修复: `ruff check . --fix`
- Format: `ruff format .`
- CI: push / PR 时会自动运行检查

### 可选：pre-commit

```bash
# 安装
pip install pre-commit

# 安装 hooks
pre-commit install

# 手动执行
pre-commit run -a
```

## 数据统计概览

**当前数据规模**（截至 2026-02-14）：
- 总术语数：3,102 个
- 学科分布：
  - 数学分析：1,547 (49.9%)
  - 概率论与数理统计：909 (29.3%)
  - 高等代数：645 (20.8%)
- 平均每术语：3.0 个定义
- 字段覆盖率：核心字段 95%+ 覆盖率

**检索系统指标**（BM25 基线，35 条查询）：
- Recall@10：45.24%
- MRR：0.3225
- nDCG@10：0.3754
- 平均查询时间：3.8ms

**详细统计**：
- 数据质量：`data/stats/chunkStatistics.json` + 可视化图表
- 检索评测：`outputs/reports/retrieval_metrics.json`