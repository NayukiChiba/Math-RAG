# Math-RAG

面向数学术语与概念问答的检索增强生成（RAG）系统，基于本地 Qwen-Math 模型与多路检索策略。

## 项目定位

Math-RAG 的目标不是"只给出答案"，而是"给出可追溯答案"：

1. 从教材语料中检索相关定义、定理与上下文片段。
2. 基于检索证据进行生成。
3. 输出可回溯来源的信息，支持实验复现与对比分析。

核心原则：

1. 检索准确率优先。
2. 实验流程可复现。
3. 工程结构清晰、可维护。

## 文档入口

| 文档             | 路径                                         |
| ---------------- | -------------------------------------------- |
| 项目总文档       | [docs/project.md](docs/project.md)           |
| API 文档首页     | [docs/index.md](docs/index.md)               |
| API 总索引       | [docs/api/index.md](docs/api/index.md)       |
| Runners 模块说明 | [src/core/runners/README.md](src/core/runners/README.md) |

## 快速开始

### 1. 环境准备

```bash
conda create -n MathRag python=3.11
conda activate MathRag
pip install -r requirements.txt
pip install -e .
```

安装完成后可直接使用统一命令：`math-rag`。

### 2. 一条命令跑通入库

```bash
math-rag ingest data/raw/数学分析.pdf
```

该命令默认执行：OCR → 术语抽取 → 结构化生成 → 检索语料/索引构建。

### 3. 一条命令进行问答

```bash
math-rag rag --query "什么是一致收敛？"
```

### 4. 启动交互界面

```bash
math-rag serve --port 7860
```

## 统一 CLI

```bash
math-rag --help
```

支持的子命令：

| 子命令              | 说明                   |
| ------------------- | ---------------------- |
| `ingest`            | PDF 入库流水线         |
| `build-index`       | 重建检索语料与索引     |
| `generate-queries`  | 生成评测查询集         |
| `build-term-mapping`| 构建评测术语映射       |
| `eval-retrieval`    | 正式检索评测           |
| `rag`               | RAG 问答               |
| `experiments`       | 端到端对比实验         |
| `eval-generation`   | 生成质量评测           |
| `report`            | 生成汇总报告           |
| `stats`             | 统计与可视化           |
| `serve`             | 启动 WebUI             |

## 常见命令示例

```bash
# 重建索引
math-rag build-index --rebuild

# 生成评测数据
math-rag generate-queries
math-rag build-term-mapping

# 正式评测
math-rag eval-retrieval --visualize

# 端到端实验
math-rag experiments --limit 10

# 生成质量评测
math-rag eval-generation --bleu --rouge

# 报告生成
math-rag report
```

## 推荐实验流程

```text
build-index → generate-queries → build-term-mapping
  → eval-retrieval → experiments → eval-generation → report
```

报告默认写入：`outputs/log/YYYYMMDD_HHMMSS/`。

## 项目结构

```text
Math-RAG/
├── src/                          # 全部 Python 源码
│   ├── config.py                 # 统一配置入口
│   ├── mathRag.py                # 统一 CLI 入口
│   ├── answerGeneration/         # 提示模板、推理封装、RAG 流程、WebUI
│   ├── dataGen/                  # OCR、术语抽取、结构化生成
│   ├── dataStat/                 # 数据统计与可视化
│   ├── evaluationData/           # 评测数据集生成
│   ├── modelEvaluation/          # 检索评测 / 生成评测 / 快速评测
│   ├── retrieval/                # 语料构建、查询改写、多策略检索器
│   ├── runners/                  # 脚本编排层（实验、报告、工具）
│   └── utils/                    # 通用工具（文件加载、输出管理）
├── api/                          # FastAPI 服务层（v1）
│   ├── middleware/               # 中间件
│   └── v1/                       # 路由、模型、端点
├── tests/                        # 测试
├── data/                         # 数据资产
│   ├── raw/                      # 原始 PDF
│   ├── processed/                # OCR / 术语 / 语料 / 索引
│   ├── evaluation/               # 评测输入数据
│   └── stats/                    # 统计输出
├── docs/                         # 项目文档 + API 文档
├── outputs/                      # 运行输出与日志
├── __init__.py                   # 项目级统一导出
├── config.toml                   # 全局参数配置
├── pyproject.toml                # 构建与工具链配置
└── requirements.txt              # 依赖清单
```

## 核心模块说明

### `src/dataGen`

从教材 PDF 生成结构化术语数据，流程：PDF → OCR → 术语抽取 → 结构化生成 → 过滤清洗。

### `src/retrieval`

语料构建、查询改写与多策略检索器实现，支持：

- **BM25 / BM25+**：经典稀疏检索
- **Vector**：基于 `bge-base-zh-v1.5` 的稠密检索
- **Hybrid / HybridPlus**：稀疏 + 稠密融合（RRF）
- **Reranker**：基于 `bge-reranker-v2-mixed` 的重排序
- **Advanced**：全流程高级检索管线

### `src/answerGeneration`

提示模板管理、多引擎推理封装（本地 / API）、RAG 端到端管线及 Gradio WebUI。

### `src/modelEvaluation`

三类评测路径：

| 模块                | 说明               |
| ------------------- | ------------------ |
| `quickEvalCore/`    | 快速检索评测       |
| `retrievalEval/`    | 正式检索评测       |
| `generationEval/`   | 生成质量评测       |

### `src/runners`

脚本编排与工具集合，包括端到端实验、报告生成、显著性检验、答辩图表等。

### `api/`

基于 FastAPI 的 REST API 服务层，提供 `/v1` 版本的端点与中间件支持。

## 配置说明

项目采用 `config.toml` + `src/config.py` 双层配置：

| 文件                | 职责                       |
| ------------------- | -------------------------- |
| `config.toml`       | 参数数据源（TOML 格式）    |
| `src/config.py`     | Python 统一读取接口与路径常量 |

常见配置段：`[paths]`、`[ocr]`、`[model]`、`[generation]`、`[retrieval]`。

## 开发与代码规范

- 使用 **Ruff** 进行检查与格式化。
- 路径统一通过 `src/config.py` 管理。
- 提交信息遵循 `type(scope): 中文说明` 格式。

```bash
ruff check .
ruff format .
```

可选启用 pre-commit：

```bash
pip install pre-commit
pre-commit install
pre-commit run -a
```