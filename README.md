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
| Runners 模块说明 | [src/core/runners/README.md](src/core/runners/README.md) |

## 快速开始

### 1. 环境准备

```bash
conda create -n MathRag python=3.11
conda activate MathRag
pip install -r requirements.txt
pip install -e .
```

安装完成后通过根目录入口 `python main.py` 统一启动所有功能。

### 2. 一条命令跑通入库

```bash
python main.py cli ingest data/raw/数学分析.pdf
```

该命令默认执行：OCR → 术语抽取 → 结构化生成 → 检索语料/索引构建。

### 3. 一条命令进行问答

```bash
python main.py cli rag --query "什么是一致收敛？"
```

### 4. 启动交互界面

```bash
python main.py ui                          # 产品线 RAG WebUI（端口 7860）
python main.py ui --port 7861 --share      # 指定端口并生成分享链接
python main.py ui --research               # 研究线实验对比 WebUI（端口 7861）
```

### 5. 查看所有可用命令

```bash
python main.py --help          # 顶层帮助（含快速示例）
python main.py cli --help      # 产品线子命令列表
python main.py research --help # 研究线子命令列表
python main.py ui --help       # WebUI 选项
```

## 文档

文档站点使用 [VitePress](https://vitepress.dev) 构建，通过 GitHub Actions 自动发布到 GitHub Pages。

```bash
npm install
npm run docs:dev      # 本地开发预览（http://localhost:5173）
npm run docs:build    # 构建静态站点
npm run docs:preview  # 预览构建产物
```

| 目录 | 说明 |
|------|------|
| `docs/guide/` | 快速开始、安装、CLI 用法、配置 |
| `docs/core/` | 产品线核心模块文档 |
| `docs/research/` | 研究线评测与实验文档 |
| `docs/reports/` | 报告与答辩图表生成 |

## 产品线（`python main.py cli`）

| 子命令        | 说明                       |
| ------------- | -------------------------- |
| `ingest`      | PDF 入库流水线             |
| `build-index` | 重建检索语料与索引         |
| `rag`         | RAG 问答（单条或批量）     |
| `serve`       | 启动产品 WebUI（同 `ui`）  |

```bash
python main.py cli ingest data/raw/数学分析.pdf
python main.py cli build-index --rebuild
python main.py cli rag --query "什么是一致收敛？"
```

## 研究线（`python main.py research`）

论文实验、评测、报告生成等均通过研究线入口调用。

| 子命令                        | 说明                               |
| ----------------------------- | ---------------------------------- |
| `eval-retrieval`              | 正式检索评测（多方法对比）         |
| `full-reports`                | 全量评测总控（日志→log，定稿→reports） |
| `publish-reports`             | 从日志发布定稿到 outputs/reports/  |
| `experiments`                 | 端到端对比实验                     |
| `eval-generation`             | 生成质量评测                       |
| `eval-generation-comparison`  | 生成质量对比                       |
| `significance-test`           | 显著性检验                         |
| `report`                      | 生成汇总报告                       |
| `quick-eval`                  | 快速检索评测                       |
| `defense-figures`             | 生成答辩图表                       |
| `add-missing-terms`           | 补充缺失术语                       |
| `stats`                       | 统计与可视化                       |
| `serve`                       | 研究线实验 WebUI（同 `ui --research`） |

```bash
python main.py research eval-retrieval --visualize
python main.py research full-reports --retrieval-only
python main.py research publish-reports --run-id 20260406_164049
```

> **兼容说明**：若已安装本项目（`pip install -e .`），原有命令 `math-rag` 与 `math-rag-research` 仍然有效，但推荐统一使用 `python main.py`。

## 推荐流程

**产品线（日常使用）：**

```text
ingest → build-index → rag / serve
```

**研究线（论文实验）：**

```text
build-index → generate-queries → build-term-mapping
  → eval-retrieval → experiments → eval-generation → report
```

报告默认写入：`outputs/log/YYYYMMDD_HHMMSS/`。

## 项目结构

```text
Math-RAG/
├── src/
│   ├── core/                     # 产品线（RAG 核心）
│   │   ├── config.py             # 统一配置入口
│   │   ├── mathRag.py            # math-rag CLI 入口
│   │   ├── cli/                  # 产品线子命令注册
│   │   ├── answerGeneration/     # 提示模板、推理封装、RAG 管线、WebUI
│   │   ├── dataGen/              # OCR、术语抽取、结构化生成
│   │   ├── retrieval/            # 语料构建、查询改写、多策略检索器
│   │   ├── runners/              # 仅 RAG 问答编排
│   │   └── utils/                # 通用工具（文件加载、输出管理）
│   ├── research/                 # 研究线（论文实验/评测）
│   │   ├── researchMain.py       # math-rag-research CLI 入口
│   │   ├── cli/                  # 研究线子命令注册
│   │   ├── evaluationData/       # 评测数据集生成
│   │   ├── modelEvaluation/      # 检索评测 / 生成评测
│   │   ├── dataStat/             # 数据统计与可视化
│   │   └── runners/              # 实验编排（experiments, evaluation, tools）
│   └── (api/ 已移除)
├── main.py                   # 统一启动入口（推荐使用）
├── reports_generation/       # 报告、快评、答辩图表
├── tests/                    # 测试
├── data/                     # 数据资产（产品与研究共用）
├── docs/                     # 项目文档
├── outputs/                  # 运行输出与日志
├── config.toml               # 全局参数配置
├── pyproject.toml            # 构建与工具链配置
└── requirements.txt          # 依赖清单
```

## 架构约束

- **导入方向**：`research` / `reports_generation` 可 `import core.*`；**禁止** `core` 反向依赖 `research` 或 `reports_generation`。
- **安装**：`pip install .` 即可运行产品线；`pip install .[research]` 额外安装论文研究所需依赖。

## 产品线核心模块

### `src/core/dataGen`

从教材 PDF 生成结构化术语数据：PDF → OCR → 术语抽取 → 结构化生成 → 过滤清洗。

### `src/core/retrieval`

语料构建、查询改写与多策略检索器：BM25 / BM25+ / Vector / Hybrid / HybridPlus / Reranker / Advanced。

### `src/core/answerGeneration`

提示模板管理、多引擎推理封装（本地 / API）、RAG 端到端管线及 Gradio WebUI。

## 研究线核心模块

### `src/research/modelEvaluation`

检索评测（`retrievalEval/`）、生成质量评测（`generationEval/`）。

### `src/research/evaluationData`

评测查询集自动生成。

### `src/research/dataStat`

术语数据统计与论文级可视化图表。

### `reports_generation/`

快速检索评测、答辩图表、完整评测报告（Markdown + PDF/PNG 图表）。

## 配置说明

项目采用 `config.toml` + `src/core/config.py` 双层配置：

| 文件                    | 职责                           |
| ----------------------- | ------------------------------ |
| `config.toml`           | 参数数据源（TOML 格式）        |
| `src/core/config.py`    | Python 统一读取接口与路径常量 |

常见配置段：`[paths]`、`[ocr]`、`[model]`、`[generation]`、`[retrieval]`。

## 开发与代码规范

- 使用 **Ruff** 进行检查与格式化。
- 路径统一通过 `src/core/config.py` 管理。
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