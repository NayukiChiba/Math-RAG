# Math-RAG

面向数学术语与概念问答的检索增强生成（RAG）系统，基于本地 Qwen-Math 模型与多路检索策略。

## 项目定位

Math-RAG 的目标不是“只给出答案”，而是“给出可追溯答案”：

1. 从教材语料中检索相关定义、定理与上下文片段。
2. 基于检索证据进行生成。
3. 输出可回溯来源的信息，支持实验复现与对比分析。

核心原则：

1. 检索准确率优先。
2. 实验流程可复现。
3. 工程结构清晰、可维护。

## 文档入口

1. 项目总文档: [docs/project.md](docs/project.md)
2. API 文档首页（GitHub Pages 入口）: [docs/index.md](docs/index.md)
3. API 总索引: [docs/api/index.md](docs/api/index.md)

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

该命令默认执行：OCR -> 术语抽取 -> 结构化生成 -> 检索语料/索引构建。

### 3. 一条命令进行问答

```bash
math-rag rag --query "什么是一致收敛？"
```

### 4. 启动交互界面

```bash
math-rag serve --port 7860
```

## 统一 CLI（建议主入口）

```bash
math-rag --help
```

当前统一支持的子命令：

1. `ingest`：PDF 入库流水线。
2. `build-index`：重建检索语料与索引。
3. `generate-queries`：生成评测查询集。
4. `build-term-mapping`：构建评测术语映射。
5. `quick-eval`：快速检索评测。
6. `eval-retrieval`：正式检索评测。
7. `rag`：RAG 问答。
8. `experiments`：端到端对比实验。
9. `eval-generation`：生成质量评测。
10. `report`：生成汇总报告。
11. `stats`：统计与可视化。
12. `serve`：启动 WebUI。

## 常见命令示例

```bash
# 重建索引
math-rag build-index --rebuild

# 生成评测数据
math-rag generate-queries
math-rag build-term-mapping

# 快速评测 / 正式评测
math-rag quick-eval --all-queries
math-rag eval-retrieval --visualize

# 端到端实验
math-rag experiments --limit 10

# 生成质量评测
math-rag eval-generation --bleu --rouge

# 报告生成
math-rag report
```

## 推荐实验流程

推荐顺序如下：

1. `math-rag build-index`
2. `math-rag generate-queries`
3. `math-rag build-term-mapping`
4. `math-rag quick-eval --all-queries`
5. `math-rag eval-retrieval --visualize`
6. `math-rag experiments`
7. `math-rag eval-generation`
8. `math-rag report`

报告默认写入：`outputs/reports/YYYYMMDD_HHMMSS/`。

## 项目结构（精简版）

```text
Math-RAG/
├── mathRag.py                # 统一 CLI 入口
├── config.toml               # 全局参数
├── config.py                 # Python 配置访问层
├── dataGen/                  # OCR、术语抽取、结构化生成
├── dataStat/                 # 统计与可视化
├── retrieval/                # 检索构建、检索器、查询改写
├── answerGeneration/         # 提示模板、推理封装、RAG 流程、WebUI
├── evaluationData/           # 评测输入数据生成
├── modelEvaluation/          # 检索评测、生成评测、快速评测
├── scripts/                  # pipelines/experiments/evaluation/tools
├── tests/                    # 测试与校验
├── docs/                     # 项目文档 + API 文档
├── data/                     # raw/processed/evaluation/stats
└── outputs/                  # 运行输出与报告
```

## 模块说明

### dataGen

用于从教材 PDF 生成结构化术语数据：

1. `pix2text_ocr.py`
2. `extract_terms_from_ocr.py`
3. `data_gen.py`
4. `filter_terms.py`
5. `clean_failed_ocr.py`

### retrieval

包含语料构建、查询改写与检索器实现。支持 BM25、BM25+、Vector、Hybrid、Reranker、Advanced 等策略。

### answerGeneration

包含提示模板、Qwen 推理封装、RAG 端到端流程及 WebUI。

### modelEvaluation

包含 `quickEval`、`evalRetrieval`、`evalGeneration` 三类评测路径。

## 配置说明

项目采用 `config.toml + config.py` 双层配置：

1. `config.toml`：配置数据源。
2. `config.py`：统一读取接口与路径常量。

常见配置段：

1. `[paths]`
2. `[ocr]`
3. `[model]`
4. `[generation]`
5. `[retrieval]`

## 开发与代码规范

1. 使用 Ruff 进行检查与格式化。
2. 路径统一通过配置层管理。
3. 提交信息遵循“英文 type + 中文说明”。

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

## 相关文档

1. [项目主文档](docs/project.md)
2. [项目规划](docs/plan.md)
3. [任务跟踪](docs/task.md)
4. [检索模块详细说明](retrieval/README.md)
5. [评测模块详细说明](modelEvaluation/README.md)