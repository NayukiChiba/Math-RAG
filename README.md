# MathRag

面向数学术语与概念问答的检索增强生成系统。

MathRag 不是一个只返回“看起来像答案”的问答工具，而是一个把数学教材
语料、检索策略、回答生成、评测分析和可视化界面串起来的完整工程。它
支持从 PDF 教材入库开始，逐步完成术语抽取、检索语料构建、RAG 问答、
检索评测、生成评测和论文图表产出。

## 项目特点

- 面向数学场景：围绕术语、定义、定理和公式相关问答设计
- 检索优先：支持 `BM25`、`BM25+`、`Vector`、`Hybrid`、`Hybrid+`
  等多种策略
- 生成可切换：同时支持本地模型推理和 OpenAI 兼容 API
- 研究友好：内置查询集生成、检索评测、显著性检验和报告生成
- 统一入口：通过 `python main.py` 管理产品线、研究线和 Web UI
- 工程完整：包含测试、文档站点、答辩图表和前后端界面

## 适用场景

MathRag 适合下面几类工作：

- 构建数学教材语料库并完成术语问答
- 对比不同检索策略在数学场景下的效果
- 做 RAG 课程设计、毕业设计或论文实验
- 产出实验报告、可视化图表和答辩材料

## 系统主线

仓库当前可以分为三条主线：

### 1. 产品线

面向日常使用，负责把原始教材变成可问答系统。

典型流程：

```text
ingest -> build-index -> rag
```

主要能力：

- PDF 入库
- OCR 与术语抽取
- 检索语料与索引构建
- 单条或批量 RAG 问答

### 2. 研究线

面向实验与论文工作，负责评测、对比和报告产出。

典型流程：

```text
generate-queries -> build-term-mapping -> eval-retrieval
-> experiments -> eval-generation -> report
```

主要能力：

- 评测查询集构建
- 检索评测与快速评测
- 生成质量评测
- 显著性检验
- 报告与答辩图表生成

### 3. Web UI

面向演示和可视化操作，后端使用 FastAPI，前端使用 Vue 3 +
Element Plus。

主要页面包括：

- `/chat`：流式 RAG 问答
- `/ingest`、`/index`：入库与索引构建
- `/research/*`：研究线命令可视化入口
- `/reports`、`/figures`、`/stats`：报告与图表浏览
- `/config`：`config.toml` 在线编辑
- `/tasks`：长任务进度与日志查看

## 快速开始

### 环境要求

- Python 3.11+
- Node.js 18+（仅 Web UI 前端和文档站点需要）
- 推荐使用虚拟环境或 Conda 环境

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install -e .
```

如果你要跑研究评测，建议补装研究依赖：

```bash
pip install -e .[research]
```

### 2. 检查配置

项目主配置文件为 `config.toml`，常见配置包括：

- `[paths]`：数据与输出目录
- `[ocr]`：OCR 引擎与参数
- `[terms_gen]`：术语结构化生成参数
- `[rag_gen]`：回答生成参数
- `[retrieval]`：检索模型、融合权重和阈值

如果使用 API 模式，请在 `.env` 或系统环境变量中设置
`config.toml` 对应的 `api_key_env` 项。当前默认键名包括：

- `API-KEY-OCR`
- `API-KEY-TERMS`
- `API-KEY-RAG`

如果使用本地模型，请把对应模块的 `engine` 改成 `local`，并设置本地模
型目录。

### 3. 跑通最小产品流程

先把 PDF 放进 `data/raw/`，然后执行：

```bash
python main.py cli ingest data/raw/数学分析.pdf
python main.py cli build-index --rebuild
python main.py cli rag --query "什么是一致收敛？"
```

如果只想看入口帮助：

```bash
python main.py --help
python main.py cli --help
python main.py research --help
python main.py ui --help
```

## 常用命令

### 顶层入口

推荐统一通过 `main.py` 启动：

```bash
python main.py cli ...
python main.py research ...
python main.py ui
```

可编辑安装后，也可以继续使用兼容入口：

```bash
math-rag
math-rag-research
```

### 产品线命令

| 命令 | 说明 |
| --- | --- |
| `python main.py cli ingest <pdf>` | 执行 PDF 入库流水线 |
| `python main.py cli build-index` | 单独构建或重建检索索引 |
| `python main.py cli rag --query "..."` | 运行单条问答 |
| `python main.py cli serve --port 7860` | 启动 Web UI |

常见示例：

```bash
python main.py cli ingest data/raw/数学分析.pdf --skip-generation
python main.py cli build-index --rebuild --skip-bm25
python main.py cli rag --query "什么是柯西列？"
python main.py cli rag --query-file data/evaluation/queries.txt
```

### 研究线命令

| 命令 | 说明 |
| --- | --- |
| `generate-queries` | 生成评测查询集 |
| `build-term-mapping` | 构建术语映射 |
| `eval-retrieval` | 正式检索评测 |
| `quick-eval` | 快速检索评测 |
| `experiments` | 端到端实验 |
| `eval-generation` | 生成质量评测 |
| `eval-generation-comparison` | 生成质量对比 |
| `significance-test` | 统计显著性检验 |
| `report` | 生成最终报告 |
| `full-reports` | 一次性跑完整评测流程 |
| `publish-reports` | 从已有日志发布定稿 |
| `defense-figures` | 生成答辩图表 |
| `add-missing-terms` | 补充缺失术语 |
| `stats` | 语料统计与可视化 |
| `serve` | 启动研究线 Web UI |

常见示例：

```bash
python main.py research eval-retrieval --visualize
python main.py research full-reports --retrieval-only
python main.py research publish-reports --run-id 20260406_164049
python main.py research serve --port 7861
```

## Web UI

### 构建前端

```bash
cd webui/frontend
npm install
npm run build
```

构建完成后会生成 `webui/frontend/dist/`，此时可以直接启动后端：

```bash
python main.py ui
```

默认地址：

```text
http://127.0.0.1:7860
```

### 前端开发模式

终端 A：

```bash
python main.py ui
```

终端 B：

```bash
cd webui/frontend
npm install
npm run dev
```

默认开发地址：

```text
http://localhost:5173
```

### OpenAPI 文档

启动后访问：

```text
http://127.0.0.1:7860/docs
```

## 文档站点

仓库内置 VitePress 文档站点，源码位于 `docs/`。

本地预览：

```bash
cd docs
npm install
npm run docs:dev
```

构建静态文档：

```bash
cd docs
npm run docs:build
npm run docs:preview
```

常看入口：

- `docs/index.md`
- `docs/project.md`
- `docs/guide/`
- `docs/core/`
- `docs/research/`
- `docs/reports/`

## 项目结构

```text
MathRag/
├── main.py                    # 统一启动入口
├── config.toml                # 全局配置
├── pyproject.toml             # 包配置与脚本入口
├── requirements.txt           # 依赖列表
├── src/
│   ├── core/                  # 产品线核心模块
│   │   ├── answerGeneration/  # 提示模板、推理封装、RAG 管线
│   │   ├── dataGen/           # OCR、术语抽取、结构化生成
│   │   ├── retrieval/         # 语料构建、检索器、查询改写
│   │   ├── runners/           # 产品线运行入口
│   │   ├── utils/             # 通用工具
│   │   └── cli/               # 产品线 CLI
│   └── research/              # 研究线模块
│       ├── evaluationData/    # 评测数据构建
│       ├── modelEvaluation/   # 检索评测与生成评测
│       ├── dataStat/          # 数据统计与可视化
│       ├── runners/           # 实验编排
│       └── cli/               # 研究线 CLI
├── reports_generation/        # 报告、快评和答辩图表
├── webui/                     # FastAPI 后端 + Vue 前端
├── docs/                      # VitePress 文档站点
├── tests/                     # 测试
├── data/                      # 原始数据、处理中间数据、评测数据
└── outputs/                   # 问答结果、实验日志、图表与报告
```

## 关键目录与产物

### 输入目录

- `data/raw/`：原始 PDF 教材
- `data/evaluation/`：评测查询、术语映射和金标准数据

### 中间产物

- `data/processed/ocr/`
- `data/processed/terms/`
- `data/processed/chunk/`
- `data/processed/retrieval/`

### 输出目录

- `outputs/rag_results.jsonl`：问答结果
- `outputs/log/`：实验过程日志
- `outputs/reports/`：正式报告和图表
- `outputs/figures/`：答辩或展示图表

## 架构约束

- `research` 和 `reports_generation` 可以依赖 `core`
- `core` 不应反向依赖 `research` 或 `reports_generation`
- 路径和默认值统一从 `config.toml` 与 `src/core/config.py` 读取

## 开发与测试

代码格式与检查：

```bash
ruff check .
ruff format .
```

运行测试：

```bash
pytest
```

如果只想做基础检查，可以先跑 smoke tests：

```bash
pytest tests/smoke
```

## 当前仓库包含什么
- 可直接运行的数学问答产品线
- 面向论文实验的研究线工具链
- 前后端一体化 Web UI
- 文档站点
- 实验报告、图表和测试代码

