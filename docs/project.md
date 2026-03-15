# Math-RAG 项目文档

最后更新：2026-03-15

## 1. 项目简介

Math-RAG 是一个面向数学术语问答的检索增强生成系统。

目标是让回答不仅“能答”，而且“答得有依据”：
- 先从教材语料检索相关术语与定义。
- 再基于检索证据进行生成。
- 最终输出包含术语、答案、来源等可追溯信息。

## 2. 技术栈

- Python 3.11+
- 本地模型推理：Transformers + PyTorch
- 检索：rank-bm25, sentence-transformers, FAISS
- 评测与可视化：numpy, matplotlib, rouge-score, nltk
- Web 界面：gradio

## 3. 目录与文件作用

### 3.1 根目录关键文件

- mathRag.py
  - 统一 CLI 入口。
  - 负责把子命令路由到各模块，实现一套命令驱动全流程。

- config.toml
  - 全局配置文件。
  - 管理路径、OCR 参数、模型参数、检索参数、生成参数。

- config.py
  - Python 配置访问层。
  - 提供配置读取与默认值处理，供各模块统一调用。

- README.md
  - 仓库入口说明。
  - 提供快速开始和高频命令示例。

- requirements.txt
  - 运行依赖清单。

- pyproject.toml
  - 包管理与安装配置。
  - 支持以可编辑模式安装项目命令。

### 3.2 业务模块目录

- dataGen
  - 数据生成链路。
  - 包含 OCR、术语抽取、术语清洗、结构化数据生成等脚本。

- dataStat
  - 数据统计与可视化。
  - 产出术语覆盖、字段质量等统计结果。

- retrieval
  - 检索核心模块。
  - 包含语料构建、检索器实现、查询改写、重排与融合逻辑。

- answerGeneration
  - 生成核心模块。
  - 包含提示模板、模型推理封装、RAG 流程、WebUI。

- evaluationData
  - 评测数据构建模块。
  - 负责生成评测查询等评测输入。

- modelEvaluation
  - 评测模块。
  - 包含快速评测、正式检索评测、生成质量评测。

- scripts
  - 可执行脚本入口集合。
  - 按 pipelines、experiments、evaluation、tools 分组，保留根目录快捷入口。

- tests
  - 测试与验证脚本。

- docs
  - 文档目录。
  - 包含项目规划、任务进度、项目主文档。

### 3.3 数据与输出目录

- data/raw
  - 原始 PDF 教材输入。

- data/processed
  - 处理后数据，包括 OCR、术语、chunk、检索语料与索引。

- data/evaluation
  - 评测输入数据，如 queries、term mapping、golden set。

- data/stats
  - 数据统计结果与图表。

- outputs
  - 运行输出目录。
  - 包含 RAG 结果、评测报告、图表、日志等。

### 3.4 scripts 目录关键文件

- scripts/runRag.py
  - RAG 命令行入口，用于单条或批量问答。

- scripts/runExperiments.py
  - 端到端对比实验入口。

- scripts/experimentWebUI.py
  - 实验可视化界面入口。

- scripts/buildTermMapping.py
  - 评测术语映射构建入口。

- scripts/evalGenerationComparison.py
  - 生成评测结果对比脚本。

- scripts/significanceTest.py
  - 统计显著性检验脚本。

- scripts/generateReport.py
  - 报告生成脚本，聚合实验结果。

- scripts/addMissingTerms.py
  - 术语补全工具脚本。

## 4. 统一 CLI 命令与使用方法

本项目提供统一命令行入口 `mathRag.py`，支持系统所有核心功能。你可以使用 `python mathRag.py <command> [options]` 或以可编辑模式安装后直接使用 `math-rag <command> [options]` 运行。

可以通过 `python mathRag.py --help` 查看所有支持的子命令。

### 4.1 数据入库：ingest
执行端到端的 PDF 入库流水线。
- **功能流程**：文档 OCR -> 提取数学术语 -> 结构化切分数据生成 -> 向量和 BM25 索引构建。
- **常用命令**：
  ```bash
  math-rag ingest my_book.pdf
  ```
- **核心参数**：
  - `pdf`：(必需) 待处理的 PDF 路径或存放于 `data/raw/` 目录中的文件名。
  - `--ocr-start-page`, `--extract-start-page`, `--generate-start-page`：多阶段调试用，只从指定页开始。
  - `--skip-generation`, `--skip-index`：用于跳过数据结构化生成阶段或索引构建阶段。
  - `--rebuild-index`：强制清除并重建检索索引。
  - `--vector-model`：指定特定的 Embedding 模型。
  - `--batch-size`：索引阶段切批送入模型的批次大小。

### 4.2 构建索引：build-index
单独执行检索语料切分和所有特征引擎索引加载。
- **功能描述**：用于你在微调了 OCR 或者重度修改 `data/processed` 的预处理文件后，单独拉起重构本地 FAISS 向量库和序列化倒排索引的任务。
- **常用命令**：
  ```bash
  math-rag build-index --rebuild
  ```
- **核心参数**：
  - `--rebuild`：强行覆盖老索引并重抽提全部语料。
  - `--skip-bm25`, `--skip-bm25plus`, `--skip-vector`：可选择性地跳过某项专门子查找分支特征构建。

### 4.3 交互查询与推理：rag
拉起带有混合搜索+大模型处理管道的答疑工作流，支持单条询问或多文件批量跑。
- **功能描述**：加载检索组件并在查找到教材源文参考结果后送入大模型，产生完整的包含“解释”和“出处链接”的回复。
- **常用命令**：
  ```bash
  math-rag rag --query "什么是洛必达法则？" --strategy hybrid
  ```
- **核心参数**：
  - `-q, --query`：你要问的纯文问题。
  - `-f, --query-file`：使用 .txt/.jsonl 清单批量询问。
  - `-s, --strategy`：指定找回方式，可设 `bm25`，`vector`，或结合两者优势的 `hybrid`（默认）。
  - `-k, --topk`：大模型从检索系统中获取的前几条节点上下文来协助回答，默认 5 条。
  - `--alpha`, `--beta`：如果使用了混合搜索，控制字面权重与向量权重的比例参数。
  - `-t, --temperature`：调整推断模型的采样发散度。
  - `-o, --output`：将检索内容及 Qwen 返回存储为可后续审视对齐验证的落库文件点。

### 4.4 评测数据集准备：generate-queries / build-term-mapping
为科学的评估和模型自动化检查准备数据集。
- **`generate-queries`**：反向依据已存的语料库批量合成带标准出处答案对应的模拟自然语言问答对（Query）。
- **`build-term-mapping`**：针对生僻词或相似表述创建词元映射词典以增加对比匹配阶段纯度。

### 4.5 检索质量评估：quick-eval / eval-retrieval
在跑重磅 LLM 推论前，先仅仅看看你的分块和搜索策略召回真值的 Recall/MRR 打分是否有下降。
- **`quick-eval`**：仅在极小的几十条数据上对比如 `top_k=5` 等环境快速试错。
- **`eval-retrieval`**：正式挂载几千条集合产生大规模实验用对比论文图表级指标。

### 4.6 端到端与生成质量验证：experiments / eval-generation / report
- **`experiments`**：控制脚本在组合搜索和预训练生成等各项超大模块环境设定之间自动执行各种配置矩阵组全流程组合试验验证参数最优解集分布操作控制口。
- **`eval-generation`**：通过裁判模型或者是硬文本切句指标对最终 RAG 最后吐出来的解答质量给分。
- **`report`**：合并评分为最后的可发版 Markdown 总质量报表视图。

### 4.7 视觉与统计扩展支持：serve / stats
- **`serve`**：
  - **功能**：拉起浏览器应用。启动可视化问答或是实验回放终端页。
  - **命令**：`math-rag serve --target webui --port 7860` 
- **`stats`**：统计算子，它直接深入到分块统计语料字数大小覆盖离散统计中出报告。

## 5. 推荐运行与参数调优流程

### 5.1 首次初始化
1. 执行 `conda activate MathRag` 并 `pip install -e .`。
2. 确保你将数学书籍（例如 `math_analysis.pdf`）放入了 `data/raw/` 路径。
3. 执行 PDF 提取全流，等待向量编排：
   ```bash
   math-rag ingest math_analysis.pdf --batch-size 32
   ```

### 5.2 日常使用实验流程与功能链
在具有已序列化的本地系统引擎底座后：
1. **轻量查询测试**：
   ```bash
   math-rag rag -q "解释一致函数的极限收敛机制" -s hybrid --topk 3
   ```
2. **Web 可视化演示**：
   快速在本地浏览器与所做出的成果交互。
   ```bash
   math-rag serve --target webui --share
   ```
3. **策略修改后的严谨自动化测试**（如调节了 chunk_size）：
   - `math-rag build-index --rebuild`
   - `math-rag quick-eval` （观察召回退化状况）
   - 若找回理想则运行 `math-rag eval-generation` 看 Qwen 模型回答质量。
   - `math-rag report` 获取项目进展综合数字体现。

## 6. 关键输入与输出

- 关键输入
  - data/raw 下教材 PDF。
  - config.toml 配置。

- 关键中间产物
  - data/processed/retrieval/corpus.jsonl
  - data/evaluation/queries.jsonl
  - data/evaluation/term_mapping.json

- 关键输出
  - outputs/rag_results.jsonl
  - outputs/reports 下评测结果与图表

## 7. 文档使用建议

- 对外介绍项目时，优先引用本文件和根 README。
- 管理任务进度时，使用 docs/task.md。
- 调整研究方向时，更新 docs/plan.md。

