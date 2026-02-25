# scripts - 脚本入口

项目顶层脚本，覆盖完整实验流程：RAG 问答、对比实验、WebUI、数据工具。

## 模块结构

```
scripts/
├── runRag.py                # RAG 问答命令行入口
├── runExperiments.py        # 四组对比实验
├── experimentWebUI.py       # 实验 WebUI
└── buildEvalTermMapping.py  # 构建评测术语映射
```

## runRag.py

端到端 RAG 问答入口，支持单条和批量查询。

**用法**：

```bash
# 单条查询
python scripts/runRag.py --query "什么是一致收敛？"

# 批量查询（从文件读取）
python scripts/runRag.py --input data/evaluation/queries.jsonl \
                         --output outputs/rag_results.jsonl

# 指定检索策略
python scripts/runRag.py --query "泰勒公式" --retrieval hybrid

# 指定生成参数
python scripts/runRag.py --query "泰勒公式" --topk 5 \
                         --max-new-tokens 256 --temperature 0.1
```

**参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--query / -q` | 单条查询文本 | — |
| `--input` | 批量查询文件（JSONL） | — |
| `--output` | 输出文件路径 | `outputs/rag_results.jsonl` |
| `--retrieval` | 检索策略（bm25/vector/hybrid） | `hybrid` |
| `--topk` | 检索返回条数 | 5 |
| `--max-new-tokens` | 最大生成 token 数 | 由 config 决定 |
| `--temperature` | 生成温度 | 由 config 决定 |
| `--top-p` | top-p 采样 | 由 config 决定 |

**输出格式（JSONL）**：

```json
{
  "query": "什么是一致收敛？",
  "retrieved_terms": ["一致收敛", "逐点收敛"],
  "answer": "一致收敛是指...",
  "sources": ["数学分析(第5版)下 第123页"],
  "latency": 1.23
}
```

**前置条件**：Qwen 模型 + 检索索引已构建

---

## runExperiments.py

批量对比实验，覆盖四个实验组。

**实验组**：

| 组名 | 检索策略 | 是否使用 RAG |
|------|----------|-------------|
| `norag` | 无检索 | 否（直接生成） |
| `bm25` | BM25 | 是 |
| `vector` | 向量检索 | 是 |
| `hybrid` | 混合检索 | 是（主实验） |

**用法**：

```bash
# 运行所有实验组
python scripts/runExperiments.py

# 运行指定实验组
python scripts/runExperiments.py --groups norag bm25 vector hybrid

# 限制查询数量（调试用）
python scripts/runExperiments.py --limit 10

# 指定查询文件和输出目录
python scripts/runExperiments.py --queries data/evaluation/queries.jsonl \
                                 --output outputs/reports/
```

**参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--groups` | 实验组列表 | 全部 |
| `--limit` | 最多处理查询数（调试） | 无限制 |
| `--queries` | 查询文件路径 | `data/evaluation/queries.jsonl` |
| `--output` | 报告输出目录 | `outputs/reports/` |
| `--topk` | 检索 TopK | 5 |

**输出**：

| 文件 | 说明 |
|------|------|
| `outputs/reports/comparison_results.json` | 汇总结果 |
| `outputs/reports/comparison_chart.png` | 对比柱状图 |
| `outputs/reports/comparison_table.md` | Markdown 表格（可直接入论文） |

**前置条件**：Qwen 模型 + 检索索引已构建

---

## experimentWebUI.py

基于 Gradio 的实验可视化界面，无需命令行即可配置和运行实验。

```bash
python scripts/experimentWebUI.py
```

访问 http://localhost:7861 使用界面。

**功能**：
- 图形化选择实验组和参数
- 实时显示实验进度
- 可视化对比结果图表

**依赖**：`gradio`

---

## buildEvalTermMapping.py

构建评测感知术语映射，供检索评测脚本使用。

从 `queries.jsonl` 提取查询与相关术语，与语料库交叉验证，生成双向映射：
- 查询 → 相关术语列表
- 相关术语 → 同组所有术语

```bash
python scripts/buildEvalTermMapping.py
```

**输入**：
- `data/evaluation/queries.jsonl`
- `data/processed/retrieval/corpus.jsonl`

**输出**：`data/evaluation/term_mapping.json`

**前置条件**：检索语料库已构建（先运行 `retrieval/buildCorpus.py`）

## 典型运行顺序

```bash
# 1. 构建检索基础设施
python retrieval/buildCorpus.py
python scripts/buildEvalTermMapping.py

# 2. 纯检索评测（不需要 Qwen 模型）
python evaluation/quickEval.py
python evaluation/evalRetrieval.py

# 3. 完整 RAG 流程（需要 Qwen 模型）
python scripts/runRag.py --query "什么是泰勒展开？"
python scripts/runExperiments.py --limit 20

# 4. 生成质量评测
python evaluation/evalGeneration.py
```
