# 评测模块（Evaluation）

检索与生成质量评测，支持多方法对比和报告生成。

## 模块结构

```
evaluation/
├── __init__.py           # 模块初始化
├── evalRetrieval.py      # 检索评测
├── evalGeneration.py     # 生成质量评测
├── quickEval.py          # 快速检索评测
└── README.md             # 本文档
```

## evalRetrieval.py - 检索评测

多方法检索评测，计算标准 IR 指标。

**评测指标**：

| 指标 | 说明 |
|------|------|
| Recall@K (K=1,3,5,10) | 前 K 个结果中的相关文档比例 |
| MRR | 第一个相关文档排名倒数的平均值 |
| nDCG@K (K=3,5,10) | 考虑排名位置的归一化相关性评分 |
| MAP | 所有相关文档的 Precision 平均值 |

**使用方法**：

```bash
# 评测所有方法
math-rag eval-retrieval

# 指定评测方法
math-rag eval-retrieval --methods bm25 vector hybrid-weighted hybrid-rrf

# 调整 TopK
python evaluation/evalRetrieval.py --topk 20

# 生成对比图表
python evaluation/evalRetrieval.py --visualize

# 指定查询集和输出路径
python evaluation/evalRetrieval.py \
    --queries data/evaluation/queries_full.jsonl \
    --output outputs/reports/full_metrics.json
```

**输出**：
- JSON 报告：`outputs/reports/retrieval_metrics.json`
- 对比图表：`outputs/reports/retrieval_comparison.png`（需 `--visualize`）

---

## quickEval.py - 快速检索评测

轻量级检索评测，适合快速验证和调参。

```bash
# 快速测试（默认 20 条查询）
math-rag quick-eval

# 指定测试数量
math-rag quick-eval --num-queries 50

# 使用全部查询
python evaluation/quickEval.py --all-queries

# 输出报告
python evaluation/quickEval.py --output reports/my_eval.json
```

**输出示例**：

```
方法              R@1      R@3      R@5     R@10      MRR   nDCG@5   时间 (s)
---------------------------------------------------------------------------
BM25            6.67%    25.24%   27.62%   45.24%   0.5234   0.3138    0.004
BM25+           8.57%    28.57%   32.38%   48.57%   0.5612   0.3521    0.005
Hybrid+         9.52%    30.48%   35.24%   52.38%   0.5891   0.3812    0.012
```

---

## evalGeneration.py - 生成质量评测

评测 RAG 生成回答的质量。

**评测指标**：
- 术语命中率（回答包含目标相关术语）
- 来源引用率（书名/页码正确引用）
- 回答非空率
- BLEU / ROUGE（可选）

```bash
math-rag eval-generation
```

---

## 评测数据集

| 数据集 | 规模 | 用途 | 评测时间 |
|--------|------|------|----------|
| `queries.jsonl` | 105 条 | 开发调试、快速对比 | ~10 秒 |
| `queries_full.jsonl` | 3102 条 | 论文实验、完整评测 | ~5-10 分钟 |

**学科分布**（105 条快速集）：
- 数学分析：53 条
- 概率论：26 条
- 高等代数：26 条

**数据格式**：
```json
{"query": "一致收敛", "relevant_terms": ["一致收敛"], "subject": "数学分析"}
```

## 前置条件

运行评测前需确保：

1. 已构建检索语料和索引：
   ```bash
   math-rag build-index
   ```

2. 已生成评测查询集：
   ```bash
   math-rag generate-queries
   ```

3. 已构建术语映射（用于检索评测）：
   ```bash
   math-rag build-term-mapping
   ```

## 依赖

| 包 | 用途 | 必需 |
|----|------|------|
| `rank_bm25` | BM25 检索 | 是 |
| `sentence-transformers` | 向量检索 | 否（仅向量/混合评测） |
| `faiss-cpu` / `faiss-gpu` | 向量索引 | 否（仅向量/混合评测） |
| `matplotlib` | 图表生成 | 否（仅 `--visualize`） |
| `numpy` | 数值计算 | 是 |
