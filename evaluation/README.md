# Evaluation 模块

检索评测模块，用于评估不同检索方法的性能。

## 模块说明

### generateQueries.py

**评测查询自动生成脚本**，从术语库智能采样并生成评测数据。

**功能**：
- 从术语库中智能采样高频、高质量术语
- 自动提取相关术语（aliases + related_terms）
- 按学科分类生成 queries.jsonl
- 与现有数据合并，自动去重

**生成策略**：
- **80% 高质量术语**：优先选择相关术语丰富的术语
- **20% 随机术语**：保证多样性
- **相关术语构建**：term + aliases + related_terms（智能筛选）

**使用方法**：

```bash
# 默认：按固定数量生成（数学分析35，高等代数20，概率论20）
python evaluation/generateQueries.py

# 🔥 生成所有符合条件的术语（3102条）
python evaluation/generateQueries.py --all

# 按比例采样（如采样50%的术语）
python evaluation/generateQueries.py --ratio 0.5

# 自定义各学科数量
python evaluation/generateQueries.py --num-ma 50 --num-gd 30 --num-gl 30

# 提高质量阈值（要求至少2个相关术语）
python evaluation/generateQueries.py --all --min-related 2

# 不合并现有数据，直接覆盖
python evaluation/generateQueries.py --all --no-merge
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--all` | 使用所有符合条件的术语 | False |
| `--ratio` | 采样比例 (0-1) | None |
| `--num-ma` | 数学分析生成数量 | 35 |
| `--num-gd` | 高等代数生成数量 | 20 |
| `--num-gl` | 概率论生成数量 | 20 |
| `--min-related` | 最少相关术语数量阈值 | 1 |
| `--output` | 输出文件路径 | data/evaluation/queries.jsonl |
| `--no-merge` | 不合并现有数据 | False |

**输出示例**：

```json
{"query": "牛顿-莱布尼茨公式", "relevant_terms": ["牛顿-莱布尼茨公式", "牛顿茨公式", "微积分基本定理", "Newton-Leibniz formula", "原函数"], "subject": "数学分析"}
{"query": "矩阵的秩", "relevant_terms": ["矩阵的秩", "秩", "矩阵秩", "rank of a matrix", "满秩矩阵", "奇异矩阵", "行空间"], "subject": "高等代数"}
{"query": "显著性水平", "relevant_terms": ["显著性水平", "检验水平", "第一类错误概率", "α水平", "第一类错误"], "subject": "概率论"}
```

**生成数量对比**：

| 模式 | 数学分析 | 高等代数 | 概率论 | 总计 |
|------|----------|----------|--------|------|
| 默认（固定数量） | 35 | 20 | 20 | 75 |
| 50%采样 | ~773 | ~322 | ~455 | ~1550 |
| 全量生成 | 1547 | 645 | 910 | **3102** |

**💡 使用建议**：

1. **快速验证**：使用默认模式生成少量数据（~75条），快速验证评测流程
2. **中等规模**：使用 `--ratio 0.3` 生成中等规模数据（~900条），平衡覆盖度和评测效率
3. **完整评测**：使用 `--all` 生成全量数据（3102条），用于最终评测和论文实验

---

### evalRetrieval.py

检索评测脚本，计算多种评测指标并生成对比报告。

**功能**：
- 加载评测查询集
- 调用多种检索方法（BM25、Vector、Hybrid）
- 计算评测指标：Recall@K、MRR、nDCG@K、MAP
- 生成 JSON 报告和对比图表

**评测指标**：

1. **Recall@K**：在前 K 个结果中找到的相关文档比例
   - 公式：`Recall@K = 前K个结果中的相关文档数 / 总相关文档数`
   - 说明：衡量检索系统的召回能力

2. **MRR (Mean Reciprocal Rank)**：第一个相关文档排名倒数的平均值
   - 公式：`MRR = 平均(1 / 第一个相关文档的排名)`
   - 说明：衡量最相关结果的排名位置

3. **nDCG@K (Normalized Discounted Cumulative Gain)**：考虑排名位置的相关性评分
   - 公式：`nDCG@K = DCG@K / IDCG@K`
   - 说明：考虑相关性程度和排名位置，归一化到 0-1

4. **MAP (Mean Average Precision)**：所有相关文档的 Precision 平均值
   - 公式：`MAP = 平均(每个查询的 AP)`
   - 说明：综合考虑精确度和召回率

## 使用方法

### 基本用法

评测所有方法（BM25、Vector、Hybrid-Weighted、Hybrid-RRF）：

```bash
python evaluation/evalRetrieval.py
```

### 指定评测方法

仅评测 BM25 和 Vector：

```bash
python evaluation/evalRetrieval.py --methods bm25 vector
```

仅评测混合方法：

```bash
python evaluation/evalRetrieval.py --methods hybrid-weighted hybrid-rrf
```

### 调整 TopK 阈值

```bash
python evaluation/evalRetrieval.py --topk 20
```

### 生成对比图表

```bash
python evaluation/evalRetrieval.py --visualize
```

### 指定查询集和输出路径

```bash
# 使用快速验证集（默认）
python evaluation/evalRetrieval.py

# 使用完整评测集
python evaluation/evalRetrieval.py \
    --queries data/evaluation/queries_full.jsonl \
    --output outputs/reports/full_metrics.json

# 使用自定义查询集
python evaluation/evalRetrieval.py \
    --queries data/evaluation/custom_queries.jsonl \
    --output outputs/reports/custom_metrics.json
```

### 完整示例

```bash
python evaluation/evalRetrieval.py \
    --methods bm25 vector hybrid-weighted hybrid-rrf \
    --topk 10 \
    --visualize \
    --output outputs/reports/retrieval_metrics.json
```

## 输出结果

### 1. JSON 报告

输出文件：`outputs/reports/retrieval_metrics.json`

```json
{
  "timestamp": "2026-02-14 10:30:00",
  "queries_file": "data/evaluation/queries.jsonl",
  "total_queries": 35,
  "subject_distribution": {
    "数学分析": 20,
    "高等代数": 7,
    "概率论": 8
  },
  "topk": 10,
  "results": [
    {
      "method": "BM25",
      "total_queries": 35,
      "avg_metrics": {
        "recall@1": 0.8571,
        "recall@3": 0.9143,
        "recall@5": 0.9429,
        "recall@10": 0.9714,
        "mrr": 0.9048,
        "map": 0.8762,
        "ndcg@3": 0.9234,
        "ndcg@5": 0.9456,
        "ndcg@10": 0.9678
      },
      "avg_query_time": 0.0123
    }
  ]
}
```

### 2. 对比图表

输出文件：`outputs/reports/retrieval_comparison.png`

包含四个子图：
- Recall@K 对比（K=1,3,5,10）
- nDCG@K 对比（K=3,5,10）
- MRR 和 MAP 对比
- 平均查询时间对比

### 3. 控制台输出

```
==============================================================
📊 Math-RAG 检索评测
==============================================================
查询集: data/evaluation/queries.jsonl
评测方法: bm25, vector, hybrid-weighted, hybrid-rrf
TopK: 10
==============================================================

✅ 加载了 35 条查询

📚 学科分布:
  数学分析: 20 条
  概率论: 8 条
  高等代数: 7 条

==============================================================
📊 评测方法: BM25
==============================================================

📈 平均指标:
  Recall@1:  0.8571
  Recall@3:  0.9143
  Recall@5:  0.9429
  Recall@10: 0.9714
  MRR:       0.9048
  MAP:       0.8762
  nDCG@3:    0.9234
  nDCG@5:    0.9456
  nDCG@10:   0.9678
  平均查询时间: 12.34ms

==============================================================
📊 评测结果汇总
==============================================================
方法                  Recall@1   Recall@10  MRR        MAP        nDCG@10    查询时间  
------------------------------------------------------------------------------------------
BM25                 0.8571     0.9714     0.9048     0.8762     0.9678     12.34ms   
Vector               0.8286     0.9571     0.8857     0.8524     0.9542     23.45ms   
Hybrid-Weighted      0.8857     0.9857     0.9238     0.8976     0.9789     35.67ms   
Hybrid-RRF           0.9000     0.9857     0.9333     0.9087     0.9823     36.12ms   

✅ 评测完成！
```

## 评测数据集

项目提供两个评测数据集：

### 1. queries.jsonl（快速验证集）

**用途**：开发调试、算法迭代、快速对比

**规模**：105 条查询
- 数学分析：53 条
- 概率论：26 条  
- 高等代数：26 条

**评测时间**：~10 秒

**使用**：
```bash
python evaluation/evalRetrieval.py
# 或显式指定
python evaluation/evalRetrieval.py --queries data/evaluation/queries.jsonl
```

### 2. queries_full.jsonl（完整评测集）

**用途**：论文实验、最终评测、全面性能分析

**规模**：3102 条查询（覆盖整个术语库）
- 数学分析：1547 条
- 概率论：910 条
- 高等代数：645 条

**评测时间**：~5-10 分钟

**使用**：
```bash
python evaluation/evalRetrieval.py --queries data/evaluation/queries_full.jsonl
```

### 数据格式

```json
{"query": "一致收敛", "relevant_terms": ["一致收敛"], "subject": "数学分析"}
{"query": "逐点收敛", "relevant_terms": ["逐点收敛", "一致收敛"], "subject": "数学分析"}
```

详见 [data/evaluation/README.md](../data/evaluation/README.md) 了解数据集详情。

## 注意事项

1. **首次运行**：确保已构建语料库和索引
   ```bash
   python retrieval/buildCorpus.py
   python retrieval/retrievalBM25.py --build
   python retrieval/retrievalVector.py --build
   ```

2. **依赖库**：需要安装以下依赖
   - `rank-bm25`：BM25 检索
   - `sentence-transformers`：向量检索
   - `faiss-gpu` 或 `faiss-cpu`：向量索引
   - `matplotlib`：图表生成（可选）
   - `numpy`：数值计算

3. **性能优化**：
   - 使用 GPU 加速向量检索（需要 `faiss-gpu`）
   - 调整 TopK 阈值以平衡性能和指标
   - 批量评测时建议禁用详细日志

4. **评测数据质量**：
   - 确保 `relevant_terms` 中的术语在 corpus 中存在
   - 术语列表应按相关性从高到低排序
   - 建议评测集包含 50-100 条查询以获得可靠指标

## 扩展开发

### 添加新的评测指标

在 `evalRetrieval.py` 中添加新的指标计算函数：

```python
def calculateNewMetric(results: list[dict], relevantTerms: list[str]) -> float:
    """计算新指标"""
    # 实现逻辑
    return score
```

然后在 `evaluateMethod()` 函数中调用：

```python
metrics["new_metric"].append(calculateNewMetric(results, relevantTerms))
```

### 添加新的检索方法

在 `main()` 函数中初始化新的检索器：

```python
if method == "new_method":
    retrievers["NewMethod"] = NewRetriever(corpusPath)
```

并在 `argparse` 中添加选项：

```python
parser.add_argument(
    "--methods",
    choices=["bm25", "vector", "hybrid-weighted", "hybrid-rrf", "new_method"],
    ...
)
```

## 相关文档

- [数据集说明](../data/evaluation/README.md)
- [BM25 检索](../retrieval/README.md#retrievalbm25py)
- [向量检索](../retrieval/README.md#retrievalvectorpy)
- [混合检索](../retrieval/README.md#retrievalhybridpy)
