# 检索系统改进指南

本文档说明检索系统的改进方案和使用方法。

## 改进方案总览

| 方案 | 难度 | 改进内容 | 预期提升 |
|------|------|----------|----------|
| BM25+ | 简单 | 查询扩展、字段加权 | Recall@5 +5-10% |
| Hybrid+ | 中等 | 改进的融合策略、自适应权重 | Recall@5 +10-15% |
| Reranker | 高级 | Cross-Encoder 重排序 | Recall@5 +15-20% |

---

## 方案一：BM25+（简单改进）

### 改进内容

1. **查询扩展**：利用术语映射，添加相关术语到查询
2. **混合分词**：同时使用词级和字符级分词
3. **增加召回**：默认召回更多候选

### 使用方法

```bash
# 基础查询
python retrieval/retrievalBM25Plus.py --query "泰勒展开" --topk 10

# 启用查询扩展
python retrieval/retrievalBM25Plus.py --query "泰勒展开" --topk 10 --expand-query

# 批量查询
python retrieval/retrievalBM25Plus.py --query-file queries.txt --output results.json --expand-query
```

### 代码示例

```python
from retrieval.retrievalBM25Plus import BM25PlusRetriever

retriever = BM25PlusRetriever(
    corpusFile="data/processed/retrieval/corpus.jsonl",
    indexFile="data/processed/retrieval/bm25plus_index.pkl",
    termsFile="data/processed/terms/all_terms.json",
)

# 加载索引
retriever.loadIndex()
retriever.loadTermsMap()

# 查询（启用查询扩展）
results = retriever.search("泰勒展开", topK=10, expandQuery=True)
```

---

## 方案二：Hybrid+（中等改进）

### 改进内容

1. **百分位数归一化**：比 Min-Max 更鲁棒
2. **自适应权重**：根据查询难度动态调整 BM25/Vector 权重
3. **改进的 RRF**：动态调整 k 值
4. **召回因子**：检索更多候选用于融合

### 使用方法

```bash
# 加权融合（默认）
python retrieval/retrievalHybridPlus.py --query "泰勒展开" --topk 10

# RRF 融合
python retrieval/retrievalHybridPlus.py --query "泰勒展开" --topk 10 --strategy rrf

# 调整召回因子
python retrieval/retrievalHybridPlus.py --query "泰勒展开" --topk 10 --recall-factor 5

# 禁用查询扩展
python retrieval/retrievalHybridPlus.py --query "泰勒展开" --topk 10 --no-expand
```

### 代码示例

```python
from retrieval.retrievalHybridPlus import HybridPlusRetriever

retriever = HybridPlusRetriever(
    corpusFile="data/processed/retrieval/corpus.jsonl",
    bm25IndexFile="data/processed/retrieval/bm25plus_index.pkl",
    vectorIndexFile="data/processed/retrieval/vector_index.faiss",
    vectorEmbeddingFile="data/processed/retrieval/vector_embeddings.npz",
    termsFile="data/processed/terms/all_terms.json",
)

# 查询（使用 RRF 策略）
results = retriever.search(
    "泰勒展开",
    topK=10,
    strategy="rrf",
    recallFactor=5,  # 召回 50 个候选
)
```

---

## 方案三：Reranker（高级改进）

### 改进内容

1. **Cross-Encoder 重排序**：使用更精细的模型对候选重排
2. **两阶段检索**：先快速召回，再精确排序
3. **多路召回**：融合 BM25 和 Vector 的候选

### 依赖安装

```bash
pip install sentence-transformers
```

### 使用方法

```bash
# 基础查询（启用重排序）
python retrieval/retrievalWithReranker.py --query "泰勒展开" --topk 10

# 指定召回数量
python retrieval/retrievalWithReranker.py --query "泰勒展开" --topk 10 --recall-topk 100

# 指定重排序模型
python retrieval/retrievalWithReranker.py --query "泰勒展开" --topk 10 --reranker-model bge-reranker-large

# 禁用重排序（仅对比）
python retrieval/retrievalWithReranker.py --query "泰勒展开" --topk 10 --no-rerank
```

### 代码示例

```python
from retrieval.retrievalWithReranker import RerankerRetriever

retriever = RerankerRetriever(
    corpusFile="data/processed/retrieval/corpus.jsonl",
    bm25IndexFile="data/processed/retrieval/bm25plus_index.pkl",
    vectorIndexFile="data/processed/retrieval/vector_index.faiss",
    vectorEmbeddingFile="data/processed/retrieval/vector_embeddings.npz",
    rerankerModel="bge-reranker-base",
)

# 查询（召回 100 个候选，重排序后返回 top10）
results = retriever.search(
    "泰勒展开",
    topK=10,
    recallTopK=100,
    useReranker=True,
)
```

### 可选的重排序模型

| 模型 | 说明 | 推荐场景 |
|------|------|----------|
| `bge-reranker-base` | 中英双语，速度快 | 通用场景 |
| `bge-reranker-large` | 更大模型，精度更高 | 追求精度 |
| `cross-encoder/ms-marco-MiniLM` | 英文优化 | 英文内容 |

---

## 快速测试系统

使用快速评测系统可以在几分钟内评估改进效果：

```bash
# 快速测试（默认 20 条查询）
python evaluation/quickEval.py

# 指定测试数量
python evaluation/quickEval.py --num-queries 50

# 测试特定方法
python evaluation/quickEval.py --methods bm25plus hybrid_plus

# 使用全部查询
python evaluation/quickEval.py --all-queries

# 输出报告
python evaluation/quickEval.py --output reports/my_eval.json
```

### 输出示例

```
============================================================
🚀 快速检索评测系统
============================================================

方法              R@1      R@3      R@5     R@10      MRR   nDCG@5   时间 (s)
---------------------------------------------------------------------------
BM25            6.67%    25.24%   27.62%   45.24%   0.5234   0.3138    0.004
BM25+           8.57%    28.57%   32.38%   48.57%   0.5612   0.3521    0.005
Hybrid+         9.52%    30.48%   35.24%   52.38%   0.5891   0.3812    0.012

🏆 Recall@5 最佳方法：Hybrid+ (35.24%)
```

---

## 改进建议

### 立即可做（简单，效果明显）

1. 使用 `Hybrid+` 替代原有混合检索
2. 增加 `recallFactor` 到 3-5
3. 启用查询扩展

### 短期改进（中等难度）

1. 尝试不同的归一化方法（percentile 推荐）
2. 调整 RRF 的 k 值（默认 60）
3. 针对特定学科调整权重

### 长期改进（需要时间）

1. 微调嵌入模型（使用数学领域数据）
2. 部署 Cross-Encoder 重排序
3. 构建术语同义词典

---

## 性能对比

| 方法 | Recall@1 | Recall@5 | MRR | 延迟 |
|------|----------|----------|-----|------|
| BM25 | 6.7% | 27.7% | 52.3% | 4ms |
| BM25+ | 8.6% | 32.4% | 56.1% | 5ms |
| Vector | 6.7% | 19.5% | 44.4% | 50ms |
| Hybrid+ | 9.5% | 35.2% | 58.9% | 12ms |
| Hybrid+ + Reranker | 12.4% | 42.1% | 65.3% | 200ms |

> 注：以上数据为示例，实际效果需要运行评测

---

## 故障排查

### 索引构建失败

```bash
# 删除旧索引，重新构建
rm -rf data/processed/retrieval/*_index*
python retrieval/retrievalBM25Plus.py --rebuild-index
```

### 术语文件不存在

```bash
# 检查术语目录
ls data/processed/terms/

# 如果为空，需要先生成术语数据
```

### 内存不足

```bash
# 减少召回因子
python retrieval/retrievalHybridPlus.py --query "测试" --recall-factor 2

# 或减少抽样数量
python evaluation/quickEval.py --num-queries 10
```
