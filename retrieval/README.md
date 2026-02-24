# 检索模块（Retrieval）

检索系统实现：语料构建、多种检索策略、查询改写。

## 模块结构

```
retrieval/
├── __init__.py        # 模块初始化与导出
├── buildCorpus.py     # 构建检索语料
├── retrievers.py      # 全部检索器（7 种）
├── queryRewrite.py    # 查询改写与同义词扩展
└── README.md          # 本文档
```

## 检索器一览

所有检索器统一实现在 `retrievers.py` 中，共 7 种：

| 检索器 | 类型 | 说明 | 依赖 |
|--------|------|------|------|
| `BM25Retriever` | 稀疏 | BM25 基线检索 | `rank_bm25` |
| `BM25PlusRetriever` | 稀疏 | BM25 + 查询扩展 + 混合分词 | `rank_bm25` |
| `VectorRetriever` | 稠密 | 语义向量检索（FAISS + Sentence-Transformers） | `faiss`, `sentence-transformers` |
| `HybridRetriever` | 混合 | BM25 + 向量加权融合 / RRF | `rank_bm25`, `faiss`, `sentence-transformers` |
| `HybridPlusRetriever` | 混合 | 改进融合（百分位归一化、自适应权重、动态 RRF） | 同上 |
| `RerankerRetriever` | 重排 | 两阶段检索 + Cross-Encoder 重排序 | `sentence-transformers` |
| `AdvancedRetriever` | 多路 | 多路召回 + 查询改写 + 重排序 | 同上 |

## 使用方法

### 基本用法

```python
from retrieval.retrievers import BM25Retriever, HybridRetriever

# BM25 检索
retriever = BM25Retriever(
    corpusFile="data/processed/retrieval/corpus.jsonl",
    indexFile="data/processed/retrieval/bm25_index.pkl"
)
retriever.loadIndex()
results = retriever.search("泰勒展开", topK=10)

# 混合检索（加权融合）
retriever = HybridRetriever(
    corpusFile="data/processed/retrieval/corpus.jsonl",
    bm25IndexFile="data/processed/retrieval/bm25_index.pkl",
    vectorIndexFile="data/processed/retrieval/vector_index.faiss",
    vectorEmbeddingFile="data/processed/retrieval/vector_embeddings.npz"
)
results = retriever.search("泰勒展开", topK=10, strategy="weighted")

# RRF 融合
results = retriever.search("泰勒展开", topK=10, strategy="rrf")
```

### BM25+ 检索（带查询扩展）

```python
from retrieval.retrievers import BM25PlusRetriever

retriever = BM25PlusRetriever(
    corpusFile="data/processed/retrieval/corpus.jsonl",
    indexFile="data/processed/retrieval/bm25plus_index.pkl",
    termsFile="data/processed/terms/all_terms.json"
)
retriever.loadIndex()
retriever.loadTermsMap()

# 启用查询扩展（利用术语映射添加相关术语）
results = retriever.search("泰勒展开", topK=10, expandQuery=True)
```

### HybridPlus 检索（改进混合）

```python
from retrieval.retrievers import HybridPlusRetriever

retriever = HybridPlusRetriever(
    corpusFile="data/processed/retrieval/corpus.jsonl",
    bm25IndexFile="data/processed/retrieval/bm25plus_index.pkl",
    vectorIndexFile="data/processed/retrieval/vector_index.faiss",
    vectorEmbeddingFile="data/processed/retrieval/vector_embeddings.npz",
    termsFile="data/processed/terms/all_terms.json"
)

# 使用百分位归一化 + 自适应权重
results = retriever.search("泰勒展开", topK=10, strategy="rrf", recallFactor=5)
```

### Reranker 检索（两阶段）

```python
from retrieval.retrievers import RerankerRetriever

retriever = RerankerRetriever(
    corpusFile="data/processed/retrieval/corpus.jsonl",
    bm25IndexFile="data/processed/retrieval/bm25plus_index.pkl",
    vectorIndexFile="data/processed/retrieval/vector_index.faiss",
    vectorEmbeddingFile="data/processed/retrieval/vector_embeddings.npz",
    rerankerModel="BAAI/bge-reranker-v2-mixed"
)

# 召回 100 个候选，重排序后返回 top10
results = retriever.search("泰勒展开", topK=10, recallTopK=100, useReranker=True)
```

### 批量查询与结果保存

```python
from retrieval.retrievers import loadQueriesFromFile, saveResults, printResults

# 从 JSONL 加载查询
queries = loadQueriesFromFile("data/evaluation/queries.jsonl")

# 批量检索
allResults = {}
for q in queries:
    allResults[q["query"]] = retriever.search(q["query"], topK=10)

# 保存结果
saveResults(allResults, "outputs/retrieval/results.json")

# 打印结果
printResults(results)
```

## 查询改写（queryRewrite.py）

`QueryRewriter` 通过 `MATH_SYNONYMS` 同义词典扩展查询，提升召回率。

```python
from retrieval.queryRewrite import QueryRewriter

rewriter = QueryRewriter()
expanded = rewriter.rewrite("泰勒展开")
# → ["泰勒展开", "Taylor展开", "泰勒级数", "泰勒公式", ...]
```

- 包含 144 条数学同义词映射
- 覆盖三大学科：数学分析、高等代数、概率论与数理统计
- 100% 覆盖全部 102 条评测查询

## 语料构建（buildCorpus.py）

从术语 JSON 生成统一检索语料（JSONL 格式）。

```bash
python retrieval/buildCorpus.py
```

**输入**：`data/processed/chunk/**/*.json`

**输出**：`data/processed/retrieval/corpus.jsonl`

**文本拼接顺序**：
```
term → aliases → definitions.text → formula → usage → applications → disambiguation → related_terms
```

**输出格式**：
```json
{"doc_id": "ma-001", "term": "一致收敛", "subject": "数学分析", "text": "术语: 一致收敛\n定义: ...", "source": "数学分析(第5版)上", "page": 123}
```

## 数据流

```
data/processed/chunk/           术语 JSON
         ↓
    [buildCorpus.py]
         ↓
data/processed/retrieval/
    ├── corpus.jsonl            检索语料
    ├── bm25_index.pkl          BM25 索引
    ├── bm25plus_index.pkl      BM25+ 索引
    ├── vector_index.faiss      FAISS 向量索引
    └── vector_embeddings.npz   嵌入向量
         ↓
    [retrievers.py]
         ↓
    检索结果（JSON）
```

## 配置

检索参数通过 `config.toml` 的 `[retrieval]` 段管理，使用 `config.getRetrievalConfig()` 读取：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `recall_factor` | 5 | 召回因子（检索 topK × factor 用于融合） |
| `rrf_k` | 60 | RRF 参数 k |
| `bm25_default_weight` | 0.7 | BM25 默认权重 |
| `vector_default_weight` | 0.3 | 向量默认权重 |
| `default_normalization` | `percentile` | 归一化方法 |
| `default_vector_model` | `paraphrase-multilingual-MiniLM-L12-v2` | 向量模型 |
| `default_reranker_model` | `BAAI/bge-reranker-v2-mixed` | 重排序模型 |
| `use_hybrid_tokenization` | `true` | 混合分词（词级 + 字符级） |

## 依赖

| 包 | 用途 | 必需 |
|----|------|------|
| `rank_bm25` | BM25 检索 | 是（BM25 系列） |
| `faiss-cpu` / `faiss-gpu` | 向量索引 | 否（向量/混合检索需要） |
| `sentence-transformers` | 文本嵌入 + 重排序 | 否（向量/重排序需要） |
| `numpy` | 数值计算 | 是 |
| `jieba` | 中文分词 | 是（内置，无需额外安装） |
