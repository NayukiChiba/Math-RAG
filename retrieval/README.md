# 检索模块（Retrieval）

检索基线实现与语料构建模块。

## 📂 模块结构

```
retrieval/
├── __init__.py           # 模块初始化
├── buildCorpus.py        # 构建检索语料
├── retrievalBM25.py      # BM25 检索基线
├── retrievalVector.py    # 向量检索基线
├── retrievalHybrid.py    # 混合检索
└── README.md             # 本文档
```

## 🔧 功能模块

### 1. buildCorpus.py - 构建检索语料

从术语 JSON 文件构建统一的检索语料（JSONL 格式）。

**功能**：
- 读取 `data/processed/chunk/**/*.json` 中的所有术语文件
- 按规则拼接文本字段
- 输出 JSONL 格式的检索语料

**文本拼接顺序**：
```
term → aliases → definitions.text → formula → usage → applications → disambiguation → related_terms
```

**输出格式**（JSONL）：
```json
{"doc_id": "ma-001", "term": "一致收敛", "subject": "数学分析", "text": "术语: 一致收敛\n定义: ...", "source": "数学分析(第5版)上", "page": 123}
{"doc_id": "aa-002", "term": "特征多项式", "subject": "高等代数", "text": "术语: 特征多项式\n定义: ...", "source": "高等代数(第五版)", "page": 45}
```

**使用方法**：
```bash
# 直接运行
python retrieval/buildCorpus.py

# 或作为模块运行
python -m retrieval.buildCorpus
```

**输入**：
- 目录：`data/processed/chunk/**/*.json`

**输出**：
- 文件：`data/processed/retrieval/corpus.jsonl`

**功能特性**：
- ✅ 自动创建输出目录
- ✅ 逐书籍统计处理进度
- ✅ 自动验证输出格式
- ✅ 显示样本数据
- ✅ 错误处理与跳过机制

---

### 2. retrievalBM25.py - BM25 检索基线

**功能**：
- 构建 BM25 索引
- 单查询检索
- 批量查询
- TopK 结果输出
- 索引保存和加载

**依赖**：
- `rank-bm25`

**使用方法**：
```bash
# 单次查询
python retrieval/retrievalBM25.py --query "泰勒展开" --topk 10

# 批量查询
python retrieval/retrievalBM25.py --query-file queries.txt --output results.json

# 重新构建索引
python retrieval/retrievalBM25.py --rebuild-index
```

**输入**：
- 语料文件：`data/processed/retrieval/corpus.jsonl`
- 查询字符串或查询文件

**输出**：
- 索引文件：`data/processed/retrieval/bm25_index.pkl`（自动保存和加载）
- 查询结果：JSON 格式，包含 rank、doc_id、term、subject、score、source、page

**输出格式示例**：
```json
{
  "泰勒展开": [
    {
      "rank": 1,
      "doc_id": "ma-积分余项",
      "term": "积分余项",
      "subject": "数学分析",
      "score": 19.9007,
      "source": "数学分析(第5版)下(华东师范大学数学系)",
      "page": 57
    }
  ]
}
```

---

### 3. retrievalVector.py - 向量检索基线

**功能**：
- 构建向量索引（FAISS）
- 使用 Sentence Transformers 进行文本嵌入
- 单查询检索
- 批量查询
- TopK 结果输出
- 索引和嵌入保存加载

**依赖**：
- `sentence-transformers`
- `faiss-cpu`

**推荐模型**：
- `paraphrase-multilingual-MiniLM-L12-v2`（多语言，384 维）
- `moka-ai/m3e-base`（中文优化，768 维）

**使用方法**：
```bash
# 单次查询
python retrieval/retrievalVector.py --query "泰勒展开" --topk 10

# 批量查询
python retrieval/retrievalVector.py --query-file queries.txt --output results.json

# 重新构建索引
python retrieval/retrievalVector.py --rebuild-index

# 指定模型
python retrieval/retrievalVector.py --model moka-ai/m3e-base --query "泰勒展开"

# 指定批次大小
python retrieval/retrievalVector.py --batch-size 64 --rebuild-index
```

**输入**：
- 语料文件：`data/processed/retrieval/corpus.jsonl`
- 查询字符串或查询文件

**输出**：
- FAISS 索引：`data/processed/retrieval/vector_index.faiss`
- 索引元数据：`data/processed/retrieval/vector_index.faiss.meta.json`
- 嵌入向量：`data/processed/retrieval/vector_embeddings.npz`
- 查询结果：JSON 格式，包含 rank、doc_id、term、subject、score、source、page

**输出格式示例**：
```json
{
  "泰勒展开": [
    {
      "rank": 1,
      "doc_id": "ma-泰勒级数",
      "term": "泰勒级数",
      "subject": "数学分析",
      "score": 0.8756,
      "source": "数学分析(第5版)下(华东师范大学数学系)",
      "page": 134
    }
  ]
}
```

**特性**：
- ✅ 余弦相似度搜索（向量标准化 + FAISS IndexFlatIP）
- ✅ 批量嵌入计算（可配置批次大小）
- ✅ 自动时间戳验证（语料更新后自动重建）
- ✅ 模型一致性检查
- ✅ 进度条显示

---

### 4. retrievalHybrid.py - 混合检索

**功能**：
- 结合 BM25 和向量检索的优势
- 支持多种归一化策略（min-max、z-score）
- 支持多种融合策略（加权融合、RRF）
- 可配置权重参数

**融合策略**：
- **加权融合（Weighted）**：对归一化后的分数进行加权求和
  - 支持 min-max 和 z-score 归一化
  - 可配置 BM25 和向量权重（alpha, beta）
- **RRF（Reciprocal Rank Fusion）**：基于排名的融合
  - 公式：`RRF(d) = Σ 1/(k + rank(d))`
  - 参数 k 默认为 60

**使用方法**：
```bash
# 加权融合（默认）
python retrieval/retrievalHybrid.py --query "泰勒展开" --topk 10

# 指定权重
python retrieval/retrievalHybrid.py --query "泰勒展开" --alpha 0.7 --beta 0.3

# 使用 z-score 归一化
python retrieval/retrievalHybrid.py --query "泰勒展开" --normalization zscore

# 使用 RRF 融合
python retrieval/retrievalHybrid.py --query "泰勒展开" --strategy rrf

# 批量查询
python retrieval/retrievalHybrid.py --query-file queries.txt --output results.json
```

**输入**：
- BM25 索引：`data/processed/retrieval/bm25_index.pkl`
- 向量索引：`data/processed/retrieval/vector_index.faiss`
- 向量嵌入：`data/processed/retrieval/vector_embeddings.npz`

**输出**：
- 混合检索结果（JSON 格式）
- 包含融合分数和各方法的原始分数/排名

**输出格式示例**（Weighted）：
```json
{
  "泰勒展开": [
    {
      "rank": 1,
      "doc_id": "ma-泰勒级数",
      "term": "泰勒级数",
      "subject": "数学分析",
      "score": 0.8523,
      "bm25_score": 0.7845,
      "vector_score": 0.9201,
      "source": "数学分析(第5版)下",
      "page": 4
    }
  ]
}
```

**输出格式示例**（RRF）：
```json
{
  "泰勒展开": [
    {
      "rank": 1,
      "doc_id": "ma-泰勒级数",
      "term": "泰勒级数",
      "subject": "数学分析",
      "score": 0.0328,
      "bm25_rank": 1,
      "vector_rank": 2,
      "source": "数学分析(第5版)下",
      "page": 4
    }
  ]
}
```

**特性**：
- ✅ 自动初始化 BM25 和向量检索器
- ✅ 索引不存在时自动构建
- ✅ 支持单查询和批量查询
- ✅ 可配置融合策略和参数
- ✅ 输出包含详细的分数/排名信息

---

## 📊 数据流

```
data/processed/chunk/          (输入：术语 JSON)
    ├── 数学分析(第5版)上/
    │   ├── ma-001.json
    │   └── ...
    ├── 高等代数(第五版)/
    │   ├── aa-001.json
    │   └── ...
    └── ...
         ↓
    [buildCorpus.py]
         ↓
data/processed/retrieval/      (输出：检索语料)
    ├── corpus.jsonl           (JSONL 格式语料)
    ├── bm25_index.pkl         (BM25 索引)
    ├── vector_index.faiss     (FAISS 向量索引)
    ├── vector_index.faiss.meta.json  (索引元数据)
    └── vector_embeddings.npz  (嵌入向量)
         ↓
    [retrievalBM25.py / retrievalVector.py / retrievalHybrid.py]
         ↓
outputs/retrieval/             (检索结果)
    ├── bm25_results.json
    ├── vector_results.json
    └── hybrid_results.json
```

## 🚀 快速开始

1. **构建语料**：
```bash
python retrieval/buildCorpus.py
```

2. **验证输出**：
```bash
# 检查文件是否生成
ls data/processed/retrieval/corpus.jsonl

# 查看行数
Get-Content data/processed/retrieval/corpus.jsonl | Measure-Object -Line
```

3. **查看样本**：
```bash
# Windows PowerShell
Get-Content data/processed/retrieval/corpus.jsonl -TotalCount 3
```

---

## 📝 开发规范

遵循项目代码书写规范（详见 `AGENTS.md`）：

- 文件命名：驼峰命名法（camelCase）
- 函数命名：驼峰命名法，动词开头
- 变量命名：驼峰命名法，名词为主
- 路径处理：统一使用 `os.path` + `config.py`
- 注释：使用中文
- 类型提示：使用现代类型注解（dict, list 等）

---

## 📌 任务进度

- [x] Task 1: 数据核验与统计（`dataStat/chunkStatistics.py`）
- [x] Task 2: 构建检索语料（`retrieval/buildCorpus.py`）
- [x] Task 3: BM25 检索基线（`retrieval/retrievalBM25.py`）
- [x] Task 4: 向量检索基线（`retrieval/retrievalVector.py`）
- [x] Task 5: 混合检索（`retrieval/retrievalHybrid.py`）
- [ ] Task 6: 评测框架

---

## 🔗 相关文档

- [项目规划](../docs/plan.md)
- [任务列表](../docs/task.md)
- [代码规范](../AGENTS.md)
