# 检索模块（Retrieval）

检索基线实现与语料构建模块。

## 📂 模块结构

```
retrieval/
├── __init__.py           # 模块初始化
├── buildCorpus.py        # 构建检索语料
├── retrievalBM25.py      # BM25 检索基线（待实现）
├── retrievalVec.py       # 向量检索基线（待实现）
├── hybridRetrieval.py    # 混合检索（待实现）
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

### 2. retrievalBM25.py - BM25 检索基线（待实现）

**功能**：
- 构建 BM25 索引
- 单查询检索
- 批量查询
- TopK 结果输出

**依赖**：
- `rank-bm25`

---

### 3. retrievalVec.py - 向量检索基线（待实现）

**功能**：
- 构建向量索引
- FAISS 加速检索
- TopK 结果输出

**依赖**：
- `sentence-transformers`
- `faiss-cpu`

---

### 4. hybridRetrieval.py - 混合检索（待实现）

**功能**：
- 融合 BM25 和向量检索结果
- 归一化和加权策略
- TopK 结果输出

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
    └── ...
         ↓
    [retrievalBM25.py / retrievalVec.py]
         ↓
outputs/retrieval/             (检索结果)
    ├── bm25_results.json
    ├── vec_results.json
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
- [ ] Task 3: BM25 检索基线
- [ ] Task 4: 向量检索基线
- [ ] Task 5: 混合检索
- [ ] Task 6: 评测框架

---

## 🔗 相关文档

- [项目规划](../docs/plan.md)
- [任务列表](../docs/task.md)
- [代码规范](../AGENTS.md)
