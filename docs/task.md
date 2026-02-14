# Math-RAG 当下计划（2026-02-14）

## 目标
- 以已完成的术语 JSON 切分为起点，落地“可跑通、可复现”的检索基线与评测流程

## 当前状态
- 已完成：`data/processed/chunk/<书名>/<术语>.json` 术语级切分
- 进行中：检索语料构建与检索基线准备

## 本阶段任务（按顺序，含实现细节）

### 任务1：数据核验与统计
**状态**：`todo`  
**输入**：`data/processed/chunk/**.json`  
**输出**：`outputs/reports/chunk_stats.json`  
**脚本**：`scripts/build_chunk_stats.py`  
**验收标准**：
- 字段缺失率统计（按字段类型）
- 长度分布统计（term、definitions、formula等）
- 学科覆盖统计（数学分析、高等代数、概率论）
- 术语总数与书籍分布
- 输出 JSON 格式规范，可读性高

**依赖**：无

---

### 任务2：构建检索语料
**状态**：`todo`  
**输入**：`data/processed/chunk/**.json`  
**输出**：`data/processed/retrieval/corpus.jsonl`  
**脚本**：`scripts/build_corpus.py`  
**拼接规则**：
- 按 `term → aliases → definitions.text → formula → usage → applications → disambiguation → related_terms` 顺序
- 用换行分隔各个部分
- 缺失字段跳过，不输出空行
- LaTeX 公式保持原样

**验收标准**：
- 每行包含必需字段：`doc_id`、`term`、`subject`、`text`、`source`、`page`
- JSONL 格式正确，每行可独立解析
- 文本拼接符合规则，无多余空白
- 与 chunk JSON 数量一致

**依赖**：任务1（可选，建议先完成数据核验）

---

### 任务3：BM25 基线检索
**状态**：`todo`  
**输入**：`data/processed/retrieval/corpus.jsonl`  
**依赖库**：`rank-bm25`  
**输出**：
- BM25 索引文件（pickle 格式）
- TopK 查询结果（JSON）
**脚本**：`scripts/retrieval_bm25.py`  

**验收标准**：
- 脚本可对任意 query 输出 TopK（默认 K=10）
- 支持批量查询（从文件读取）
- 输出包含：doc_id、term、score、rank
- 索引可保存和加载，避免重复构建

**依赖**：任务2

---

### 任务4：向量检索基线
**状态**：`todo`  
**输入**：`data/processed/retrieval/corpus.jsonl`  
**依赖库**：`sentence-transformers`、`faiss-cpu`  
**模型**：推荐使用 `paraphrase-multilingual-MiniLM-L12-v2` 或 `moka-ai/m3e-base`  
**输出**：
- FAISS 索引文件（.index）
- 向量嵌入文件（.npy）
- TopK 查询结果（JSON）
**脚本**：`scripts/retrieval_vec.py`  

**验收标准**：
- 脚本可对任意 query 输出 TopK（默认 K=10）
- 支持批量查询
- 索引构建可配置（维度、距离度量）
- 查询速度满足需求（<1s per query）

**依赖**：任务2

---

### 任务5：混合检索
**状态**：`todo`  
**输入**：
- BM25 检索结果
- 向量检索结果
**策略**：
- 归一化分数：min-max 或 z-score
- 加权融合：默认 alpha=0.5（BM25），beta=0.5（向量）
- 支持可配置权重
**输出**：
- 混合 TopK 结果（JSON）
**脚本**：`scripts/retrieval_hybrid.py`  

**验收标准**：
- 脚本输出混合 TopK
- 支持权重配置（命令行参数或配置文件）
- 输出格式与单一检索方法一致
- 融合策略可扩展（支持RRF等其他方法）

**依赖**：任务3、任务4

---

### 任务6：评测集与指标
**状态**：`todo`  
**输入**：
- 手工标注数据：`data/evaluation/queries.jsonl`
  - 格式：`{"query": "一致收敛", "relevant_terms": ["一致收敛", "逐点收敛"], "subject": "数学分析"}`
  - 建议 50-100 条查询
**输出**：`outputs/reports/retrieval_metrics.json`  
**指标**：
- Recall@K（K=1,3,5,10）
- MRR（Mean Reciprocal Rank）
- nDCG@K（K=3,5,10）
- MAP（Mean Average Precision）
**脚本**：`scripts/eval_retrieval.py`  

**验收标准**：
- 固定格式输出，结果可复现
- 支持对比多种检索方法（BM25、向量、混合）
- 输出包含详细指标和统计信息
- 生成对比图表（可选）

**依赖**：任务3、任务4、任务5

## 产出物
- `data/processed/retrieval/corpus.jsonl` - 检索语料库
- `scripts/` - 数据处理、检索、评测脚本
  - `build_chunk_stats.py` - 数据统计
  - `build_corpus.py` - 语料构建
  - `retrieval_bm25.py` - BM25 检索
  - `retrieval_vec.py` - 向量检索
  - `retrieval_hybrid.py` - 混合检索
  - `eval_retrieval.py` - 检索评测
- `outputs/` - 配置与实验记录
  - `reports/` - 统计报告与评测结果
  - `indexes/` - 检索索引文件（BM25、FAISS）
  - `logs/` - 运行日志

## 目录结构建议
根据当前阶段需求，建议创建以下目录：
```
Math-RAG/
├── data/
│   ├── raw/                    # 原始 PDF
│   ├── processed/
│   │   ├── ocr/                # OCR 输出
│   │   ├── terms/              # 术语提取结果
│   │   ├── chunk/              # 术语 JSON 切分
│   │   └── retrieval/          # 检索语料（新增）
│   └── evaluation/             # 评测数据（新增）
│       └── queries.jsonl       # 标注查询集
├── scripts/                    # 脚本目录（新增）
│   ├── build_chunk_stats.py
│   ├── build_corpus.py
│   ├── retrieval_bm25.py
│   ├── retrieval_vec.py
│   ├── retrieval_hybrid.py
│   └── eval_retrieval.py
├── outputs/                    # 输出目录（新增）
│   ├── reports/
│   ├── indexes/
│   └── logs/
├── dataGen/                    # 数据生成脚本（已有）
└── docs/                       # 文档（已有）
```

## 依赖变更（仅记录，实施时再改）
可能新增依赖：`rank-bm25`、`sentence-transformers`、`faiss-cpu`

## 风险与验证
- 术语 JSON 字段不一致需要先做字段映射
- 检索噪声可能影响小模型生成质量，先验证 BM25 与向量检索质量
