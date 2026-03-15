# Math-RAG 任务进度（2026-02-24）

## 当前状态

- ✅ 已完成：数据层、检索层（重构完成）、RAG 生成层（Task-1~11）
- 🔄 进行中：评测体系完善（Task-12~16）

## 本阶段任务（按顺序，含实现细节）

### 任务1：数据核验与统计
**状态**：`completed` ✅  
**输入**：`data/processed/chunk/**.json`  
**输出**：
- `data/stats/chunkStatistics.json`（统计报告）
- `data/stats/visualizations/*.png`（可视化图表）

**脚本**：`dataStat/chunkStatistics.py`  
**验收标准**：
- ✅ 字段缺失率统计（按字段类型）
- ✅ 长度分布统计（term、definitions、formula等）
- ✅ 学科覆盖统计（数学分析、高等代数、概率论）
- ✅ 术语总数与书籍分布
- ✅ 输出 JSON 格式规范，可读性高
- ✅ 可视化图表生成（6张高清图表）

**依赖**：无

---

### 任务2：构建检索语料
**状态**：`completed` ✅  
**输入**：`data/processed/chunk/**.json`  
**输出**：`data/processed/retrieval/corpus.jsonl`  
**脚本**：`retrieval/buildCorpus.py`  

**验收标准**：
- ✅ 每行包含必需字段：`doc_id`、`term`、`subject`、`text`、`source`、`page`
- ✅ JSONL 格式正确，每行可独立解析
- ✅ 文本拼接符合规则，无多余空白
- ✅ 与 chunk JSON 数量一致

**依赖**：任务1

---

### 任务3：BM25 基线检索
**状态**：`completed` ✅  
**输入**：`data/processed/retrieval/corpus.jsonl`  
**依赖库**：`rank-bm25`  
**输出**：
- BM25 索引文件（pickle 格式）
- TopK 查询结果（JSON）

**实现**：`retrieval/retrievers.py` → `BM25Retriever`

**验收标准**：
- ✅ 支持对任意 query 输出 TopK（默认 K=10）
- ✅ 支持批量查询（从文件读取）
- ✅ 输出包含：doc_id、term、score、rank
- ✅ 索引可保存和加载，避免重复构建

**依赖**：任务2

---

### 任务4：向量检索基线
**状态**：`completed` ✅  
**输入**：`data/processed/retrieval/corpus.jsonl`  
**依赖库**：`sentence-transformers`、`faiss-cpu`  
**模型**：`moka-ai/m3e-base`  
**输出**：
- FAISS 索引文件（.index）
- 向量嵌入文件（.npy）
- TopK 查询结果（JSON）

**实现**：`retrieval/retrievers.py` → `VectorRetriever`

**验收标准**：
- ✅ 支持对任意 query 输出 TopK（默认 K=10）
- ✅ 支持批量查询
- ✅ 索引构建可配置（维度、距离度量）
- ✅ 支持 GPU 加速，CPU fallback 正常

**依赖**：任务2

---

### 任务5：混合检索
**状态**：`completed` ✅  
**输入**：
- BM25 检索结果
- 向量检索结果

**策略**：RRF（Reciprocal Rank Fusion）+ 加权融合（alpha/beta 可配置）
**实现**：`retrieval/retrievers.py` → `HybridRetriever`

**验收标准**：
- ✅ 输出混合 TopK
- ✅ 支持权重配置（config.toml [retrieval]）
- ✅ 输出格式与单一检索方法一致
- ✅ 融合策略可扩展（支持 RRF、加权线性融合）

**依赖**：任务3、任务4

---

### 任务6：评测集与指标
**状态**：`completed` ✅  
**输入**：`data/evaluation/queries.jsonl`（支持全量生成、比例采样、固定数量三种模式）  
**输出**：`outputs/reports/retrieval_metrics.json`  
**指标**：
- ✅ Recall@K（K=1,3,5,10）
- ✅ MRR（Mean Reciprocal Rank）
- ✅ nDCG@K（K=3,5,10）
- ✅ MAP（Mean Average Precision）

**脚本**：`evaluation/evalRetrieval.py`、`generation/generateQueries.py`  

**验收标准**：
- ✅ 固定格式输出，结果可复现
- ✅ 支持对比多种检索方法（BM25、向量、混合）
- ✅ 评测查询自动生成脚本，支持多种采样模式
- ✅ 输出包含详细指标和统计信息

**依赖**：任务3、任务4、任务5

---

### 任务7：RAG 提示模板设计
**状态**：`completed` ✅ （Issue #23）  
**输入**：检索结果（TopK 术语与定义）、用户查询  
**输出**：`generation/promptTemplates.py`  

**验收标准**：
- ✅ 基础模板：system + user prompt，含检索上下文拼接
- ✅ 支持多条检索结果按 rank 排序拼接，控制总长度（MAX_CONTEXT_CHARS=2000）
- ✅ LaTeX 公式保持原样，来源字段（source、page）格式化为【书名 第X页】
- ✅ f-string 实现（buildPrompt/buildMessages）+ Jinja2 模板常量及 buildPromptJinja2()
- ✅ 空检索结果时退化为直接问答，不崩溃
- ✅ config.toml 新增 [generation] 节，config.py 新增 getGenerationConfig() 和 QWEN_MODEL_DIR

**依赖**：任务5、任务6

---

### 任务8：Qwen2.5-Math-1.5B 本地推理集成
**状态**：`completed` ✅ （Issue #24）
**输入**：本地模型路径（`Qwen-model/`，由 `config.QWEN_MODEL_DIR` 管理）
**输出**：`generation/qwenInference.py`

**验收标准**：
- ✅ 支持从本地路径加载模型（`transformers.AutoModelForCausalLM`）
- ✅ 封装 `generate(prompt, max_new_tokens)` 接口，支持单条和批量推理
- ✅ 支持 GPU 加速（`device_map="auto"`），CPU fallback 正常工作
- ✅ 推理参数可通过 `config.toml` 配置（temperature、top_p、max_new_tokens）

**依赖**：任务7

---

### 任务9：端到端 RAG 问答流程
**状态**：`completed` ✅ （Issue #25）
**输入**：用户查询、检索索引、Qwen 推理接口、提示模板
**输出**：
- `generation/ragPipeline.py`
- `scripts/runRag.py`
- `outputs/rag_results.jsonl`

**验收标准**：
- ✅ 单条查询：输入问题 → 输出含来源的结构化回答（定义 + 公式 + 出处）
- ✅ 批量查询：从文件读取，输出 JSONL 结果文件
- ✅ 检索策略可切换（BM25 / 向量 / 混合），通过参数指定
- ✅ 输出字段：`query`、`retrieved_terms`、`answer`、`sources`、`latency`
- ✅ 检索为空时不崩溃，给出提示

**依赖**：任务7、任务8

---

### 任务10：生成质量评估
**状态**：`completed` ✅ （Issue #26）
**输入**：`outputs/rag_results.jsonl`、`data/evaluation/queries.jsonl`
**输出**：`outputs/reports/generation_metrics.json`
**指标**：
- ✅ 术语命中率（回答包含目标相关术语）
- ✅ 来源引用率（书名/页码正确引用）
- ✅ 回答非空率
- ✅ （可选）BLEU / ROUGE

**脚本**：`evaluation/evalGeneration.py`
**依赖**：任务9

---

### 任务11：对比实验：RAG vs 无检索
**状态**：`completed` ✅ （Issue #27）
**实验组**：

| 实验组 | 检索策略 | RAG |
|--------|----------|-----|
| baseline-norag | 无 | ❌ |
| baseline-bm25 | BM25 | ✅ |
| baseline-vector | 向量检索 | ✅ |
| exp-hybrid | 混合检索（主实验） | ✅ |

**输出**：
- `outputs/reports/comparison_results.json`
- `outputs/reports/comparison_chart.png`（对比柱状图）
- `outputs/reports/comparison_table.md`
- `scripts/runExperiments.py`（一键运行所有实验组）
- `scripts/experimentWebUI.py`（可视化实验界面）

**验收标准**：
- ✅ 所有实验组使用相同测试集和模型，保证可比性
- ✅ 输出汇总表格（Markdown 格式），可直接入论文
- ✅ 实验配置通过 `config.toml` 管理，结果可复现
- ✅ 随机种子固定

**依赖**：任务9、任务10

---

### 任务12：黄金测试集构建
**状态**：`todo` （Issue #33）
**输入**：`data/processed/chunk/` 下的术语数据、人工标注
**输出**：`data/evaluation/golden_set.jsonl`

**验收标准**：
- [ ] 覆盖三个学科：数学分析、高等代数、概率论
- [ ] 每个学科至少 20 条高质量查询
- [ ] 标注规范统一，便于评测脚本解析
- [ ] 包含不同难度：简单定义查询、复杂推理查询

**依赖**：属于 Plan-6 评测体系子任务

---

### 任务13：检索指标实现
**状态**：`todo` （Issue #34）
**输入**：检索结果、黄金测试集
**输出**：`outputs/reports/retrieval_metrics.json`

**评测指标**：
- [ ] Recall@K（K=1, 3, 5, 10）
- [ ] MRR（Mean Reciprocal Rank）
- [ ] nDCG@K（K=3, 5, 10）
- [ ] MAP（Mean Average Precision）

**脚本**：`evaluation/evalRetrieval.py`
**依赖**：任务12

---

### 任务14：生成质量评估扩展
**状态**：`todo` （Issue #35）
**输入**：RAG 问答结果、黄金测试集
**输出**：`outputs/reports/generation_metrics_full.json`

**新增指标**：
- [ ] 答案完整性评分（是否包含定义、公式、来源）
- [ ] 语义相似度（与参考答案的 embedding 距离）
- [ ] 幻觉检测（是否引用了不存在的来源）
- [ ] 人工评测样本抽取

**依赖**：任务10、任务12

---

### 任务15：对比实验完善
**状态**：`todo` （Issue #36）
**输入**：现有实验结果
**输出**：
- `outputs/reports/ablation_study.json`
- `outputs/reports/significance_test.json`

**新增实验**：
- [ ] 消融实验：TopK 数量对性能的影响（K=1,3,5,10）
- [ ] 消融实验：混合检索权重敏感性（alpha=0.3/0.5/0.7）
- [ ] 统计显著性检验（t-test 或 bootstrap）
- [ ] （可选）Qwen2.5-Math-7B 对比

**依赖**：任务11

---

### 任务16：评测报告生成
**状态**：`todo` （Issue #37）
**输入**：所有评测结果
**输出**：
- `outputs/reports/final_report.md`
- `outputs/figures/`（论文级别图表）

**报告内容**：
- [ ] 实验设置总结（数据集、模型、参数）
- [ ] 检索性能对比表格
- [ ] 生成质量对比表格
- [ ] 消融实验结果图表
- [ ] 典型案例分析

**依赖**：任务13、任务14、任务15

---

## 已完成产出物
- `data/processed/retrieval/corpus.jsonl` - 检索语料库
- `data/stats/` - 数据统计报告与可视化
- `data/evaluation/queries.jsonl` - 评测查询集（105 条）
- `data/evaluation/term_mapping.json` - 评测术语映射
- `retrieval/buildCorpus.py` - 语料构建
- `retrieval/retrievers.py` - 7 种检索器（BM25/BM25+/向量/混合/HybridPlus/Reranker/Advanced）
- `retrieval/queryRewrite.py` - 查询改写（144 条同义词映射）
- `evaluation/evalRetrieval.py` - 检索评测
- `evaluation/evalGeneration.py` - 生成质量评测
- `generation/generateQueries.py` - 评测查询生成
- `evaluation/quickEval.py` - 快速检索评测
- `generation/promptTemplates.py` - RAG 提示模板
- `generation/qwenInference.py` - Qwen 推理封装
- `generation/ragPipeline.py` - 端到端 RAG 流程
- `generation/webui.py` - Gradio WebUI
- `scripts/runRag.py` - RAG 问答脚本
- `scripts/runExperiments.py` - 对比实验脚本
- `scripts/experimentWebUI.py` - 实验 WebUI
- `scripts/buildEvalTermMapping.py` - 评测术语映射构建
- `outputs/reports/` - 检索评测、生成评测、对比实验结果

## 待完成产出物
- `data/evaluation/golden_set.jsonl` - 黄金测试集（Task-12）
- `outputs/reports/ablation_study.json` - 消融实验结果（Task-15）
- `outputs/reports/significance_test.json` - 显著性检验（Task-15）
- `outputs/reports/final_report.md` - 最终评测报告（Task-16）
- `outputs/figures/` - 论文级别图表（Task-16）

## 风险与注意事项
- Qwen2.5-Math-1.5B 上下文窗口有限（约 4096 token），提示长度需严格控制
- 无检索 baseline 需固定随机种子，保证可对比性
- 若无 GPU，1.5B 模型在 CPU 下推理较慢，建议先小批量验证
