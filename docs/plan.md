# Math-RAG 毕业论文项目规划

本项目目标：构建面向数学名词的高精度搜索型 RAG 系统。

基础模型：
- `Qwen2.5-Math-1.5B-Instruct`（本地运行主力）
- `Qwen2.5-Math-7B-Instruct`（可选：有空或有算力时租服务器运行）

核心要求：
- 检索准确率优先
- 可复现、可对比的实验流程
- 代码清晰易读

## 项目范围与输出

研究问题：
- 数学名词场景下，RAG 结构对检索与回答的提升幅度有多大？
- 可选：如果具备算力，补充 7B 结果用于对比

名词范围（学科边界）：
- 数学分析
- 高等代数
- 概率论

数据来源：仅 OCR 教材数据，不包含维基百科路线。

名词定义格式（统一模板）：
- id: 唯一标识（建议稳定、可追踪）
- term: 名词（标准写法）
- aliases: 别名/同义词（可为空数组）
- sense_id: 义项编号（用于歧义词）
- subject: 学科（数学分析/高等代数/概率论）
- definitions: 定义列表（至少 1 条，允许多种定义）
  - type: 定义类型（strict/alternative/informal/operational）
  - text: 定义正文（严格、清晰；涉及数学符号必须用 LaTeX，建议用 $...$ 包裹）
  - conditions: 适用条件/前提（可为空）
  - notation: 记号/符号约定（可为空）
  - reference: 该定义的来源标识
- notation: 常用符号/记号（必须为 LaTeX，便于检索）
- formula: 相关公式/引理/定理（数组元素必须为 LaTeX）
- usage: 用法（常见表述或使用方式，1-3 句）
- applications: 用途/作用（理论用途或应用场景，1-3 句）
- disambiguation: 与相近名词的区分说明（可为空）
- related_terms: 相关名词（数组，用于扩展召回）
- sources: 参考来源列表（数组，至少 1 条）
- search_keys: 归一化检索键（数组，建议包含去空格/大小写/符号统一版本）
- lang: 语言标记（zh/en）
- confidence: 标注置信度（high/medium/low）

示例格式（JSON）：
```json
[
  {
    "id": "ma-uniform-convergence-001",
    "term": "一致收敛",
    "aliases": ["uniform convergence"],
    "sense_id": "1",
    "subject": "数学分析",
    "definitions": [
      {
        "type": "strict",
        "text": "函数列 $\\{f_n\\}$ 在集合 $E$ 上一致收敛于 $f$，若对任意 $\\epsilon>0$，存在 $N$，使得当 $n\\ge N$ 时，对所有 $x\\in E$ 有 $|f_n(x)-f(x)|<\\epsilon$。",
        "conditions": "E 为定义域的子集",
        "notation": "$f_n \\to f$ (uniformly on $E$)",
        "reference": "数学分析(第3版) 第7章"
      },
      {
        "type": "alternative",
        "text": "一致收敛等价于 Cauchy 一致性：对任意 $\\epsilon>0$，存在 $N$，使得对所有 $m,n\\ge N$ 有 $\\sup_{x\\in E}|f_n(x)-f_m(x)|<\\epsilon$。",
        "conditions": "E 上定义",
        "notation": "$\\sup$ 范数",
        "reference": "数学分析(第3版) 第7章"
      }
    ],
    "notation": "$\\|f_n - f\\|_\\infty \\to 0$",
    "formula": [
      "\\forall\\,\\epsilon>0\\,\\exists N\\,\\forall m,n\\ge N:\\sup_{x\\in E}|f_n(x)-f_m(x)|<\\epsilon",
      "\\left(|f_n(x)|\\le M_n\\right)\\wedge\\left(\\sum_{n=1}^{\\infty} M_n<\\infty\\right)\\Rightarrow\\sum_{n=1}^{\\infty} f_n(x)\\text{ 在 }E\\text{ 上一致收敛}",
      "\\left(f_n\\text{ 连续且 }f_n\\to f\\text{ 一致收敛}\\right)\\Rightarrow f\\text{ 连续}"
    ],
    "usage": "用于判断函数列/级数的收敛方式，常见于交换极限与积分、求和的条件讨论。",
    "applications": "保证极限函数的连续性或可积性；用于证明级数逐项积分/微分的合法性。",
    "disambiguation": "区别于逐点收敛：一致收敛在整个集合上具有统一的收敛速度。",
    "related_terms": ["逐点收敛", "一致连续", "Weierstrass判别法"],
    "sources": [
      "数学分析(第3版) 第7章"
    ],
    "search_keys": ["一致收敛", "一致 收敛", "uniform convergence", "uniformconvergence"],
    "lang": "zh",
    "confidence": "high"
  }
]
```

交付物：
- 可运行的 RAG 系统（训练/构建索引 + 推理/检索 + 生成）
- 可复现实验结果与报告
- 论文与实验记录

## 系统构成（高层）

- 数据层：数学名词及定义/解释/公式的语料与结构化索引
- 检索层：稀疏检索 / 向量检索 / 混合检索
- 重排层：交叉编码器或 LLM 重排（可选）
- 生成层：Qwen2.5-Math-1.5B/7B + RAG 提示
- 评测层：检索指标 + 生成质量指标

## 目录规划建议

- `docs/`：论文与实验规划、里程碑、设计说明
- `data/`：原始数据与处理后的数据
- `src/`：核心实现
- `configs/`：可复现实验配置
- `scripts/`：数据处理、建索引、评测脚本
- `outputs/`：实验记录与结果输出

## 构建步骤（路线图）

### 任务1：任务定义与评测标准（✅ 已完成）
- 目标：明确“数学名词”范围、检索目标与评测指标
- 输出：统一的任务定义与评测口径

### 任务2：数据准备（✅ 已完成）
- 目标：构建可复现的检索语料与黄金集
- 输入：OCR 教材产出的术语与定义数据
- 输出：统一语料与黄金集
- 子任务：
  - ✅ 数据核验与统计（dataStat/chunkStatistics.py）
  - ✅ 构建检索语料（retrieval/buildCorpus.py → data/processed/retrieval/corpus.jsonl）
  - ✅ 评测查询集构建（evaluation/generateQueries.py → data/evaluation/queries.jsonl）

### 任务3：教材 OCR + LLM 构建数学名词数据（✅ 已完成）
- 目标：遍历 `data/raw/` 下的 PDF 教材，OCR 后生成数学相关术语与结构化 JSON
- 输入：`data/raw/` 下的多本 PDF
- 输出：
  - `data/processed/ocr/<书名>/pages/*.md`（分页 OCR）
  - `data/processed/ocr/<书名>/terms_map.json`（术语-页码映射）
  - `data/processed/ocr/terms_json_all.json`（跨书籍聚合 JSON）
  - `data/processed/chunk/<书名>/<术语>.json`（术语级 JSON 切分）
- 约束：术语仅限数学相关，JSON 需标注书名与页码来源
- **已完成**：
  - ✅ 配置管理系统（config.py + config.toml）
  - ✅ OCR 流程脚本（pix2text_ocr.py, ocr_to_json_all.py）
  - ✅ 术语提取脚本（extract_terms_from_ocr.py）
  - ✅ JSON 生成脚本（data_gen.py）
  - ✅ 术语 JSON 切分（输出到 `data/processed/chunk/`）
  - ✅ 数学分析(第5版) 上/下 OCR 处理
- **使用方法**：
  ```bash
  # 配置 config.toml 中的 [ocr] 和 [model] 部分
  # 将 PDF 放入 data/raw/ 目录
  python scripts/ocr_to_json_all.py
  ```

### 任务4：检索层构建（✅ 已完成）
- 目标：实现多种检索策略，支持可插拔切换
- 子任务：
  - ✅ BM25 稀疏检索基线（retrieval/retrievalBM25.py）
  - ✅ 向量检索（sentence-transformers + FAISS，retrieval/retrievalVector.py）
  - ✅ 混合检索（RRF 策略，retrieval/retrievalHybrid.py）
  - ✅ 统一检索接口设计（retrieval/__init__.py）
- 输入：`data/processed/retrieval/corpus.jsonl`
- 输出：`retrieval/` 检索模块，索引文件保存至 `outputs/`

### 任务5：RAG 生成层（🔄 进行中）
- 目标：集成 Qwen2.5-Math 模型，实现检索增强生成
- 子任务：
  - [ ] 提示模板设计（Task-7 #23）
  - [ ] Qwen2.5-Math-1.5B 本地推理集成（Task-8 #24）
  - [ ] 端到端问答流程（Task-9 #25）
  - [ ] （可选）Qwen2.5-Math-7B 对比实验
- 输出：`generation/` 生成模块

### 任务6：评测体系（🔄 进行中）
- 目标：构建可复现的评测流程
- 子任务：
  - ✅ 检索指标实现（Recall@K, MRR, nDCG，evaluation/evalRetrieval.py）
  - ✅ 评测查询集自动生成（evaluation/generateQueries.py）
  - [ ] 生成质量评估（Task-10 #26）
  - [ ] 对比实验（RAG vs 无检索，Task-11 #27）
- 输出：`evaluation/` 评测模块，结果保存至 `outputs/reports/`

---

## 详细步骤

### 1. 明确任务定义与评测标准
- 定义"数学名词"范围与输出格式
- 定义检索目标（定义、公式、出处、上下文）
- 确定评测指标：
  - 检索：Recall@K、MRR、nDCG
  - 生成：准确性（人工 + 规则）、引用命中率

### 2. 数据准备
- 收集语料：OCR 教材产出的结构化术语与定义
- 建立“名词-定义”标准集合（黄金集）
- 预处理：
  - 清洗、去重、分段、结构化
  - 统一格式（title / definition / formula / source）

### 3. 索引与检索基线
- 稀疏检索（BM25）基线
- 向量检索（embedding + ANN）
- 混合检索（稀疏 + 向量）
- 统一接口，保证可插拔

### 4. RAG 生成与提示工程
- 基础提示模板（定义、公式、来源引用）
- 检索结果拼接策略（长度、去重、排序）
- 1.5B 为主；7B 视算力情况补充

### 5. 重排与优化（可选）
- 引入交叉编码器或 LLM 评分
- 与无重排方案对比

### 6. 评测与对比实验
- 检索层对比：BM25 vs 向量 vs 混合
- 模型对比：1.5B 为主；7B 可选
- RAG 对比：无检索 vs 有检索
- 统计显著性（如适用）

### 7. 结果整理与论文撰写
- 记录每个实验配置与结果
- 形成对比表格与图表
- 分析误差类型与改进方向

## 里程碑建议（可调整）

- ✅ M1：数据集与黄金集完成
- ✅ M2：检索基线完成（BM25 + 向量 + 混合，含检索评测）
- 🔄 M3：RAG 生成完成（Task-7~9，进行中）
- 🔄 M4：评测体系完成（Task-10~11，进行中）
- M5：论文完成（如有 7B 结果则补充对比）

## 风险与注意事项

- 数据版权与来源合法性
- 名词定义的歧义与多版本
- 小模型可能需要更严格的检索质量控制
- 结果复现依赖随机种子与配置记录
