# 推荐工作流

## 产品线（日常使用）

```text
ingest → build-index → rag / ui
```

### 步骤详解

```bash
# 1. PDF 入库（首次或新增教材时执行）
python main.py cli ingest data/raw/数学分析.pdf

# 2. 重建索引（微调预处理参数后执行）
python main.py cli build-index --rebuild

# 3. RAG 问答
python main.py cli rag --query "什么是一致收敛？"

# 4. 图形界面
python main.py ui
```

## 研究线（论文实验）

```text
build-index → generate-queries → build-term-mapping
  → eval-retrieval → experiments → eval-generation → full-reports
```

### 步骤详解

```bash
# 1. 确保索引已构建
python main.py cli build-index

# 2. 生成评测查询集
python main.py research generate-queries

# 3. 构建评测术语映射
python main.py research build-term-mapping

# 4. 检索评测（生成对比图表）
python main.py research eval-retrieval --visualize \
  --queries data/evaluation/queries_full.jsonl

# 5. 全量评测总控（含消融与报告）
python main.py research full-reports --retrieval-only

# 6. 发布定稿到 outputs/reports/
python main.py research publish-reports --run-id 20260406_164049
```

## 输出目录

| 目录 | 说明 |
|------|------|
| `outputs/log/<run_id>/` | 单次评测完整痕迹（JSON、run_trace、exports） |
| `outputs/reports/` | 定稿区（final_report.md、figures/、json/） |
| `outputs/figures/defense/` | 答辩演示图表 |
