# 答辩图表生成（generateDefenseFigures）

**路径**：`reports_generation/reports/generateDefenseFigures.py`

## 功能

自动生成答辩演示所需的 16 张图表，写入 `outputs/figures/defense/`。

## 包含图表

| 序号 | 文件名 | 内容 |
|------|--------|------|
| 01 | `01_terms_by_book.png` | 各教材术语数量 |
| 02 | `02_terms_by_subject.png` | 各学科术语分布 |
| 03 | `03_field_coverage.png` | 字段覆盖率 |
| 04 | `04_definitions_analysis.png` | 定义质量分析 |
| 05 | `05_term_length_dist.png` | 术语长度分布 |
| 06 | `06_field_length_comparison.png` | 字段长度对比 |
| 07 | `07_corpus_text_length.png` | 语料文本长度 |
| 08 | `08_query_difficulty.png` | 查询难度分布 |
| 09 | `09_retrieval_strategy_comparison.png` | 检索策略对比 |
| 10 | `10_hybrid_weight_sensitivity.png` | 混合权重敏感性 |
| 11 | `11_topk_recall_curve.png` | TopK 召回曲线 |
| 12 | `12_pipeline_stats.png` | 流水线统计 |
| 13 | `13_duplicate_analysis.png` | 重复分析 |
| 14 | `14_formula_related_terms.png` | 公式相关术语 |
| 15 | `15_corpus_heatmap.png` | 语料热力图 |
| 16 | `16_system_dashboard.png` | 系统仪表板 |

## 启动

```bash
python main.py research defense-figures
```
