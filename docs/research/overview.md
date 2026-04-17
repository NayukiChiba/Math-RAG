# 研究线概述（research）

`src/research` 包含 Math-RAG 研究线的所有功能，用于论文实验、检索评测、生成质量评测与报告生成。

## 模块划分

| 模块 | 路径 | 说明 |
|------|------|------|
| 评测数据 | `research/evaluationData/` | 评测查询集生成 |
| 模型评测 | `research/modelEvaluation/` | 检索评测、生成评测、快速评测 |
| 数据统计 | `research/dataStat/` | 术语数据统计与可视化 |
| 运行器 | `research/runners/` | 实验编排（fullReports、publishReports 等） |
| CLI 入口 | `research/researchMain.py` | math-rag-research 命令行入口 |

## 评测输出

| 目录 | 说明 |
|------|------|
| `outputs/log/<run_id>/` | 单次评测完整痕迹 |
| `outputs/reports/` | 定稿区（final_report.md、figures/） |
| `outputs/figures/defense/` | 答辩演示图表 |

## 快速导航

- [评测数据](/research/evaluationData/index)
- [模型评测](/research/modelEvaluation/index)
- [数据统计](/research/dataStat/index)
- [运行器](/research/runners/runRag)
