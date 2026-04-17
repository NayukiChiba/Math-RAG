# 报告生成概述

`reports_generation/` 包含报告、快评与答辩图表生成功能。

## 功能

| 模块 | 说明 |
|------|------|
| `generateReport` | 生成完整 Markdown 评测报告与图表（PDF/PNG） |
| `generateDefenseFigures` | 生成答辩演示图表，写入 `outputs/figures/defense/` |
| `quick_eval/` | 快速检索评测（调试用，小数据集） |
| `viz/` | 可视化工具（方法对比图、消融图等） |

## 启动方式

```bash
# 生成最终报告
python main.py research report --results outputs/log/<run_id>/json/all_methods.json

# 生成答辩图表
python main.py research defense-figures

# 从已有 log 发布定稿
python main.py research publish-reports --run-id 20260406_164049
```

## 输出位置

- **定稿报告**：`outputs/reports/final_report.md`
- **图表**：`outputs/reports/figures/`（方法对比、消融、学科分解等）
- **答辩图**：`outputs/reports/figures/defense/`（16 张演示图）
- **JSON 快照**：`outputs/reports/json/`（主结果、消融、显著性）
