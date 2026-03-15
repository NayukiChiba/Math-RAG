# reporting.py

## 概述
`modelEvaluation/generationEval/reporting.py` 它是在已经从生成器评测完成对多指标度量提取完成后进行的一个高度复杂而且对外的关于生成层面的可视化或是结果整合归纳表生成功能。它能比如产成各种生成参数（`temperature` 不同）导致最终结果如准确性与流畅性波动的对开和排列综合分析或者是提供类似于 LaTeX 内结果或 HTML 表格以便于研究使用环境等视图报告综合实现包裹组件。